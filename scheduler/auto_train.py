from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

from core.features.builder import FeatureBuilder
from core.model.predictor import Predictor
from core.model.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class AutoTrainer:
    MARKET_FEATURES = {
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
    }

    def __init__(self, predictor: Predictor | None = None, model_dir: str | Path = "data/models") -> None:
        self.last_train_time: datetime | None = None
        self.predictor = predictor or Predictor()
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "model.pkl"
        self.schema_path = self.model_dir / "feature_schema.json"
        self.calibrated_model_path = self.model_dir / "calibrated_model.pkl"

        self.model_file_exists: bool = False
        self.feature_schema_loaded: bool = False
        self.models_loaded: bool = False
        self.predictor_trained: bool = False
        # Optional Database reference for runtime DB stats lookups (shots, possession, xg, draw_pct, home_advantage).
        # Set externally via trainer.db = database after construction.
        self.db: Any | None = None
        self._refresh_runtime_state()

    def should_run(self) -> bool:
        if self.last_train_time is None:
            return True
        return datetime.utcnow() - self.last_train_time >= timedelta(hours=24)

    def mark_done(self) -> None:
        self.last_train_time = datetime.utcnow()
        logger.info("Auto training completed at %s", self.last_train_time.isoformat())

    def train(self, dataset: list[dict[str, Any]], calibrate: bool = False) -> Any:
        """Train model through predictor runtime and update orchestration state."""
        model = self.predictor.train(
            dataset=dataset,
            model_path=self.model_path,
            calibrate=calibrate,
        )
        self.mark_done()
        self._refresh_runtime_state()
        return model

    def train_from_db(
        self,
        database: Any,
        calibrate: bool = False,
        predict_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Train model from SQLite and return compact quality summary.

        Returns blocks:
        - dataset_summary
        - target_distribution
        - feature_quality_summary
        - weak_features
        - predict_quality_summary
        - warnings
        """
        try:
            logger.info("train_from_db: starting training from SQLite database")

            completed_matches = database.get_completed_matches() if hasattr(database, "get_completed_matches") else []
            completed_matches_found = len(completed_matches) if isinstance(completed_matches, list) else 0
            db_audit = self._build_db_audit_summary(database)

            dataset = database.build_training_dataset_from_db()
            dataset_debug = getattr(database, "last_training_dataset_debug", {})
            if not isinstance(dataset_debug, dict):
                dataset_debug = {}

            raw_completed_rows = int(dataset_debug.get("raw_completed_rows", completed_matches_found) or 0)
            unique_completed_matches = int(dataset_debug.get("unique_completed_matches", completed_matches_found) or 0)
            rows_after_feature_build = int(dataset_debug.get("rows_after_feature_build", len(dataset)) or 0)
            rows_after_dedup = int(dataset_debug.get("rows_after_dedup", len(dataset)) or 0)
            duplicate_rows_removed = int(dataset_debug.get("duplicate_rows_removed", 0) or 0)

            if not dataset:
                msg = "No trainable rows were built from DB"
                logger.warning("train_from_db: %s", msg)
                return {
                    "status": "empty",
                    "message": msg,
                    "completed_matches_found": completed_matches_found,
                    "training_rows": 0,
                    "dropped_rows": 0,
                    "dataset_summary": {
                        "completed_matches_found": completed_matches_found,
                        "raw_completed_rows": raw_completed_rows,
                        "unique_completed_matches": unique_completed_matches,
                        "dataset_rows_raw": 0,
                        "rows_after_feature_build": rows_after_feature_build,
                        "rows_after_dedup": rows_after_dedup,
                        "duplicate_rows_removed": duplicate_rows_removed,
                        "training_rows": 0,
                        "final_train_rows": 0,
                        "dropped_rows": 0,
                        "draw_rows_in_dataset": 0,
                        "dataset_minus_completed": 0,
                        "dataset_expansion_reason": "no rows",
                    },
                    "db_input_summary": db_audit,
                    "used_feature_set": {
                        "training_feature_mode": "none",
                        "active_feature_count": 0,
                        "extended_ready": False,
                        "baseline_ready": False,
                    },
                    "train_debug_report": {},
                    "validation_summary": self._build_validation_summary_not_run("dataset is empty"),
                    "target_distribution": {"home_win_count": 0, "draw_count": 0, "away_win_count": 0},
                    "feature_quality_summary": {
                        "feature_count": 0,
                        "top_healthy_features": [],
                        "top_weak_features": [],
                    },
                    "weak_features": [],
                    "predict_quality_summary": self._build_predict_quality_summary(predict_summary),
                    "warnings": [
                        "no dataset rows were built from DB",
                        "pipeline is technical-only until completed matches are available",
                    ],
                    "model_file_exists": self.model_file_exists,
                    "feature_schema_loaded": self.feature_schema_loaded,
                    "models_loaded": self.models_loaded,
                    "predictor_trained": self.predictor_trained,
                }

            trainer = ModelTrainer(model_path=str(self.model_path))
            if hasattr(trainer, "clean_data_with_report"):
                cleaned_rows, clean_report = trainer.clean_data_with_report(dataset)
            else:
                cleaned_rows = trainer.clean_data(dataset)
                clean_report = {
                    "training_rows_before_clean": len(dataset),
                    "training_rows_after_clean": len(cleaned_rows),
                    "training_feature_mode": "unknown",
                    "active_feature_count": len(trainer.feature_columns),
                    "required_features_missing_top": [],
                    "sample_filtered_rows_reasons": [],
                }

            dataset_rows_raw = len(dataset)
            training_rows = len(cleaned_rows)
            dropped_rows = max(dataset_rows_raw - training_rows, 0)
            draw_rows_in_dataset = sum(1 for row in dataset if int(row.get("target", -1)) == 0)
            dataset_minus_completed = dataset_rows_raw - completed_matches_found
            final_train_rows = training_rows

            dataset_summary = {
                "completed_matches_found": completed_matches_found,
                "raw_completed_rows": raw_completed_rows,
                "unique_completed_matches": unique_completed_matches,
                "dataset_rows_raw": dataset_rows_raw,
                "rows_after_feature_build": rows_after_feature_build,
                "rows_after_dedup": rows_after_dedup,
                "duplicate_rows_removed": duplicate_rows_removed,
                "training_rows": training_rows,
                "final_train_rows": final_train_rows,
                "dropped_rows": dropped_rows,
                "draw_rows_in_dataset": draw_rows_in_dataset,
                "dataset_minus_completed": dataset_minus_completed,
                "dataset_expansion_reason": (
                    "build_training_dataset_from_db may include draw rows; "
                    "completed_matches_found may differ from unique completed set"
                ),
            }

            used_feature_set = {
                "training_feature_mode": str(clean_report.get("training_feature_mode", "unknown")),
                "active_feature_count": int(clean_report.get("active_feature_count", len(trainer.feature_columns)) or 0),
                "active_feature_columns": list(trainer.feature_columns),
                "extended_ready": bool(clean_report.get("extended_candidate_rows_after_clean", 0) > 0),
                "baseline_ready": bool(clean_report.get("baseline_candidate_rows_after_clean", 0) > 0),
                "extended_candidate_rows_after_clean": int(clean_report.get("extended_candidate_rows_after_clean", 0) or 0),
                "baseline_candidate_rows_after_clean": int(clean_report.get("baseline_candidate_rows_after_clean", 0) or 0),
            }

            target_distribution = self._build_target_distribution(cleaned_rows)
            feature_profile = self._build_feature_profile(cleaned_rows, trainer.feature_columns)
            weak_features = feature_profile["weak_features"]
            healthy_features = feature_profile["healthy_features"]
            warnings = self._build_quality_warnings(
                completed_matches_found=completed_matches_found,
                dataset_rows_raw=dataset_rows_raw,
                training_rows=training_rows,
                target_distribution=target_distribution,
                weak_features=weak_features,
                predict_summary=predict_summary,
            )
            validation_summary = self._run_time_based_validation(trainer, cleaned_rows)

            if training_rows == 0:
                msg = "All rows were filtered before training (clean_data produced empty dataset)"
                logger.warning("train_from_db: %s", msg)
                return {
                    "status": "error",
                    "message": msg,
                    "completed_matches_found": completed_matches_found,
                    "training_rows": 0,
                    "dropped_rows": dropped_rows,
                    "dataset_summary": dataset_summary,
                    "db_input_summary": db_audit,
                    "used_feature_set": used_feature_set,
                    "train_debug_report": clean_report,
                    "validation_summary": validation_summary,
                    "target_distribution": target_distribution,
                    "feature_quality_summary": {
                        "feature_count": len(trainer.feature_columns),
                        "top_healthy_features": healthy_features,
                        "top_weak_features": weak_features,
                    },
                    "weak_features": weak_features,
                    "predict_quality_summary": self._build_predict_quality_summary(predict_summary),
                    "warnings": warnings,
                    "model_file_exists": self.model_file_exists,
                    "feature_schema_loaded": self.feature_schema_loaded,
                    "models_loaded": self.models_loaded,
                    "predictor_trained": self.predictor_trained,
                    "error": msg,
                }

            try:
                self.train(dataset=cleaned_rows, calibrate=calibrate)
            except ValueError as exc:
                error_msg = str(exc)
                logger.warning("train_from_db: training ValueError: %s", error_msg)
                return {
                    "status": "error",
                    "message": f"Training failed: {error_msg}",
                    "completed_matches_found": completed_matches_found,
                    "training_rows": training_rows,
                    "dropped_rows": dropped_rows,
                    "dataset_summary": dataset_summary,
                    "db_input_summary": db_audit,
                    "used_feature_set": used_feature_set,
                    "train_debug_report": clean_report,
                    "validation_summary": validation_summary,
                    "target_distribution": target_distribution,
                    "feature_quality_summary": {
                        "feature_count": len(trainer.feature_columns),
                        "top_healthy_features": healthy_features,
                        "top_weak_features": weak_features,
                    },
                    "weak_features": weak_features,
                    "predict_quality_summary": self._build_predict_quality_summary(predict_summary),
                    "warnings": warnings,
                    "model_file_exists": self.model_file_exists,
                    "feature_schema_loaded": self.feature_schema_loaded,
                    "models_loaded": self.models_loaded,
                    "predictor_trained": self.predictor_trained,
                    "error": error_msg,
                }
            except Exception as exc:
                error_msg = str(exc)
                logger.exception("train_from_db: unexpected training error")
                return {
                    "status": "error",
                    "message": f"Training failed unexpectedly: {error_msg}",
                    "completed_matches_found": completed_matches_found,
                    "training_rows": training_rows,
                    "dropped_rows": dropped_rows,
                    "dataset_summary": dataset_summary,
                    "db_input_summary": db_audit,
                    "used_feature_set": used_feature_set,
                    "train_debug_report": clean_report,
                    "validation_summary": validation_summary,
                    "target_distribution": target_distribution,
                    "feature_quality_summary": {
                        "feature_count": len(trainer.feature_columns),
                        "top_healthy_features": healthy_features,
                        "top_weak_features": weak_features,
                    },
                    "weak_features": weak_features,
                    "predict_quality_summary": self._build_predict_quality_summary(predict_summary),
                    "warnings": warnings,
                    "model_file_exists": self.model_file_exists,
                    "feature_schema_loaded": self.feature_schema_loaded,
                    "models_loaded": self.models_loaded,
                    "predictor_trained": self.predictor_trained,
                    "error": error_msg,
                }

            logger.info("train_from_db: training succeeded. rows_raw=%s rows_train=%s", dataset_rows_raw, training_rows)
            post_train_backtest = self._run_post_train_backtest(
                raw_rows=dataset,
                feature_columns=list(trainer.feature_columns),
                train_ratio=0.8,
            )
            return {
                "status": "success",
                "message": "Model trained successfully from SQLite database",
                "completed_matches_found": completed_matches_found,
                "training_rows": training_rows,
                "dropped_rows": dropped_rows,
                "dataset_summary": dataset_summary,
                "db_input_summary": db_audit,
                "used_feature_set": used_feature_set,
                "train_debug_report": clean_report,
                "validation_summary": validation_summary,
                "post_train_backtest": post_train_backtest,
                "target_distribution": target_distribution,
                "feature_quality_summary": {
                    "feature_count": len(trainer.feature_columns),
                    "top_healthy_features": healthy_features,
                    "top_weak_features": weak_features,
                },
                "weak_features": weak_features,
                "predict_quality_summary": self._build_predict_quality_summary(predict_summary),
                "warnings": warnings,
                "model_file_exists": self.model_file_exists,
                "feature_schema_loaded": self.feature_schema_loaded,
                "models_loaded": self.models_loaded,
                "predictor_trained": self.predictor_trained,
            }

        except Exception as exc:
            error_msg = str(exc)
            logger.exception("train_from_db: unexpected outer error")
            return {
                "status": "error",
                "message": f"Unexpected error: {error_msg}",
                "completed_matches_found": 0,
                "training_rows": 0,
                "dropped_rows": 0,
                "dataset_summary": {
                    "completed_matches_found": 0,
                    "dataset_rows_raw": 0,
                    "training_rows": 0,
                    "dropped_rows": 0,
                    "draw_rows_in_dataset": 0,
                    "dataset_minus_completed": 0,
                    "dataset_expansion_reason": "unexpected training error",
                },
                "db_input_summary": {},
                "used_feature_set": {
                    "training_feature_mode": "unknown",
                    "active_feature_count": 0,
                    "extended_ready": False,
                    "baseline_ready": False,
                },
                "train_debug_report": {},
                "validation_summary": self._build_validation_summary_not_run("unexpected training error"),
                "target_distribution": {"home_win_count": 0, "draw_count": 0, "away_win_count": 0},
                "feature_quality_summary": {
                    "feature_count": 0,
                    "top_healthy_features": [],
                    "top_weak_features": [],
                },
                "weak_features": [],
                "predict_quality_summary": self._build_predict_quality_summary(predict_summary),
                "warnings": ["unexpected training error"],
                "model_file_exists": self.model_file_exists,
                "feature_schema_loaded": self.feature_schema_loaded,
                "models_loaded": self.models_loaded,
                "predictor_trained": self.predictor_trained,
                "error": error_msg,
            }

    def load(self) -> None:
        """Load model artifacts through predictor runtime and update orchestration state."""
        self.predictor.load(self.model_dir)
        self._refresh_runtime_state()

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        """Run single prediction through predictor runtime.

        Automatically attempts artifact load when files exist but runtime is not ready.
        """
        if not self.predictor.is_ready and self.model_path.exists() and self.schema_path.exists():
            self.predictor.load(self.model_dir)
            self._refresh_runtime_state()

        return self.predictor.predict(features)

    def predict_batch(self, rows: list[dict[str, float]]) -> list[dict[str, float]]:
        """Run batch prediction through predictor runtime."""
        return [self.predict(row) for row in rows]

    def prepare_toto_match_for_inference(
        self,
        match: dict[str, Any],
        pool_probs: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        TOTO probability bridge with explicit 5-tier routing:

                    Tier 1: model               – full schema feature vector + predictor.predict().
                    Tier 1B: no_odds_fallback   – predictor.predict_with_diagnostics() without odds.
                    Tier 2: provided_match_probs – precomputed model probabilities in payload.
                    Tier 3: implied_from_odds   – odds present → compute implied_prob_1/x/2, use them.
                    Tier 4: bookmaker_market    – TotoBrief bk_win_1/bk_draw/bk_win_2 available.
                    Tier 5: pool_context_only   – last resort when no market input is available.

        Always returns a dict with:
          - probs              : best available {P1, PX, P2} (normalised to sum 1)
          - source             : legacy compatibility source
          - source_label       : runtime source label for diagnostics
          - mode_name          : human-readable routing label
          - model_probs        : non-None only when Tier 1 succeeded
          - bookmaker_probs    : non-None when bk quotes were usable
          - implied_probs      : non-None when odds were present and parseable
          - pool_probs         : normalised pool context (always set)
          - fallback_reason    : why Tier 1 was skipped, or None on model success
          - market_inputs_available : dict with detected inputs
        """
        pool_probs_resolved = self._resolve_pool_probs(match, pool_probs)
        provided_validation = self._validate_provided_probs(match)
        provided_probs = provided_validation.get("accepted_probs")
        rejected_reason = str(provided_validation.get("rejected_reason") or "")
        market_inputs = self._extract_market_inputs(match)

        odds_status = str(match.get("odds_status", "")).strip().lower() if isinstance(match, dict) else ""
        invalid_odds_status_count = 1 if odds_status in {"odds_missing", "odds_partial"} else 0
        rejected_distribution: dict[str, int] = {}
        if rejected_reason:
            rejected_distribution[rejected_reason] = 1

        result: dict[str, Any] = {
            "model_probs": None,
            "bookmaker_probs": None,
            "implied_probs": None,
            "pool_probs": pool_probs_resolved,
            "probs": pool_probs_resolved,
            "source": "pool_context_only",
            "source_label": "hard_fallback",
            "mode_name": "pool_only_fallback",
            "fallback_reason": None,
            "market_inputs_available": market_inputs,
            "provided_probs": provided_probs,
            "provided_probs_validation": provided_validation,
            "provided_match_probs_used_count": 0,
            "rejected_provided_probs_count": 1 if rejected_reason else 0,
            "rejected_reasons_distribution": rejected_distribution,
            "trusted_precomputed_count": 0,
            "runtime_model_count": 0,
            "no_odds_fallback_count": 0,
            "no_odds_skipped_count": 0,
            "no_odds_skipped_reasons": {},
            "hard_fallback_count": 0,
            "invalid_odds_status_count": invalid_odds_status_count,
        }

        # Compute implied probs from odds eagerly – available in all tiers, not just Tier 1.
        implied_probs_from_odds: dict[str, float] | None = None
        if market_inputs["has_odds"]:
            try:
                normalized = self._normalize_odds(
                    market_inputs["odds_ft_1"],
                    market_inputs["odds_ft_x"],
                    market_inputs["odds_ft_2"],
                )
                implied_probs_from_odds = {
                    "P1": normalized["implied_prob_1"],
                    "PX": normalized["implied_prob_x"],
                    "P2": normalized["implied_prob_2"],
                }
                result["implied_probs"] = implied_probs_from_odds
            except Exception:
                pass

        # Compute bookmaker-derived probs from TotoBrief bk quotes if available.
        bookmaker_probs: dict[str, float] | None = None
        if market_inputs["has_bookmaker_quotes"]:
            bookmaker_probs = self._probs_from_bookmaker_quotes(market_inputs)
            result["bookmaker_probs"] = bookmaker_probs

        # ── TIER 1: model_based ─────────────────────────────────────────────────
        # Requires: predictor ready + odds present (implied_prob features need odds).
        model_skip_reasons: list[str] = []

        if not market_inputs["has_odds"]:
            model_skip_reasons.append("no odds in payload (implied_prob schema features require odds)")

        if not model_skip_reasons:
            if not self.predictor.is_ready:
                if self.model_path.exists() and self.schema_path.exists():
                    try:
                        self.predictor.load(self.model_dir)
                        self._refresh_runtime_state()
                    except Exception as exc:
                        model_skip_reasons.append(f"model load failed: {exc}")
                else:
                    model_skip_reasons.append("model artifacts unavailable")

        feature_columns: list[str] | None = None
        if not model_skip_reasons:
            feature_columns = getattr(self.predictor, "feature_columns", None)
            if not feature_columns:
                model_skip_reasons.append("model feature schema missing")

        # Always build runtime feature snapshot for diagnostics and TOTO optimizer payload.
        resolved_columns = self._resolve_runtime_feature_columns(feature_columns)
        runtime_feature_snapshot = self.build_runtime_feature_snapshot(
            match=match,
            pool_probs=pool_probs_resolved,
            required_columns=resolved_columns,
        )
        result["runtime_features"] = runtime_feature_snapshot.get("features", {})
        result["feature_context_level"] = runtime_feature_snapshot.get("context_level", "degraded_context")
        result["feature_context"] = runtime_feature_snapshot

        if not model_skip_reasons and feature_columns is not None:
            features_result = runtime_feature_snapshot.get("features")
            if not isinstance(features_result, dict) or not features_result:
                model_skip_reasons.append(str(runtime_feature_snapshot.get("reason") or "runtime feature snapshot is empty"))
            else:
                try:
                    model_prediction = self.predict(features_result)  # type: ignore[arg-type]
                    model_probs = self._normalize_probs(model_prediction)
                    result.update({
                        "model_probs": model_probs,
                        "probs": model_probs,
                        "source": "model",
                        "source_label": "model_runtime",
                        "mode_name": "model_based",
                        "fallback_reason": None,
                        "runtime_model_count": 1,
                    })
                    return result
                except Exception as exc:
                    model_skip_reasons.append(f"prediction error: {exc}")

        # Record consolidated reason why Tier 1 was skipped.
        if model_skip_reasons:
            result["fallback_reason"] = "; ".join(model_skip_reasons)

        # ── TIER 1B: no_odds_fallback via predictor diagnostics ──────────────────
        # For rows without odds, give predictor a chance to infer from non-odds
        # features before dropping to hard market/public fallbacks.
        if not market_inputs["has_odds"]:
            no_odds_skip_reason: str | None = None
            if not self.predictor.is_ready:
                if self.model_path.exists() and self.schema_path.exists():
                    try:
                        self.predictor.load(self.model_dir)
                        self._refresh_runtime_state()
                    except Exception as exc:
                        no_odds_skip_reason = f"model load failed: {exc}"
                else:
                    no_odds_skip_reason = "model artifacts unavailable"

            feature_columns = getattr(self.predictor, "feature_columns", None)
            if no_odds_skip_reason is None and not feature_columns:
                no_odds_skip_reason = "model feature schema missing"

            if no_odds_skip_reason is None and feature_columns:
                diag_features = self._build_runtime_features_for_diagnostics(
                    match=match,
                    required_columns=feature_columns,
                    market_inputs=market_inputs,
                    implied_probs=implied_probs_from_odds,
                )
                try:
                    diagnostics_result = self.predictor.predict_with_diagnostics(
                        diag_features,
                        allow_no_odds_fallback=True,
                    )
                except Exception as exc:
                    diagnostics_result = {
                        "status": "skipped",
                        "reason": f"predict_with_diagnostics error: {exc}",
                        "probs": None,
                    }

                diag_status = str(diagnostics_result.get("status", "")).strip().lower()
                diag_reason = str(diagnostics_result.get("reason") or "Нет кф, fallback-прогноз невозможен")
                diag_probs = diagnostics_result.get("probs")
                if diag_status == "predicted" and isinstance(diag_probs, dict):
                    model_probs = self._normalize_probs(diag_probs)
                    result.update({
                        "model_probs": model_probs,
                        "probs": model_probs,
                        "source": "no_odds_fallback",
                        "source_label": "no_odds_fallback",
                        "mode_name": "no_odds_fallback",
                        "fallback_reason": None,
                        "no_odds_fallback_count": 1,
                        "runtime_model_count": 0,
                    })
                    return result

                no_odds_skip_reason = diag_reason
                result["no_odds_skipped_count"] = 1
                result["no_odds_skipped_reasons"] = {diag_reason: 1}
                result["fallback_reason"] = diag_reason
                rejected_distribution[diag_reason] = rejected_distribution.get(diag_reason, 0) + 1
                result["rejected_reasons_distribution"] = rejected_distribution

            if no_odds_skip_reason is not None and result.get("no_odds_skipped_count", 0) == 0:
                result["no_odds_skipped_count"] = 1
                result["no_odds_skipped_reasons"] = {no_odds_skip_reason: 1}
                result["fallback_reason"] = no_odds_skip_reason
                rejected_distribution[no_odds_skip_reason] = rejected_distribution.get(no_odds_skip_reason, 0) + 1
                result["rejected_reasons_distribution"] = rejected_distribution

        # ── TIER 2: provided_match_probs ─────────────────────────────────────────
        if provided_probs is not None:
            result.update({
                "probs": provided_probs,
                "source": "provided_match_probs",
                "source_label": "trusted_precomputed",
                "mode_name": "provided_probs_based",
                "provided_match_probs_used_count": 1,
                "trusted_precomputed_count": 1,
            })
            return result

        if rejected_reason:
            result["source_label"] = "rejected_precomputed"

        # ── TIER 3: implied_from_odds ────────────────────────────────────────────
        if implied_probs_from_odds is not None:
            result.update({
                "probs": implied_probs_from_odds,
                "source": "implied_from_odds",
                "source_label": "implied_only",
                "mode_name": "implied_based",
            })
            return result

        # ── TIER 4: bookmaker_market ─────────────────────────────────────────────
        if bookmaker_probs is not None:
            result.update({
                "probs": bookmaker_probs,
                "source": "bookmaker_market",
                "source_label": "bookmaker_only",
                "mode_name": "bookmaker_market_based",
            })
            return result

        # ── TIER 5: pool_context_only ────────────────────────────────────────────
        if not result["fallback_reason"]:
            result["fallback_reason"] = "no odds and no bookmaker quotes found in payload"
        result["source_label"] = self._derive_fallback_source_label(match, rejected_reason)
        result["hard_fallback_count"] = 1
        return result

    def build_runtime_feature_snapshot(
        self,
        match: dict[str, Any],
        pool_probs: dict[str, float] | None = None,
        required_columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build a shared runtime feature payload for MATCHES/TOTO live paths.

        Returns strict-schema-ready features plus context-level diagnostics.
        """
        columns = self._resolve_runtime_feature_columns(required_columns)
        if not columns:
            return {
                "features": {},
                "required_columns": [],
                "context_level": "degraded_context",
                "available_non_market_count": 0,
                "total_non_market_count": 0,
                "fallback_non_market_count": 0,
                "missing_critical_features": ["feature_schema"],
                "status": "error",
                "reason": "feature schema unavailable",
            }

        pool = self._resolve_pool_probs(match, pool_probs)
        market_inputs = self._extract_market_inputs(match)
        payload_numeric = self._collect_payload_numeric_values(match, columns)

        features_result, build_reason, source_meta = self._build_match_features_with_meta(match, columns, pool)
        if features_result is None:
            missing_feature_names = [
                col for col in columns
                if col in ("odds_ft_1", "odds_ft_x", "odds_ft_2", "implied_prob_1", "implied_prob_x", "implied_prob_2")
            ]
            missing_non_market_names = [col for col in missing_feature_names if col not in self.MARKET_FEATURES]
            return {
                "features": {},
                "required_columns": list(columns),
                "context_level": "degraded_context",
                "feature_context_level": "degraded_context",
                "available_non_market_count": 0,
                "total_non_market_count": len([c for c in columns if c not in self.MARKET_FEATURES]),
                "fallback_non_market_count": len([c for c in columns if c not in self.MARKET_FEATURES]),
                "missing_feature_count": len(missing_feature_names),
                "missing_non_market_count": len(missing_non_market_names),
                "degraded_context_flag": True,
                "market_only_or_market_heavy_flag": True,
                "available_feature_names": [],
                "missing_feature_names": missing_feature_names,
                "missing_non_market_names": missing_non_market_names,
                "missing_critical_features": [
                    col for col in ("odds_ft_1", "odds_ft_x", "odds_ft_2", "implied_prob_1", "implied_prob_x", "implied_prob_2")
                    if col in columns
                ],
                "status": "error",
                "reason": str(build_reason or "runtime feature build failed"),
                "market_inputs": market_inputs,
            }

        features: dict[str, float] = features_result
        non_market_cols = [col for col in columns if col not in self.MARKET_FEATURES]
        available_non_market_names = [
            col for col in non_market_cols
            if str(source_meta.get(col, "")).startswith("payload")
            or str(source_meta.get(col, "")).startswith("derived")
            or str(source_meta.get(col, "")).startswith("db_lookup")
        ]
        missing_non_market_names = [col for col in non_market_cols if col not in available_non_market_names]

        available_non_market_count = len(available_non_market_names)
        total_non_market_count = len(non_market_cols)
        fallback_non_market_count = max(total_non_market_count - available_non_market_count, 0)

        context_level = self._classify_feature_context_level(
            has_odds=bool(market_inputs.get("has_odds")),
            total_non_market_count=total_non_market_count,
            available_non_market_count=available_non_market_count,
        )

        available_feature_names = [
            col for col in columns
            if col in self.MARKET_FEATURES or col in available_non_market_names
        ]
        missing_feature_names = [col for col in columns if col not in available_feature_names]

        market_only_or_market_heavy_flag = bool(
            total_non_market_count <= 0 or available_non_market_count <= max(1, total_non_market_count // 3)
        )
        degraded_context_flag = bool(context_level == "degraded_context")
        quality_summary = self._build_runtime_quality_summary(
            features=features,
            context_level=context_level,
            available_non_market_count=available_non_market_count,
            total_non_market_count=total_non_market_count,
            market_only_or_market_heavy_flag=market_only_or_market_heavy_flag,
            has_odds=bool(market_inputs.get("has_odds")),
        )

        return {
            "features": features,
            "required_columns": list(columns),
            "context_level": context_level,
            "feature_context_level": context_level,
            "available_non_market_count": available_non_market_count,
            "total_non_market_count": total_non_market_count,
            "fallback_non_market_count": fallback_non_market_count,
            "missing_feature_count": len(missing_feature_names),
            "missing_non_market_count": len(missing_non_market_names),
            "degraded_context_flag": degraded_context_flag,
            "market_only_or_market_heavy_flag": market_only_or_market_heavy_flag,
            "available_feature_names": available_feature_names,
            "missing_feature_names": missing_feature_names,
            "missing_non_market_names": missing_non_market_names,
            "missing_critical_features": [],
            "status": "ok",
            "reason": None,
            "market_inputs": market_inputs,
            "source_meta": source_meta,
            **quality_summary,
        }

    def _resolve_runtime_feature_columns(self, required_columns: list[str] | None = None) -> list[str]:
        if required_columns:
            return list(required_columns)

        predictor_columns = getattr(self.predictor, "feature_columns", None)
        if isinstance(predictor_columns, list) and predictor_columns:
            return list(predictor_columns)

        if self.schema_path.exists():
            try:
                loaded = self.predictor.schema.load(str(self.schema_path))
                if loaded:
                    return list(loaded)
            except Exception:
                logger.debug("runtime feature columns: failed to load schema from %s", self.schema_path)

        template = FeatureBuilder.build_features({}, {}, {}, {})
        return [name for name in template.keys() if not str(name).startswith("__source_")]

    def _collect_payload_numeric_values(self, match: dict[str, Any], columns: list[str]) -> dict[str, float]:
        values: dict[str, float] = {}
        if not isinstance(match, dict):
            return values

        feature_payload = match.get("features")
        if not isinstance(feature_payload, dict):
            feature_payload = {}

        for column in columns:
            candidate = feature_payload.get(column)
            if candidate is None:
                candidate = match.get(column)
            if candidate is None:
                continue
            try:
                values[column] = float(candidate)
            except (TypeError, ValueError):
                continue
        return values

    @staticmethod
    def _classify_feature_context_level(
        *,
        has_odds: bool,
        total_non_market_count: int,
        available_non_market_count: int,
    ) -> str:
        if not has_odds:
            return "degraded_context"
        if total_non_market_count <= 0:
            return "odds_only_context"
        if available_non_market_count <= 0:
            return "odds_only_context"
        if available_non_market_count >= total_non_market_count:
            return "full_context"
        return "partial_context"

    @staticmethod
    def _safe_feature_value(features: dict[str, float], name: str, default: float = 0.0) -> float:
        try:
            raw = features.get(name)
            if raw is None:
                return float(default)
            value = float(raw)
            if np.isnan(value) or np.isinf(value):
                return float(default)
            return value
        except (TypeError, ValueError):
            return float(default)

    def _infer_market_probs_from_features(self, features: dict[str, float]) -> dict[str, float]:
        p1 = self._safe_feature_value(features, "implied_prob_1", default=-1.0)
        px = self._safe_feature_value(features, "implied_prob_x", default=-1.0)
        p2 = self._safe_feature_value(features, "implied_prob_2", default=-1.0)
        if min(p1, px, p2) >= 0.0:
            total = p1 + px + p2
            if total > 0.0:
                return {"P1": p1 / total, "PX": px / total, "P2": p2 / total}
        return {"P1": 1.0 / 3.0, "PX": 1.0 / 3.0, "P2": 1.0 / 3.0}

    def _derive_stats_context_probs(
        self,
        features: dict[str, float],
        market_probs: dict[str, float],
        non_market_ratio: float,
    ) -> dict[str, float]:
        stats_edge = (
            0.18 * self._safe_feature_value(features, "ppg_diff")
            + 0.14 * self._safe_feature_value(features, "split_advantage")
            + 0.10 * self._safe_feature_value(features, "goals_diff")
            + 0.16 * self._safe_feature_value(features, "xg_diff")
            + 0.04 * (self._safe_feature_value(features, "shots_diff") / 3.0)
            + 0.03 * (self._safe_feature_value(features, "possession_diff") / 8.0)
            + 0.22 * self._safe_feature_value(features, "home_advantage")
        )
        draw_signal = (
            self._safe_feature_value(features, "draw_pct")
            + 0.12 * self._safe_feature_value(features, "entropy")
            - 0.10 * self._safe_feature_value(features, "gap")
        )
        stats_probs = {
            "P1": max(0.05, 0.33 + stats_edge),
            "PX": max(0.05, draw_signal),
            "P2": max(0.05, 0.33 - stats_edge),
        }
        total = sum(stats_probs.values())
        if total <= 0.0:
            stats_probs = {"P1": 1.0 / 3.0, "PX": 1.0 / 3.0, "P2": 1.0 / 3.0}
        else:
            stats_probs = {key: value / total for key, value in stats_probs.items()}

        blend = min(0.75, max(0.15, float(non_market_ratio)))
        return {
            outcome: ((1.0 - blend) * float(market_probs[outcome])) + (blend * float(stats_probs[outcome]))
            for outcome in ("P1", "PX", "P2")
        }

    def _build_runtime_quality_summary(
        self,
        *,
        features: dict[str, float],
        context_level: str,
        available_non_market_count: int,
        total_non_market_count: int,
        market_only_or_market_heavy_flag: bool,
        has_odds: bool,
    ) -> dict[str, Any]:
        non_market_ratio = (
            (available_non_market_count / total_non_market_count) if total_non_market_count > 0 else 0.0
        )
        non_market_rich_flag = bool(
            total_non_market_count > 0 and available_non_market_count >= max(3, int(total_non_market_count * 0.6))
        )

        base_scores = {
            "full_context": 0.85,
            "partial_context": 0.62,
            "odds_only_context": 0.36,
            "degraded_context": 0.24,
        }
        quality_score = base_scores.get(context_level, 0.24)
        quality_score = (0.72 * quality_score) + (0.23 * non_market_ratio) + (0.05 if non_market_rich_flag else 0.0)
        if market_only_or_market_heavy_flag:
            quality_score -= 0.08
        if not has_odds:
            quality_score -= 0.10
        quality_score = max(0.0, min(1.0, quality_score))

        market_probs = self._infer_market_probs_from_features(features)
        stats_probs = self._derive_stats_context_probs(features, market_probs, non_market_ratio)
        market_alignment_score = max(
            0.0,
            min(
                1.0,
                1.0 - (0.5 * sum(abs(float(market_probs[key]) - float(stats_probs[key])) for key in ("P1", "PX", "P2"))),
            ),
        )
        market_favorite = max(market_probs, key=market_probs.get)
        stats_favorite = max(stats_probs, key=stats_probs.get)
        weak_favorite_flag = bool(
            market_favorite != "PX"
            and float(market_probs[market_favorite]) < 0.50
            and self._safe_feature_value(features, "gap") < 0.12
        )
        draw_risk_flag = bool(
            float(stats_probs["PX"]) >= 0.34
            and float(market_probs["PX"]) <= 0.30
            and (
                self._safe_feature_value(features, "draw_pct") >= 0.27
                or self._safe_feature_value(features, "entropy") >= 0.80
            )
        )
        stats_override_signal = bool(
            quality_score >= 0.60
            and stats_favorite != market_favorite
            and stats_favorite != "PX"
            and (float(stats_probs[stats_favorite]) - float(market_probs[stats_favorite])) >= 0.08
        )
        market_disagreement_flag = bool(
            quality_score >= 0.45
            and market_favorite != stats_favorite
            and market_alignment_score < 0.72
        )
        suspicious_market_disagreement_flag = bool(
            market_disagreement_flag and quality_score < 0.60 and not stats_override_signal
        )

        if quality_score >= 0.72:
            signal_strength = "strong_signal"
        elif quality_score >= 0.45:
            signal_strength = "medium_signal"
        else:
            signal_strength = "weak_signal"

        return {
            "non_market_ratio": round(non_market_ratio, 4),
            "non_market_rich_flag": non_market_rich_flag,
            "quality_score": round(quality_score, 4),
            "signal_strength": signal_strength,
            "market_alignment_score": round(market_alignment_score, 4),
            "market_disagreement_flag": market_disagreement_flag,
            "suspicious_market_disagreement_flag": suspicious_market_disagreement_flag,
            "weak_favorite_flag": weak_favorite_flag,
            "draw_risk_flag": draw_risk_flag,
            "hidden_draw_risk": draw_risk_flag,
            "stats_override_signal": stats_override_signal,
            "market_favorite_outcome": market_favorite,
            "stats_context_favorite_outcome": stats_favorite,
        }

    def _build_runtime_features_for_diagnostics(
        self,
        match: dict[str, Any],
        required_columns: list[str],
        market_inputs: dict[str, Any],
        implied_probs: dict[str, float] | None,
    ) -> dict[str, float]:
        """Build the widest safe runtime feature payload for diagnostics inference.

        Unlike strict odds-based feature assembly, this helper accepts rows without
        odds and only forwards numeric values that are actually present in payload.
        """
        if not isinstance(match, dict):
            return {}

        features_payload = match.get("features")
        if not isinstance(features_payload, dict):
            features_payload = {}

        built: dict[str, float] = {}
        for column in required_columns:
            raw: Any = None

            if column in features_payload:
                raw = features_payload.get(column)
            elif column in match:
                raw = match.get(column)
            elif column == "odds_ft_1":
                raw = market_inputs.get("odds_ft_1")
            elif column == "odds_ft_x":
                raw = market_inputs.get("odds_ft_x")
            elif column == "odds_ft_2":
                raw = market_inputs.get("odds_ft_2")
            elif column == "implied_prob_1" and implied_probs is not None:
                raw = implied_probs.get("P1")
            elif column == "implied_prob_x" and implied_probs is not None:
                raw = implied_probs.get("PX")
            elif column == "implied_prob_2" and implied_probs is not None:
                raw = implied_probs.get("P2")

            if raw is None:
                continue

            try:
                built[column] = float(raw)
            except (TypeError, ValueError):
                continue

        # ── Additional derivation: PPG features from raw match context keys ─────────
        # The loop above looks for exact column names (e.g. 'home_home_ppg'), but the
        # normalized match payload uses 'home_ppg' / 'pre_match_home_ppg' etc.
        # Derive here so no-odds path gets the same PPG signal as the main path.
        raw_dict = match.get("raw")
        raw_dict = raw_dict if isinstance(raw_dict, dict) else {}

        def _diag_pick(*keys: str) -> float | None:
            for k in keys:
                for src in (features_payload, match, raw_dict):
                    v = src.get(k)
                    if v is None:
                        continue
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        continue
            return None

        home_ppg_v = _diag_pick("home_home_ppg", "home_ppg", "pre_match_home_ppg")
        away_ppg_v = _diag_pick("away_away_ppg", "away_ppg", "pre_match_away_ppg")

        if home_ppg_v is not None and "home_home_ppg" in required_columns and "home_home_ppg" not in built:
            built["home_home_ppg"] = home_ppg_v
        if away_ppg_v is not None and "away_away_ppg" in required_columns and "away_away_ppg" not in built:
            built["away_away_ppg"] = away_ppg_v

        if home_ppg_v is not None and away_ppg_v is not None:
            if "ppg_diff" in required_columns and "ppg_diff" not in built:
                built["ppg_diff"] = home_ppg_v - away_ppg_v
            if "split_advantage" in required_columns and "split_advantage" not in built:
                sh = _diag_pick("pre_match_home_ppg", "home_home_ppg", "home_ppg")
                sa = _diag_pick("pre_match_away_ppg", "away_away_ppg", "away_ppg")
                built["split_advantage"] = float((sh if sh is not None else home_ppg_v) - (sa if sa is not None else away_ppg_v))

        if "avg_goals" in required_columns and "avg_goals" not in built:
            avg_g = _diag_pick("avg_goals", "league_avg_goals", "season_avg_goals", "avg_potential", "seasonAVG")
            if avg_g is not None:
                built["avg_goals"] = avg_g

        # ── DB enrichment for no-odds path ────────────────────────────────────────────
        # When a DB is wired and IDs are available, enrich with stats/season context that
        # can't be derived from the match payload alone (xg, shots, possession, draw_pct, etc.).
        if self.db is not None:
            diag_season_id = match.get("season_id")
            diag_h_tid = self._to_float_or_none(match.get("home_team_id"))
            diag_a_tid = self._to_float_or_none(match.get("away_team_id"))

            # Team-level stats: xg / shots / possession from team_season_stats
            if diag_h_tid is not None and diag_a_tid is not None and diag_season_id is not None:
                try:
                    dh = self._fetch_team_stats_from_db(int(diag_h_tid), diag_season_id)
                    da = self._fetch_team_stats_from_db(int(diag_a_tid), diag_season_id)

                    if "xg_diff" in required_columns and "xg_diff" not in built:
                        h_xg = self._to_float_or_none(dh.get("xg_for_avg_overall"))
                        a_xg = self._to_float_or_none(da.get("xg_for_avg_overall"))
                        if h_xg is not None and a_xg is not None:
                            built["xg_diff"] = h_xg - a_xg

                    if "shots_diff" in required_columns and "shots_diff" not in built:
                        h_sh_v = dh.get("shots_avg_home")
                        h_sh = self._to_float_or_none(h_sh_v if h_sh_v is not None else dh.get("shots_avg_overall"))
                        a_sh_v = da.get("shots_avg_away")
                        a_sh = self._to_float_or_none(a_sh_v if a_sh_v is not None else da.get("shots_avg_overall"))
                        if h_sh is not None and a_sh is not None:
                            built["shots_diff"] = h_sh - a_sh

                    if "possession_diff" in required_columns and "possession_diff" not in built:
                        h_po_v = dh.get("possession_avg_home")
                        h_po = self._to_float_or_none(h_po_v if h_po_v is not None else dh.get("possession_avg_overall"))
                        a_po_v = da.get("possession_avg_away")
                        a_po = self._to_float_or_none(a_po_v if a_po_v is not None else da.get("possession_avg_overall"))
                        if h_po is not None and a_po is not None:
                            built["possession_diff"] = h_po - a_po

                    if "goals_diff" in required_columns and "goals_diff" not in built:
                        h_g_v = dh.get("goals_for_avg_home")
                        h_g = self._to_float_or_none(h_g_v if h_g_v is not None else dh.get("goals_for_avg_overall"))
                        a_g_v = da.get("goals_for_avg_away")
                        a_g = self._to_float_or_none(a_g_v if a_g_v is not None else da.get("goals_for_avg_overall"))
                        if h_g is not None and a_g is not None:
                            built["goals_diff"] = h_g - a_g
                except Exception:
                    logger.debug("_build_runtime_features_for_diagnostics: team stats DB lookup failed")

            # League-level context: draw_pct / home_advantage / avg_goals from league_season_stats
            if diag_season_id is not None:
                try:
                    league_ctx = self._fetch_league_season_stats_from_db(diag_season_id)
                    if "draw_pct" in required_columns and "draw_pct" not in built and "draw_pct" in league_ctx:
                        built["draw_pct"] = league_ctx["draw_pct"]
                    if "home_advantage" in required_columns and "home_advantage" not in built and "home_advantage" in league_ctx:
                        built["home_advantage"] = league_ctx["home_advantage"]
                    if "avg_goals" in required_columns and "avg_goals" not in built and "avg_goals" in league_ctx:
                        built["avg_goals"] = league_ctx["avg_goals"]
                except Exception:
                    logger.debug("_build_runtime_features_for_diagnostics: league stats DB lookup failed")

        return built

    def _validate_provided_probs(self, match: dict[str, Any]) -> dict[str, Any]:
        """Validate whether provided precomputed probabilities are trustworthy.

        Returns:
            {
                "accepted_probs": dict|None,
                "rejected_reason": str|None,
                "candidate_probs": dict|None,
            }
        """
        empty_result = {
            "accepted_probs": None,
            "rejected_reason": None,
            "candidate_probs": None,
        }
        if not isinstance(match, dict):
            return {**empty_result, "rejected_reason": "empty_probs"}

        candidate_probs = self._extract_raw_provided_probs(match)
        if candidate_probs is None:
            return {**empty_result, "rejected_reason": "empty_probs"}

        p1 = self._safe_float(candidate_probs.get("P1"))
        px = self._safe_float(candidate_probs.get("PX"))
        p2 = self._safe_float(candidate_probs.get("P2"))
        total = p1 + px + p2

        if any(v < 0.0 or v > 1.0 for v in (p1, px, p2)):
            return {**empty_result, "rejected_reason": "out_of_range_probability", "candidate_probs": candidate_probs}

        if total <= 0.0:
            return {**empty_result, "rejected_reason": "empty_probs", "candidate_probs": candidate_probs}

        # Validate raw sum before normalization: large drift means bad payload quality.
        if abs(total - 1.0) > 0.03:
            return {**empty_result, "rejected_reason": "invalid_probability_sum", "candidate_probs": candidate_probs}

        normalized = self._normalize_probs({"P1": p1, "PX": px, "P2": p2})
        if self._is_fallback_template_probs(normalized):
            return {**empty_result, "rejected_reason": "fallback_template_detected", "candidate_probs": normalized}

        if self._is_fake_pool_derived(match, normalized):
            return {**empty_result, "rejected_reason": "fake_pool_derived", "candidate_probs": normalized}

        odds_status = str(match.get("odds_status", "")).strip().lower()
        if odds_status in {"odds_missing", "odds_partial"}:
            return {**empty_result, "rejected_reason": "invalid_odds_status", "candidate_probs": normalized}

        source_label = str(match.get("probabilities_source", "")).strip().lower()
        source_col = str(match.get("source", "")).strip().lower()
        source_reason = str(match.get("source_reason", "")).strip().lower()

        if source_label == "feature_error" or source_col == "feature_error" or "feature_error" in source_reason:
            return {**empty_result, "rejected_reason": "feature_error", "candidate_probs": normalized}

        if source_label == "predict_error" or source_col == "predict_error" or "predict_error" in source_reason:
            return {**empty_result, "rejected_reason": "predict_error", "candidate_probs": normalized}

        if "pool" in source_label or "public" in source_label:
            return {**empty_result, "rejected_reason": "fake_pool_derived", "candidate_probs": normalized}

        trusted_sources = {"model_runtime", "trusted_precomputed"}
        if source_label and source_label not in trusted_sources:
            return {**empty_result, "rejected_reason": "missing_trusted_source", "candidate_probs": normalized}
        if not source_label and source_col not in {"matches_tab", "model_runtime"}:
            return {**empty_result, "rejected_reason": "missing_trusted_source", "candidate_probs": normalized}

        return {
            "accepted_probs": normalized,
            "rejected_reason": None,
            "candidate_probs": normalized,
        }

    def _extract_raw_provided_probs(self, match: dict[str, Any]) -> dict[str, float] | None:
        if not isinstance(match, dict):
            return None

        model_probs = match.get("model_probs")
        if isinstance(model_probs, dict):
            p1 = self._safe_float(model_probs.get("P1"))
            px = self._safe_float(model_probs.get("PX"))
            p2 = self._safe_float(model_probs.get("P2"))
            if p1 + px + p2 > 0:
                return {"P1": p1, "PX": px, "P2": p2}

        p1 = self._safe_float(match.get("model_prob_1"))
        px = self._safe_float(match.get("model_prob_x"))
        p2 = self._safe_float(match.get("model_prob_2"))
        if p1 + px + p2 > 0:
            return {"P1": p1, "PX": px, "P2": p2}

        return None

    def _is_fallback_template_probs(self, probs: dict[str, float]) -> bool:
        templates = (
            {"P1": 1.0 / 3.0, "PX": 1.0 / 3.0, "P2": 1.0 / 3.0},
            {"P1": 0.4408, "PX": 0.2424, "P2": 0.3168},
        )
        for template in templates:
            if (
                abs(self._safe_float(probs.get("P1")) - template["P1"]) <= 1e-4
                and abs(self._safe_float(probs.get("PX")) - template["PX"]) <= 1e-4
                and abs(self._safe_float(probs.get("P2")) - template["P2"]) <= 1e-4
            ):
                return True
        return False

    def _is_fake_pool_derived(self, match: dict[str, Any], probs: dict[str, float]) -> bool:
        if not isinstance(match, dict):
            return False
        pool_raw = match.get("pool_probs")
        if not isinstance(pool_raw, dict) or not pool_raw:
            return False

        pool_p1 = self._safe_float(pool_raw.get("P1"))
        pool_px = self._safe_float(pool_raw.get("PX"))
        pool_p2 = self._safe_float(pool_raw.get("P2"))
        if pool_p1 + pool_px + pool_p2 <= 0:
            return False

        pool_norm = self._normalize_probs({"P1": pool_p1, "PX": pool_px, "P2": pool_p2})
        distance = (
            abs(pool_norm["P1"] - self._safe_float(probs.get("P1")))
            + abs(pool_norm["PX"] - self._safe_float(probs.get("PX")))
            + abs(pool_norm["P2"] - self._safe_float(probs.get("P2")))
        )
        return distance <= 1e-6

    def _derive_fallback_source_label(self, match: dict[str, Any], rejected_reason: str | None) -> str:
        if not isinstance(match, dict):
            return "hard_fallback"
        odds_status = str(match.get("odds_status", "")).strip().lower()
        if odds_status == "odds_missing":
            return "no_odds"
        if odds_status == "odds_partial":
            return "odds_partial"

        source_col = str(match.get("source", "")).strip().lower()
        source_reason = str(match.get("source_reason", "")).strip().lower()
        if source_col == "feature_error" or "feature_error" in source_reason:
            return "feature_error"
        if source_col == "predict_error" or "predict_error" in source_reason:
            return "predict_error"
        if rejected_reason:
            return "rejected_precomputed"
        return "hard_fallback"

    def _build_match_features(
        self,
        match: dict[str, Any],
        required_columns: list[str],
        pool_probs: dict[str, float],
    ) -> dict[str, float] | tuple[None, str]:
        features, reason, _ = self._build_match_features_with_meta(match, required_columns, pool_probs)
        if features is None:
            return None, str(reason or "runtime feature build failed")
        return features

    def _build_match_features_with_meta(
        self,
        match: dict[str, Any],
        required_columns: list[str],
        pool_probs: dict[str, float],
    ) -> tuple[dict[str, float] | None, str | None, dict[str, str]]:
        """
        Собрать feature payload для матча на основе required_columns.

        CRITICAL DIFFERENCES from DB build:
        1. MUST compute implied_prob_1/x/2 from odds (NOT from pool_probs)
        2. Stats features (goals_diff, xg_diff, etc.) are NOT available from TOTO payload
           → Use book odds probabilities as proxy estimates where possible
        3. Handles aliases: odds_ft_1/O1, odds_ft_x/OX, odds_ft_2/O2

        Returns:
            dict: feature vector ready for model.predict()
            OR tuple (None, reason): if critical features cannot be assembled
        """
        if not isinstance(match, dict):
            return None, "invalid match payload (not a dict)", {}

        # Extract odds from match or nested odds dict
        odds_payload = match.get("odds")
        odds = odds_payload if isinstance(odds_payload, dict) else {}

        # Priority: direct match fields > nested odds dict > aliases
        odds_ft_1 = match.get("odds_ft_1") or odds.get("odds_ft_1") or odds.get("O1")
        odds_ft_x = match.get("odds_ft_x") or odds.get("odds_ft_x") or odds.get("OX")
        odds_ft_2 = match.get("odds_ft_2") or odds.get("odds_ft_2") or odds.get("O2")

        # CRITICAL: implied_prob_* MUST be computed from odds, not from pool_probs
        # This represents "fair" market pricing; pool is for context only
        if all([odds_ft_1, odds_ft_x, odds_ft_2]):
            try:
                odds_float_1 = self._safe_float(odds_ft_1)
                odds_float_x = self._safe_float(odds_ft_x)
                odds_float_2 = self._safe_float(odds_ft_2)
                implied = self._normalize_odds(odds_float_1, odds_float_x, odds_float_2)
            except (TypeError, ValueError):
                return None, f"odds conversion failed (O1={odds_ft_1}, OX={odds_ft_x}, O2={odds_ft_2})", {}
        else:
            return None, f"missing odds (O1={odds_ft_1}, OX={odds_ft_x}, O2={odds_ft_2})", {}

        feature_payload = match.get("features")
        if not isinstance(feature_payload, dict):
            feature_payload = {}
        raw_data = match.get("raw")
        raw_payload: dict[str, Any] = raw_data if isinstance(raw_data, dict) else {}

        source_meta: dict[str, str] = {}

        # Assemble baseline features that TOTO payload CAN provide
        raw_values: dict[str, Any] = {
            # Odds tier
            "odds_ft_1": odds_ft_1,
            "odds_ft_x": odds_ft_x,
            "odds_ft_2": odds_ft_2,
            # IMPLIED PROBS from odds (not pool)
            "implied_prob_1": implied["implied_prob_1"],
            "implied_prob_x": implied["implied_prob_x"],
            "implied_prob_2": implied["implied_prob_2"],
            # Uncertainty metrics (computed from odds/implied)
            "entropy": self._calc_entropy([implied["implied_prob_1"], implied["implied_prob_x"], implied["implied_prob_2"]]),
            "gap": implied.get("gap", 0.0),
            "volatility": implied.get("volatility", 0.0),
            # PPG placeholder: when not in payload, use implied probabilities as neutral default
            "ppg_diff": 0.0,
            "home_home_ppg": 0.0,
            "away_away_ppg": 0.0,
            "split_advantage": 0.0,
            # Stats features: NOT available from TOTO payload (requires team/league history)
            # Use 0.0 with explicit source markers
            "goals_diff": 0.0,
            "xg_diff": 0.0,
            "shots_diff": 0.0,
            "possession_diff": 0.0,
            # League features: NOT available from TOTO payload
            "draw_pct": 0.0,
            "home_advantage": 0.0,
            "avg_goals": 0.0,
        }
        source_meta.update({
            "odds_ft_1": "market",
            "odds_ft_x": "market",
            "odds_ft_2": "market",
            "implied_prob_1": "market",
            "implied_prob_x": "market",
            "implied_prob_2": "market",
            "entropy": "market_derived",
            "gap": "market_derived",
            "volatility": "market_derived",
            "ppg_diff": "fallback_default",
            "home_home_ppg": "fallback_default",
            "away_away_ppg": "fallback_default",
            "split_advantage": "fallback_default",
            "goals_diff": "fallback_default",
            "xg_diff": "fallback_default",
            "shots_diff": "fallback_default",
            "possession_diff": "fallback_default",
            "draw_pct": "fallback_default",
            "home_advantage": "fallback_default",
            "avg_goals": "fallback_default",
        })

        # Prefer explicit runtime payload values for non-market features when present.
        for column in required_columns:
            if column in self.MARKET_FEATURES:
                continue
            candidate = feature_payload.get(column)
            if candidate is None:
                candidate = match.get(column)
            if candidate is None:
                continue
            raw_values[column] = candidate
            source_meta[column] = "payload"

        def _pick_numeric(*keys: str) -> float | None:
            for key in keys:
                if key in feature_payload:
                    value = self._to_float_or_none(feature_payload.get(key))
                    if value is not None:
                        return value
                if key in match:
                    value = self._to_float_or_none(match.get(key))
                    if value is not None:
                        return value
                if key in raw_payload:
                    value = self._to_float_or_none(raw_payload.get(key))
                    if value is not None:
                        return value
            return None

        # Reconstruct non-market runtime features from payload context when available.
        home_ppg = _pick_numeric("home_home_ppg", "home_ppg", "pre_match_home_ppg")
        away_ppg = _pick_numeric("away_away_ppg", "away_ppg", "pre_match_away_ppg")
        split_home = _pick_numeric("pre_match_home_ppg", "home_home_ppg", "home_ppg")
        split_away = _pick_numeric("pre_match_away_ppg", "away_away_ppg", "away_ppg")

        if home_ppg is not None and away_ppg is not None and source_meta.get("ppg_diff") != "payload":
            raw_values["ppg_diff"] = float(home_ppg - away_ppg)
            source_meta["ppg_diff"] = "derived"
        if home_ppg is not None and source_meta.get("home_home_ppg") != "payload":
            raw_values["home_home_ppg"] = float(home_ppg)
            source_meta["home_home_ppg"] = "derived"
        if away_ppg is not None and source_meta.get("away_away_ppg") != "payload":
            raw_values["away_away_ppg"] = float(away_ppg)
            source_meta["away_away_ppg"] = "derived"
        if split_home is not None and split_away is not None and source_meta.get("split_advantage") != "payload":
            raw_values["split_advantage"] = float(split_home - split_away)
            source_meta["split_advantage"] = "derived"

        home_goals_avg = _pick_numeric("home_goals_for_avg_home", "home_goals_for_avg", "goals_for_avg_home")
        away_goals_avg = _pick_numeric("away_goals_for_avg_away", "away_goals_for_avg", "goals_for_avg_away")
        if home_goals_avg is not None and away_goals_avg is not None and source_meta.get("goals_diff") != "payload":
            raw_values["goals_diff"] = float(home_goals_avg - away_goals_avg)
            source_meta["goals_diff"] = "derived"

        # xg_diff: training uses xg_for_avg_overall; also accept common FootyStats variant keys
        home_xg_avg = _pick_numeric(
            "home_xg_for_avg", "xg_for_avg_home", "home_xg_avg",
            "pre_match_home_xg", "home_pre_xg",
        )
        away_xg_avg = _pick_numeric(
            "away_xg_for_avg", "xg_for_avg_away", "away_xg_avg",
            "pre_match_away_xg", "away_pre_xg",
        )
        if home_xg_avg is not None and away_xg_avg is not None and source_meta.get("xg_diff") != "payload":
            raw_values["xg_diff"] = float(home_xg_avg - away_xg_avg)
            source_meta["xg_diff"] = "derived"

        # shots_diff: overall average mirrors training column (shots_avg_overall)
        home_shots_avg = _pick_numeric(
            "home_shots_avg_home", "home_shots_avg", "shots_avg_home",
            "home_shots_avg_overall", "home_shotsAVG",
        )
        away_shots_avg = _pick_numeric(
            "away_shots_avg_away", "away_shots_avg", "shots_avg_away",
            "away_shots_avg_overall", "away_shotsAVG",
        )
        if home_shots_avg is not None and away_shots_avg is not None and source_meta.get("shots_diff") != "payload":
            raw_values["shots_diff"] = float(home_shots_avg - away_shots_avg)
            source_meta["shots_diff"] = "derived"

        # possession_diff: mirrors training column (possession_avg_overall)
        home_poss_avg = _pick_numeric(
            "home_possession_avg_home", "home_possession_avg", "possession_avg_home",
            "home_possession_avg_overall", "home_possessionAVG",
        )
        away_poss_avg = _pick_numeric(
            "away_possession_avg_away", "away_possession_avg", "possession_avg_away",
            "away_possession_avg_overall", "away_possessionAVG",
        )
        if home_poss_avg is not None and away_poss_avg is not None and source_meta.get("possession_diff") != "payload":
            raw_values["possession_diff"] = float(home_poss_avg - away_poss_avg)
            source_meta["possession_diff"] = "derived"

        # draw_pct: training uses league_season_stats.draw_pct aliased as league_draw_pct.
        # Accept FootyStats raw keys (draw_percentage) that may appear in raw_payload.
        draw_pct = _pick_numeric(
            "draw_pct", "league_draw_pct", "draw_percentage", "league_draw_percentage",
        )
        if draw_pct is not None and source_meta.get("draw_pct") != "payload":
            raw_values["draw_pct"] = float(self._normalize_percent_like(draw_pct))
            source_meta["draw_pct"] = "derived"

        # home_advantage: training uses league_season_stats.home_advantage aliased as league_home_advantage.
        home_advantage = _pick_numeric(
            "home_advantage", "league_home_advantage", "home_advantage_pct",
        )
        if home_advantage is None:
            home_win_pct = _pick_numeric(
                "league_home_win_pct", "home_win_pct", "home_win_percentage",
            )
            away_win_pct = _pick_numeric(
                "league_away_win_pct", "away_win_pct", "away_win_percentage",
            )
            if home_win_pct is not None and away_win_pct is not None:
                home_advantage = (
                    self._normalize_percent_like(home_win_pct)
                    - self._normalize_percent_like(away_win_pct)
                )
        if home_advantage is not None and source_meta.get("home_advantage") != "payload":
            raw_values["home_advantage"] = float(home_advantage)
            source_meta["home_advantage"] = "derived"

        avg_goals = _pick_numeric(
            "avg_goals", "league_avg_goals", "season_avg_goals",
            "avg_potential", "seasonAVG",
        )
        if avg_goals is not None and source_meta.get("avg_goals") != "payload":
            raw_values["avg_goals"] = float(avg_goals)
            source_meta["avg_goals"] = "derived"

        # --- DB stats lookup (when DB wired and team/season IDs are known) ---
        # Recovers shots_diff, possession_diff, xg_diff from team_season_stats and
        # draw_pct / home_advantage from seasonal match history — matching training path.
        if self.db is not None:
            raw_home_team_id = match.get("home_team_id")
            raw_away_team_id = match.get("away_team_id")
            raw_season_id = match.get("season_id")
            h_tid = self._to_float_or_none(raw_home_team_id)
            a_tid = self._to_float_or_none(raw_away_team_id)
            if h_tid is not None and a_tid is not None and raw_season_id is not None:
                try:
                    h_stats = self._fetch_team_stats_from_db(int(h_tid), raw_season_id)
                    a_stats = self._fetch_team_stats_from_db(int(a_tid), raw_season_id)

                    # xg_diff — consistent with training (xg_for_avg_overall is the source)
                    if source_meta.get("xg_diff") not in ("payload", "derived"):
                        h_xg = self._to_float_or_none(h_stats.get("xg_for_avg_overall"))
                        a_xg = self._to_float_or_none(a_stats.get("xg_for_avg_overall"))
                        if h_xg is not None and a_xg is not None:
                            raw_values["xg_diff"] = h_xg - a_xg
                            source_meta["xg_diff"] = "db_lookup"

                    # goals_diff — prefer goals_for_avg_home/away, fallback to xg proxy (mirrors training)
                    if source_meta.get("goals_diff") not in ("payload", "derived"):
                        h_g = self._to_float_or_none(
                            h_stats.get("goals_for_avg_home") or h_stats.get("goals_for_avg_overall")
                        )
                        a_g = self._to_float_or_none(
                            a_stats.get("goals_for_avg_away") or a_stats.get("goals_for_avg_overall")
                        )
                        if h_g is not None and a_g is not None:
                            raw_values["goals_diff"] = h_g - a_g
                            source_meta["goals_diff"] = "db_lookup"
                        elif source_meta.get("xg_diff") == "db_lookup":
                            # xg proxy — same fallback logic as training build
                            raw_values["goals_diff"] = raw_values.get("xg_diff", 0.0)
                            source_meta["goals_diff"] = "db_lookup_xg_proxy"

                    # shots_diff from context-specific averages
                    if source_meta.get("shots_diff") not in ("payload", "derived"):
                        h_sh = self._to_float_or_none(
                            h_stats.get("shots_avg_home") or h_stats.get("shots_avg_overall")
                        )
                        a_sh = self._to_float_or_none(
                            a_stats.get("shots_avg_away") or a_stats.get("shots_avg_overall")
                        )
                        if h_sh is not None and a_sh is not None:
                            raw_values["shots_diff"] = h_sh - a_sh
                            source_meta["shots_diff"] = "db_lookup"

                    # possession_diff from context-specific averages
                    if source_meta.get("possession_diff") not in ("payload", "derived"):
                        h_po = self._to_float_or_none(
                            h_stats.get("possession_avg_home") or h_stats.get("possession_avg_overall")
                        )
                        a_po = self._to_float_or_none(
                            a_stats.get("possession_avg_away") or a_stats.get("possession_avg_overall")
                        )
                        if h_po is not None and a_po is not None:
                            raw_values["possession_diff"] = h_po - a_po
                            source_meta["possession_diff"] = "db_lookup"

                    # ppg_diff / home_home_ppg / away_away_ppg from context-specific season PPG
                    # These are absent from the TOTO payload but stored in team_season_stats.
                    if source_meta.get("home_home_ppg") not in ("payload", "derived"):
                        h_ppg_h = self._to_float_or_none(h_stats.get("season_ppg_home"))
                        if h_ppg_h is not None:
                            raw_values["home_home_ppg"] = h_ppg_h
                            source_meta["home_home_ppg"] = "db_lookup"

                    if source_meta.get("away_away_ppg") not in ("payload", "derived"):
                        a_ppg_a = self._to_float_or_none(a_stats.get("season_ppg_away"))
                        if a_ppg_a is not None:
                            raw_values["away_away_ppg"] = a_ppg_a
                            source_meta["away_away_ppg"] = "db_lookup"

                    if source_meta.get("ppg_diff") not in ("payload", "derived"):
                        h_ppg_o = self._to_float_or_none(
                            h_stats.get("season_ppg_overall") or h_stats.get("season_ppg_home")
                        )
                        a_ppg_o = self._to_float_or_none(
                            a_stats.get("season_ppg_overall") or a_stats.get("season_ppg_away")
                        )
                        if h_ppg_o is not None and a_ppg_o is not None:
                            raw_values["ppg_diff"] = h_ppg_o - a_ppg_o
                            source_meta["ppg_diff"] = "db_lookup"

                    # split_advantage: recompute from DB-sourced context-specific PPG
                    if source_meta.get("split_advantage") not in ("payload", "derived"):
                        h_h_v = raw_values.get("home_home_ppg")
                        a_a_v = raw_values.get("away_away_ppg")
                        if (
                            h_h_v is not None
                            and a_a_v is not None
                            and source_meta.get("home_home_ppg") == "db_lookup"
                            and source_meta.get("away_away_ppg") == "db_lookup"
                        ):
                            raw_values["split_advantage"] = float(h_h_v) - float(a_a_v)
                            source_meta["split_advantage"] = "db_lookup"
                except Exception:
                    logger.debug("_build_match_features_with_meta: DB team stats lookup failed")

                # Season context: league_season_stats first (same source as training JOIN),
                # then fall back to computing from completed match history.
                try:
                    league_ctx = self._fetch_league_season_stats_from_db(raw_season_id)
                    if source_meta.get("draw_pct") not in ("payload", "derived") and "draw_pct" in league_ctx:
                        raw_values["draw_pct"] = league_ctx["draw_pct"]
                        source_meta["draw_pct"] = "db_lookup"
                    if source_meta.get("home_advantage") not in ("payload", "derived") and "home_advantage" in league_ctx:
                        raw_values["home_advantage"] = league_ctx["home_advantage"]
                        source_meta["home_advantage"] = "db_lookup"
                    if source_meta.get("avg_goals") not in ("payload", "derived") and "avg_goals" in league_ctx:
                        raw_values["avg_goals"] = league_ctx["avg_goals"]
                        source_meta["avg_goals"] = "db_lookup"
                except Exception:
                    logger.debug("_build_match_features_with_meta: league_season_stats lookup failed")

                # Fallback: compute season context from completed match history
                # (used when league_season_stats row is missing or fields are NULL)
                _still_need_season = (
                    source_meta.get("draw_pct") not in ("payload", "derived", "db_lookup")
                    or source_meta.get("home_advantage") not in ("payload", "derived", "db_lookup")
                    or source_meta.get("avg_goals") not in ("payload", "derived", "db_lookup")
                )
                if _still_need_season:
                    try:
                        season_ctx = self._compute_season_context_from_db(raw_season_id)
                        if source_meta.get("draw_pct") not in ("payload", "derived", "db_lookup") and "draw_pct" in season_ctx:
                            raw_values["draw_pct"] = season_ctx["draw_pct"]
                            source_meta["draw_pct"] = "db_lookup_computed"
                        if source_meta.get("home_advantage") not in ("payload", "derived", "db_lookup") and "home_advantage" in season_ctx:
                            raw_values["home_advantage"] = season_ctx["home_advantage"]
                            source_meta["home_advantage"] = "db_lookup_computed"
                        if source_meta.get("avg_goals") not in ("payload", "derived", "db_lookup") and "avg_goals" in season_ctx:
                            raw_values["avg_goals"] = season_ctx["avg_goals"]
                            source_meta["avg_goals"] = "db_lookup_computed"
                    except Exception:
                        logger.debug("_build_match_features_with_meta: DB season context lookup failed")

        # Collect missing features that model requires
        features: dict[str, float] = {}
        missing: list[str] = []

        for column in required_columns:
            if column in raw_values:
                try:
                    features[column] = float(raw_values[column])
                except (TypeError, ValueError):
                    return None, f"feature '{column}' conversion failed (value={raw_values[column]})", source_meta
            else:
                missing.append(column)

        # If there are missing features critical to model, return error
        critical_features = {"odds_ft_1", "odds_ft_x", "odds_ft_2", "implied_prob_1", "implied_prob_x", "implied_prob_2"}
        critical_missing = [f for f in missing if f in critical_features]
        if critical_missing:
            return None, f"missing critical features: {critical_missing}", source_meta

        # Non-critical missing features (stats) are OK; they default to 0.0
        # This allows TOTO to operate in degraded mode (odds+pool only, no deeper stats)
        return features, None, source_meta

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_percent_like(value: float) -> float:
        if value > 1.0:
            return value / 100.0
        return value

    def _fetch_team_stats_from_db(self, team_id: int, season_id: Any) -> dict[str, Any]:
        """Fetch pre-match team season stats row from local DB. Returns empty dict on miss."""
        try:
            conn = self.db.conn  # type: ignore[union-attr]
            row = conn.execute(
                "SELECT * FROM team_season_stats WHERE team_id = ? AND season_id = ?",
                (team_id, int(season_id)),
            ).fetchone()
            if row:
                return dict(row)
        except Exception:
            logger.debug("_fetch_team_stats_from_db: lookup failed team_id=%s season_id=%s", team_id, season_id)
        return {}

    def _fetch_league_season_stats_from_db(self, season_id: Any) -> dict[str, float]:
        """Fetch league season stats from local DB for the given season.

        Primary source for draw_pct, home_advantage, avg_goals — matches the LEFT JOIN
        to league_season_stats used in build_training_dataset_from_db.
        Returns empty dict on miss or when None values are found.
        """
        try:
            conn = self.db.conn  # type: ignore[union-attr]
            row = conn.execute(
                "SELECT draw_pct, home_win_pct, away_win_pct, home_advantage, season_avg_goals"
                " FROM league_season_stats WHERE season_id = ?",
                (int(season_id),),
            ).fetchone()
            if not row:
                return {}
            row_dict = dict(row)
            result: dict[str, float] = {}

            d_pct = self._to_float_or_none(row_dict.get("draw_pct"))
            if d_pct is not None:
                result["draw_pct"] = self._normalize_percent_like(d_pct)

            h_adv = self._to_float_or_none(row_dict.get("home_advantage"))
            if h_adv is None:
                # Compute from win percentages when direct column is absent
                hw = self._to_float_or_none(row_dict.get("home_win_pct"))
                aw = self._to_float_or_none(row_dict.get("away_win_pct"))
                if hw is not None and aw is not None:
                    h_adv = self._normalize_percent_like(hw) - self._normalize_percent_like(aw)
            if h_adv is not None:
                result["home_advantage"] = h_adv

            avg_g = self._to_float_or_none(row_dict.get("season_avg_goals"))
            if avg_g is not None:
                result["avg_goals"] = avg_g

            return result
        except Exception:
            logger.debug("_fetch_league_season_stats_from_db: lookup failed for season_id=%s", season_id)
        return {}

    def _compute_season_context_from_db(self, season_id: Any) -> dict[str, float]:
        """Compute draw_pct, home_advantage, and avg_goals from completed match history for `season_id`.

        Mirrors the fallback logic in build_training_dataset_from_db, so the live path derives 
        the same values as training when league_season_stats fields are empty.
        """
        try:
            conn = self.db.conn  # type: ignore[union-attr]
            rows = conn.execute(
                "SELECT home_team_id, away_team_id, winning_team_id, home_goals, away_goals"
                " FROM matches WHERE season_id = ? AND status = 'completed'",
                (int(season_id),),
            ).fetchall()
            total = len(rows)
            if total < 3:
                return {}
            draws = sum(
                1
                for r in rows
                if (r[2] is None or r[2] == 0)
                and r[3] is not None
                and r[4] is not None
                and int(r[3]) == int(r[4])
            )
            home_wins = sum(1 for r in rows if r[2] and r[2] == r[0])
            away_wins = sum(1 for r in rows if r[2] and r[2] == r[1])
            
            # Compute avg_goals from all completed matches in season
            total_goals = sum(
                int(r[3]) + int(r[4])
                for r in rows
                if r[3] is not None and r[4] is not None
            )
            total_matches_with_goals = sum(
                1 for r in rows
                if r[3] is not None and r[4] is not None
            )
            avg_goals_season = total_goals / total_matches_with_goals if total_matches_with_goals > 0 else 1.5
            
            return {
                "draw_pct": draws / total,
                "home_advantage": (home_wins - away_wins) / total,
                "avg_goals": avg_goals_season,
            }
        except Exception:
            logger.debug("_compute_season_context_from_db: failed for season_id=%s", season_id)
        return {}

    @staticmethod
    def _normalize_odds(odds_1: float, odds_x: float, odds_2: float) -> dict[str, float]:
        """
        Normalize 1X2 odds to implied probabilities.
        
        Formula: implied_prob_i = (1 / odds_i) / sum(1/odds_j)
        This removes bookmaker margin and gives "fair" probabilities.
        """
        eps = 1e-12
        try:
            inv_1 = 1.0 / (float(odds_1) + eps)
            inv_x = 1.0 / (float(odds_x) + eps)
            inv_2 = 1.0 / (float(odds_2) + eps)
        except (ZeroDivisionError, TypeError):
            # Fallback: equal probabilities
            return {
                "implied_prob_1": 1.0 / 3,
                "implied_prob_x": 1.0 / 3,
                "implied_prob_2": 1.0 / 3,
                "gap": 0.0,
                "volatility": 0.0,
            }

        total = inv_1 + inv_x + inv_2
        if total <= 0:
            return {
                "implied_prob_1": 1.0 / 3,
                "implied_prob_x": 1.0 / 3,
                "implied_prob_2": 1.0 / 3,
                "gap": 0.0,
                "volatility": 0.0,
            }

        probs = [float(inv_1 / total), float(inv_x / total), float(inv_2 / total)]
        probs_sorted = sorted(probs, reverse=True)
        gap = float(probs_sorted[0] - probs_sorted[1]) if len(probs_sorted) >= 2 else 0.0
        
        # Volatility: std dev of probabilities
        mean_prob = sum(probs) / len(probs)
        variance = sum((p - mean_prob) ** 2 for p in probs) / len(probs)
        volatility = float(variance ** 0.5)

        return {
            "implied_prob_1": probs[0],
            "implied_prob_x": probs[1],
            "implied_prob_2": probs[2],
            "gap": gap,
            "volatility": volatility,
        }

    @staticmethod
    def _calc_entropy(probs: list[float]) -> float:
        """Calculate Shannon entropy (0 = certain, 1 = max uncertainty for 3 classes)."""
        import math
        eps = 1e-12
        entropy = -sum(p * math.log(max(p, eps), 2) for p in probs if p > 0.0)
        max_entropy = math.log(3, 2)  # log2(3) for 3 classes
        if max_entropy <= 0:
            return 0.0
        return float(entropy / max_entropy)

    def _resolve_pool_probs(
        self,
        match: dict[str, Any],
        pool_probs: dict[str, float] | None,
    ) -> dict[str, float]:
        """Resolve and normalise pool probabilities from explicit arg or match payload."""
        if isinstance(pool_probs, dict) and pool_probs:
            return self._normalize_probs(pool_probs)
        pool_probs_raw = match.get("pool_probs", {}) if isinstance(match, dict) else {}
        if not isinstance(pool_probs_raw, dict):
            pool_probs_raw = {}
        return self._normalize_probs({
            "P1": pool_probs_raw.get("P1", 1.0 / 3),
            "PX": pool_probs_raw.get("PX", 1.0 / 3),
            "P2": pool_probs_raw.get("P2", 1.0 / 3),
        })

    def _extract_market_inputs(self, match: dict[str, Any]) -> dict[str, Any]:
        """
        Detect and validate market inputs from a TOTO match payload.

        Supports:
          - Direct odds fields: odds_ft_1/x/2
          - Nested odds dict with aliases: odds.O1/OX/O2
          - TotoBrief bookmaker quotes: bk_win_1/bk_draw/bk_win_2
            (top-level in TotoBrief enriched payload or inside bookmaker_quotes sub-dict)
        """
        base: dict[str, Any] = {
            "has_odds": False,
            "has_bookmaker_quotes": False,
            "odds_ft_1": None,
            "odds_ft_x": None,
            "odds_ft_2": None,
            "bk_win_1": None,
            "bk_draw": None,
            "bk_win_2": None,
        }
        if not isinstance(match, dict):
            return base

        odds_payload = match.get("odds")
        odds_dict = odds_payload if isinstance(odds_payload, dict) else {}
        raw_o1 = match.get("odds_ft_1") or odds_dict.get("odds_ft_1") or odds_dict.get("O1")
        raw_ox = match.get("odds_ft_x") or odds_dict.get("odds_ft_x") or odds_dict.get("OX")
        raw_o2 = match.get("odds_ft_2") or odds_dict.get("odds_ft_2") or odds_dict.get("O2")
        if raw_o1 is not None and raw_ox is not None and raw_o2 is not None:
            try:
                o1, ox, o2 = float(raw_o1), float(raw_ox), float(raw_o2)
                if o1 > 1.0 and ox > 1.0 and o2 > 1.0:
                    base.update({"has_odds": True, "odds_ft_1": o1, "odds_ft_x": ox, "odds_ft_2": o2})
            except (TypeError, ValueError):
                pass

        bk_src = match.get("bookmaker_quotes", {})
        if not isinstance(bk_src, dict):
            bk_src = {}
        raw_bk1 = match.get("bk_win_1") or bk_src.get("bk_win_1")
        raw_bkx = match.get("bk_draw") or bk_src.get("bk_draw")
        raw_bk2 = match.get("bk_win_2") or bk_src.get("bk_win_2")
        if raw_bk1 is not None or raw_bkx is not None or raw_bk2 is not None:
            try:
                bk1, bkx, bk2 = float(raw_bk1 or 0), float(raw_bkx or 0), float(raw_bk2 or 0)
                if bk1 + bkx + bk2 > 0:
                    base.update({"has_bookmaker_quotes": True, "bk_win_1": bk1, "bk_draw": bkx, "bk_win_2": bk2})
            except (TypeError, ValueError):
                pass

        return base

    def _probs_from_bookmaker_quotes(self, market_inputs: dict[str, Any]) -> dict[str, float] | None:
        """
        Convert TotoBrief bookmaker quotes to normalised probabilities.
        bk_win_1/bk_draw/bk_win_2 are percentages (0-100); handles both percent and fraction.
        Returns None when quotes are absent or sum to zero.
        """
        bk1 = self._safe_float(market_inputs.get("bk_win_1"))
        bkx = self._safe_float(market_inputs.get("bk_draw"))
        bk2 = self._safe_float(market_inputs.get("bk_win_2"))
        total = bk1 + bkx + bk2
        if total <= 0:
            return None
        # Detect percentage representation (0-100) vs fraction (0-1).
        divisor = 100.0 if total > 1.5 else 1.0
        return self._normalize_probs({"P1": bk1 / divisor, "PX": bkx / divisor, "P2": bk2 / divisor})

    def _build_target_distribution(self, rows: list[dict[str, Any]]) -> dict[str, int]:
        home = 0
        draw = 0
        away = 0
        for row in rows:
            target = int(row.get("target", -1))
            if target == 1:
                home += 1
            elif target == 0:
                draw += 1
            elif target == 2:
                away += 1
        return {
            "home_win_count": home,
            "draw_count": draw,
            "away_win_count": away,
        }

    def _build_feature_profile(
        self,
        rows: list[dict[str, Any]],
        feature_columns: list[str],
    ) -> dict[str, Any]:
        total = len(rows)
        if total <= 0:
            return {
                "weak_features": [],
                "healthy_features": [],
                "per_feature": {},
            }

        per_feature: dict[str, dict[str, Any]] = {}
        for feature in feature_columns:
            non_null = 0
            zero_like = 0
            fallback_like = 0
            values: list[float] = []

            for row in rows:
                value = row.get(feature)
                if value is None:
                    continue
                non_null += 1
                try:
                    fv = float(value)
                except (TypeError, ValueError):
                    continue
                values.append(fv)
                if abs(fv) <= 1e-12:
                    zero_like += 1

                source_key = f"__source_{feature}"
                source = str(row.get(source_key, "")).strip().lower()
                # "missing_sentinel" = API returned -1/-2 sentinel (still not real data)
                if source in ("fallback_default", "fallback", "default", "hard_default", "missing_sentinel"):
                    fallback_like += 1

            fill_rate = non_null / total if total else 0.0
            zero_rate = zero_like / non_null if non_null else 1.0
            fallback_rate = fallback_like / non_null if non_null else 0.0
            diversity = len({round(v, 6) for v in values})

            per_feature[feature] = {
                "non_null_count": non_null,
                "zero_like_count": zero_like,
                "fallback_like_count": fallback_like,
                "fill_rate": round(fill_rate, 4),
                "zero_rate": round(zero_rate, 4),
                "fallback_rate": round(fallback_rate, 4),
                "distinct_values": diversity,
            }

        weak_rank = sorted(
            per_feature.items(),
            key=lambda item: (
                item[1]["fill_rate"],
                -item[1]["fallback_rate"],
                -item[1]["zero_rate"],
                item[1]["distinct_values"],
            ),
        )
        healthy_rank = sorted(
            per_feature.items(),
            key=lambda item: (
                -item[1]["fill_rate"],
                item[1]["zero_rate"],
                -item[1]["distinct_values"],
            ),
        )

        weak_features = [
            {
                "feature": name,
                **stats,
            }
            for name, stats in weak_rank[:5]
            if stats["fill_rate"] < 0.95 or stats["fallback_rate"] > 0.6 or stats["zero_rate"] > 0.75 or stats["distinct_values"] <= 3
        ]
        healthy_features = [
            {
                "feature": name,
                **stats,
            }
            for name, stats in healthy_rank[:5]
            if stats["fill_rate"] >= 0.95 and stats["fallback_rate"] < 0.2 and stats["zero_rate"] < 0.25 and stats["distinct_values"] > 5
        ]

        return {
            "weak_features": weak_features,
            "healthy_features": healthy_features,
            "per_feature": per_feature,
        }

    def _build_predict_quality_summary(self, predict_summary: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(predict_summary, dict):
            return {
                "predicted_count": 0,
                "failed_count": 0,
                "skipped_count": 0,
                "usable_prediction_rate": 0.0,
                "note": "predict summary not provided",
            }

        predicted = int(predict_summary.get("predicted_count", 0) or 0)
        failed = int(predict_summary.get("failed_count", 0) or 0)
        skipped = int(predict_summary.get("skipped_count", 0) or 0)
        total = predicted + failed + skipped
        usable_rate = (predicted / total) if total > 0 else 0.0

        return {
            "predicted_count": predicted,
            "failed_count": failed,
            "skipped_count": skipped,
            "usable_prediction_rate": round(usable_rate, 4),
            "total_evaluated": total,
        }

    def _build_validation_summary_not_run(self, reason: str) -> dict[str, Any]:
        return {
            "validation_run": False,
            "reason": reason,
            "validation_rows": 0,
            "predicted_rows": 0,
            "skipped_rows": 0,
            "accuracy": None,
            "top1_hit_rate": None,
            "double_outcome_hit_rate": None,
            "average_confidence": None,
            "log_loss": None,
        }

    def _run_time_based_validation(self, trainer: ModelTrainer, cleaned_rows: list[dict[str, Any]]) -> dict[str, Any]:
        if len(cleaned_rows) < 120:
            return self._build_validation_summary_not_run("not enough cleaned rows for holdout validation (<120)")

        try:
            x_train, x_valid, y_train, y_valid = trainer.split_chronological(cleaned_rows, train_ratio=0.8)
            model = LGBMClassifier(
                objective="multiclass",
                num_class=3,
                learning_rate=0.05,
                n_estimators=220,
                max_depth=-1,
                random_state=42,
            )
            model.fit(x_train, y_train)

            probabilities = model.predict_proba(x_valid)
            predictions = np.argmax(probabilities, axis=1)
            max_conf = np.max(probabilities, axis=1)
            valid_rows = int(len(y_valid))

            top2_indices = np.argsort(probabilities, axis=1)[:, -2:]
            margin = np.take_along_axis(probabilities, top2_indices[:, 1:2], axis=1) - np.take_along_axis(
                probabilities,
                top2_indices[:, 0:1],
                axis=1,
            )
            is_double_case = (np.abs(margin).reshape(-1) <= 0.15)
            double_hits = 0
            double_total = 0
            for idx, flag in enumerate(is_double_case):
                if not bool(flag):
                    continue
                double_total += 1
                if int(y_valid[idx]) in set(int(v) for v in top2_indices[idx]):
                    double_hits += 1

            return {
                "validation_run": True,
                "reason": "time-based holdout",
                "validation_rows": valid_rows,
                "predicted_rows": valid_rows,
                "skipped_rows": 0,
                "accuracy": round(float(accuracy_score(y_valid, predictions)), 4),
                "top1_hit_rate": round(float(accuracy_score(y_valid, predictions)), 4),
                "double_outcome_hit_rate": round((double_hits / double_total), 4) if double_total > 0 else None,
                "double_outcome_cases": int(double_total),
                "average_confidence": round(float(np.mean(max_conf)), 4) if valid_rows > 0 else None,
                "log_loss": round(float(log_loss(y_valid, probabilities, labels=[0, 1, 2])), 4),
                "feature_mode": ",".join(trainer.feature_columns[:3]) if trainer.feature_columns else "unknown",
            }
        except Exception as exc:
            logger.warning("validation_summary: holdout failed: %s", exc)
            return self._build_validation_summary_not_run(f"validation failed: {exc}")

    def _build_post_train_backtest_not_run(self, reason: str) -> dict[str, Any]:
        return {
            "backtest_run": False,
            "reason": reason,
            "split_mode": "time_based_holdout",
            "train_ratio": 0.8,
            "total_eval_matches": 0,
            "evaluated_matches": 0,
            "skipped_matches": 0,
            "skipped_reasons": {},
            "metrics": {
                "accuracy": None,
                "log_loss": None,
                "average_confidence": None,
                "average_calibrated_confidence": None,
                "brier_score": None,
            },
            "class_metrics": {
                "actual_counts": {"1": 0, "X": 0, "2": 0},
                "predicted_counts": {"1": 0, "X": 0, "2": 0},
                "hit_rate": {"1": None, "X": None, "2": None},
                "confusion": {
                    "1": {"1": 0, "X": 0, "2": 0},
                    "X": {"1": 0, "X": 0, "2": 0},
                    "2": {"1": 0, "X": 0, "2": 0},
                },
            },
            "context_metrics": {
                "full_context": {"count": 0, "accuracy": None},
                "partial_context": {"count": 0, "accuracy": None},
                "degraded_context": {"count": 0, "accuracy": None},
                "no_odds_mode": {"count": 0, "accuracy": None},
            },
            "signal_quality": {
                "strong_signal": {"count": 0, "accuracy": None, "avg_confidence": None},
                "medium_signal": {"count": 0, "accuracy": None, "avg_confidence": None},
                "weak_signal": {"count": 0, "accuracy": None, "avg_confidence": None},
                "confidence_buckets": {
                    "high_confidence": {"count": 0, "accuracy": None, "avg_confidence": None},
                    "mid_confidence": {"count": 0, "accuracy": None, "avg_confidence": None},
                    "low_confidence": {"count": 0, "accuracy": None, "avg_confidence": None},
                },
            },
            "market_vs_stats": {
                "agreement_count": 0,
                "disagreement_count": 0,
                "accuracy_when_aligned": None,
                "accuracy_when_disagreeing": None,
                "suspicious_market_disagreement_count": 0,
                "stats_override_signal_count": 0,
                "weak_favorite_flag_count": 0,
                "draw_risk_flag_count": 0,
            },
            "summary_lines": [
                f"Backtest not run: {reason}",
            ],
        }

    def _run_post_train_backtest(
        self,
        raw_rows: list[dict[str, Any]],
        feature_columns: list[str],
        train_ratio: float = 0.8,
    ) -> dict[str, Any]:
        if not raw_rows:
            return self._build_post_train_backtest_not_run("empty dataset")
        if not feature_columns:
            return self._build_post_train_backtest_not_run("feature columns unavailable")
        if len(raw_rows) < 120:
            return self._build_post_train_backtest_not_run("not enough rows for post-train holdout (<120)")

        rows = sorted(
            raw_rows,
            key=lambda row: str(row.get("match_date") or row.get("timestamp") or ""),
        )
        split_idx = int(len(rows) * train_ratio)
        split_idx = max(1, min(split_idx, len(rows) - 1))
        eval_rows = rows[split_idx:]
        total_eval = len(eval_rows)

        class_to_label = {1: "1", 0: "X", 2: "2"}
        outcome_to_class = {"P1": 1, "PX": 0, "P2": 2}
        eval_targets: list[int] = []
        eval_prob_vectors: list[list[float]] = []
        raw_confidences: list[float] = []
        calibrated_confidences: list[float] = []
        skipped_reasons: dict[str, int] = {}

        actual_counts = {"1": 0, "X": 0, "2": 0}
        predicted_counts = {"1": 0, "X": 0, "2": 0}
        class_hits = {"1": 0, "X": 0, "2": 0}
        confusion = {
            "1": {"1": 0, "X": 0, "2": 0},
            "X": {"1": 0, "X": 0, "2": 0},
            "2": {"1": 0, "X": 0, "2": 0},
        }

        context_hits: dict[str, int] = {"full_context": 0, "partial_context": 0, "degraded_context": 0, "no_odds_mode": 0}
        context_counts: dict[str, int] = {"full_context": 0, "partial_context": 0, "degraded_context": 0, "no_odds_mode": 0}

        signal_hits: dict[str, int] = {"strong_signal": 0, "medium_signal": 0, "weak_signal": 0}
        signal_counts: dict[str, int] = {"strong_signal": 0, "medium_signal": 0, "weak_signal": 0}
        signal_conf_sum: dict[str, float] = {"strong_signal": 0.0, "medium_signal": 0.0, "weak_signal": 0.0}

        bucket_hits: dict[str, int] = {"high_confidence": 0, "mid_confidence": 0, "low_confidence": 0}
        bucket_counts: dict[str, int] = {"high_confidence": 0, "mid_confidence": 0, "low_confidence": 0}
        bucket_conf_sum: dict[str, float] = {"high_confidence": 0.0, "mid_confidence": 0.0, "low_confidence": 0.0}

        aligned_count = 0
        aligned_hits = 0
        disagreement_count = 0
        disagreement_hits = 0
        suspicious_count = 0
        stats_override_count = 0
        weak_favorite_count = 0
        draw_risk_count = 0

        for row in eval_rows:
            skip_reason: str | None = None

            target_raw = row.get("target")
            try:
                target = int(target_raw)
            except (TypeError, ValueError):
                target = -1
            if target not in (0, 1, 2):
                skip_reason = "invalid_target"

            features: dict[str, float] = {}
            if skip_reason is None:
                for feature_name in feature_columns:
                    value = row.get(feature_name)
                    if value is None:
                        skip_reason = f"missing_feature:{feature_name}"
                        break
                    try:
                        value_f = float(value)
                    except (TypeError, ValueError):
                        skip_reason = f"non_numeric:{feature_name}"
                        break
                    if np.isnan(value_f) or np.isinf(value_f):
                        skip_reason = f"nan_or_inf:{feature_name}"
                        break
                    features[feature_name] = value_f

            if skip_reason is not None:
                skipped_reasons[skip_reason] = skipped_reasons.get(skip_reason, 0) + 1
                continue

            try:
                probs = self.predict(features)
            except Exception:
                skipped_reasons["prediction_error"] = skipped_reasons.get("prediction_error", 0) + 1
                continue

            pred_outcome = max(probs, key=probs.get)
            pred_class = outcome_to_class[pred_outcome]

            quality = self._build_backtest_row_quality(
                row=row,
                features=features,
                probs=probs,
                feature_columns=feature_columns,
            )
            context_bucket = quality["context_bucket"]
            signal_strength = quality["signal_strength"]
            raw_conf = float(quality["raw_confidence"])
            calibrated_conf = float(quality["calibrated_confidence"])
            confidence_bucket = quality["confidence_bucket"]

            eval_targets.append(target)
            eval_prob_vectors.append([float(probs["PX"]), float(probs["P1"]), float(probs["P2"])])
            raw_confidences.append(raw_conf)
            calibrated_confidences.append(calibrated_conf)

            actual_label = class_to_label[target]
            pred_label = class_to_label[pred_class]
            actual_counts[actual_label] += 1
            predicted_counts[pred_label] += 1
            confusion[actual_label][pred_label] += 1

            hit = pred_class == target
            if hit:
                class_hits[actual_label] += 1

            context_counts[context_bucket] = context_counts.get(context_bucket, 0) + 1
            if hit:
                context_hits[context_bucket] = context_hits.get(context_bucket, 0) + 1

            signal_counts[signal_strength] = signal_counts.get(signal_strength, 0) + 1
            signal_conf_sum[signal_strength] = signal_conf_sum.get(signal_strength, 0.0) + raw_conf
            if hit:
                signal_hits[signal_strength] = signal_hits.get(signal_strength, 0) + 1

            bucket_counts[confidence_bucket] = bucket_counts.get(confidence_bucket, 0) + 1
            bucket_conf_sum[confidence_bucket] = bucket_conf_sum.get(confidence_bucket, 0.0) + raw_conf
            if hit:
                bucket_hits[confidence_bucket] = bucket_hits.get(confidence_bucket, 0) + 1

            if bool(quality["market_agreement"]):
                aligned_count += 1
                if hit:
                    aligned_hits += 1
            else:
                disagreement_count += 1
                if hit:
                    disagreement_hits += 1

            if bool(quality["suspicious_market_disagreement"]):
                suspicious_count += 1
            if bool(quality["stats_override_signal"]):
                stats_override_count += 1
            if bool(quality["weak_favorite_flag"]):
                weak_favorite_count += 1
            if bool(quality["draw_risk_flag"]):
                draw_risk_count += 1

        evaluated = len(eval_targets)
        skipped = total_eval - evaluated
        if evaluated <= 0:
            return {
                **self._build_post_train_backtest_not_run("all holdout rows were skipped"),
                "total_eval_matches": total_eval,
                "skipped_matches": skipped,
                "skipped_reasons": skipped_reasons,
            }

        y_true = np.array(eval_targets, dtype=int)
        y_prob = np.array(eval_prob_vectors, dtype=float)
        y_pred = np.argmax(y_prob, axis=1)

        accuracy = float(accuracy_score(y_true, y_pred))
        try:
            ll = float(log_loss(y_true, y_prob, labels=[0, 1, 2]))
        except Exception:
            ll = None  # type: ignore[assignment]

        one_hot = np.zeros_like(y_prob)
        one_hot[np.arange(evaluated), y_true] = 1.0
        brier = float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1) / y_prob.shape[1]))

        def _acc(hits: int, count: int) -> float | None:
            return round((hits / count), 4) if count > 0 else None

        context_metrics = {
            key: {
                "count": int(context_counts.get(key, 0)),
                "accuracy": _acc(context_hits.get(key, 0), context_counts.get(key, 0)),
            }
            for key in ("full_context", "partial_context", "degraded_context", "no_odds_mode")
        }

        signal_quality = {
            key: {
                "count": int(signal_counts.get(key, 0)),
                "accuracy": _acc(signal_hits.get(key, 0), signal_counts.get(key, 0)),
                "avg_confidence": round(signal_conf_sum.get(key, 0.0) / signal_counts.get(key, 1), 4)
                if signal_counts.get(key, 0) > 0
                else None,
            }
            for key in ("strong_signal", "medium_signal", "weak_signal")
        }
        signal_quality["confidence_buckets"] = {
            key: {
                "count": int(bucket_counts.get(key, 0)),
                "accuracy": _acc(bucket_hits.get(key, 0), bucket_counts.get(key, 0)),
                "avg_confidence": round(bucket_conf_sum.get(key, 0.0) / bucket_counts.get(key, 1), 4)
                if bucket_counts.get(key, 0) > 0
                else None,
            }
            for key in ("high_confidence", "mid_confidence", "low_confidence")
        }

        market_vs_stats = {
            "agreement_count": int(aligned_count),
            "disagreement_count": int(disagreement_count),
            "accuracy_when_aligned": _acc(aligned_hits, aligned_count),
            "accuracy_when_disagreeing": _acc(disagreement_hits, disagreement_count),
            "suspicious_market_disagreement_count": int(suspicious_count),
            "stats_override_signal_count": int(stats_override_count),
            "weak_favorite_flag_count": int(weak_favorite_count),
            "draw_risk_flag_count": int(draw_risk_count),
        }

        ll_text = f"{ll:.4f}" if ll is not None else "n/a"

        summary_lines = [
            f"Holdout backtest evaluated {evaluated}/{total_eval} matches (skipped={skipped}).",
            f"Accuracy={accuracy:.4f}, LogLoss={ll_text}, Brier={brier:.4f}.",
            f"Context: full={context_metrics['full_context']['accuracy']}, partial={context_metrics['partial_context']['accuracy']}, degraded={context_metrics['degraded_context']['accuracy']}, no_odds={context_metrics['no_odds_mode']['accuracy']}.",
            f"Signal buckets: strong={signal_quality['strong_signal']['accuracy']}, medium={signal_quality['medium_signal']['accuracy']}, weak={signal_quality['weak_signal']['accuracy']}.",
            f"Market disagreement: count={market_vs_stats['disagreement_count']} accuracy={market_vs_stats['accuracy_when_disagreeing']} suspicious={market_vs_stats['suspicious_market_disagreement_count']}.",
        ]

        class_metrics = {
            "actual_counts": actual_counts,
            "predicted_counts": predicted_counts,
            "hit_rate": {
                label: _acc(class_hits[label], actual_counts[label])
                for label in ("1", "X", "2")
            },
            "confusion": confusion,
        }

        return {
            "backtest_run": True,
            "reason": "ok",
            "split_mode": "time_based_holdout",
            "train_ratio": train_ratio,
            "total_eval_matches": total_eval,
            "evaluated_matches": evaluated,
            "skipped_matches": skipped,
            "skipped_reasons": skipped_reasons,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "log_loss": round(ll, 4) if ll is not None else None,
                "average_confidence": round(float(np.mean(raw_confidences)), 4),
                "average_calibrated_confidence": round(float(np.mean(calibrated_confidences)), 4),
                "brier_score": round(brier, 4),
            },
            "class_metrics": class_metrics,
            "context_metrics": context_metrics,
            "signal_quality": signal_quality,
            "market_vs_stats": market_vs_stats,
            "summary_lines": summary_lines,
        }

    def _build_backtest_row_quality(
        self,
        *,
        row: dict[str, Any],
        features: dict[str, float],
        probs: dict[str, float],
        feature_columns: list[str],
    ) -> dict[str, Any]:
        signal_non_market = [
            name for name in feature_columns
            if name not in self.MARKET_FEATURES and name not in {"entropy", "gap", "volatility"}
        ]

        def _has_real_signal(name: str) -> bool:
            source_key = f"__source_{name}"
            source = str(row.get(source_key, "") or "").strip().lower()
            if source.startswith("real_") or source.startswith("db_lookup"):
                return True
            value = features.get(name)
            if value is None:
                return False
            try:
                return abs(float(value)) > 1e-12
            except (TypeError, ValueError):
                return False

        available_non_market = sum(1 for name in signal_non_market if _has_real_signal(name))
        total_non_market = len(signal_non_market)
        non_market_ratio = (available_non_market / total_non_market) if total_non_market > 0 else 0.0

        has_odds = bool(
            features.get("odds_ft_1", 0.0) > 1.0
            and features.get("odds_ft_x", 0.0) > 1.0
            and features.get("odds_ft_2", 0.0) > 1.0
        )
        if not has_odds:
            context_bucket = "no_odds_mode"
        elif non_market_ratio >= 0.75:
            context_bucket = "full_context"
        elif non_market_ratio >= 0.35:
            context_bucket = "partial_context"
        else:
            context_bucket = "degraded_context"

        market_probs = {
            "P1": float(features.get("implied_prob_1", 1.0 / 3.0)),
            "PX": float(features.get("implied_prob_x", 1.0 / 3.0)),
            "P2": float(features.get("implied_prob_2", 1.0 / 3.0)),
        }
        market_total = market_probs["P1"] + market_probs["PX"] + market_probs["P2"]
        if market_total > 0:
            market_probs = {k: v / market_total for k, v in market_probs.items()}
        else:
            market_probs = {"P1": 1.0 / 3.0, "PX": 1.0 / 3.0, "P2": 1.0 / 3.0}

        raw_conf = max(float(probs["P1"]), float(probs["PX"]), float(probs["P2"]))
        conf_mult = {
            "full_context": 1.00,
            "partial_context": 0.94,
            "degraded_context": 0.86,
            "no_odds_mode": 0.82,
        }.get(context_bucket, 0.86)
        calibrated_conf = max(1.0 / 3.0, min(1.0, raw_conf * conf_mult))

        quality_score = {
            "full_context": 0.82,
            "partial_context": 0.60,
            "degraded_context": 0.34,
            "no_odds_mode": 0.24,
        }.get(context_bucket, 0.34)
        quality_score = max(0.0, min(1.0, (0.75 * quality_score) + (0.25 * non_market_ratio)))

        market_favorite = max(market_probs, key=market_probs.get)
        model_favorite = max(probs, key=probs.get)
        market_agreement = model_favorite == market_favorite

        model_gain_vs_market = float(probs[model_favorite]) - float(market_probs.get(model_favorite, 1.0 / 3.0))
        market_disagreement = not market_agreement
        stats_override_signal = bool(
            market_disagreement
            and context_bucket in {"full_context", "partial_context"}
            and model_gain_vs_market >= 0.07
        )
        suspicious_market_disagreement = bool(
            market_disagreement and not stats_override_signal and quality_score < 0.60
        )

        weak_favorite_flag = bool(
            market_favorite != "PX"
            and float(market_probs[market_favorite]) < 0.50
            and float(features.get("gap", 0.0)) < 0.12
        )
        draw_risk_flag = bool(
            float(probs["PX"]) >= 0.34
            and float(market_probs["PX"]) <= 0.30
            and (float(features.get("draw_pct", 0.0)) >= 0.27 or float(features.get("entropy", 0.0)) >= 0.80)
        )

        if quality_score >= 0.72 and calibrated_conf >= 0.56:
            signal_strength = "strong_signal"
        elif quality_score >= 0.45 and calibrated_conf >= 0.44:
            signal_strength = "medium_signal"
        else:
            signal_strength = "weak_signal"

        if raw_conf >= 0.60:
            conf_bucket = "high_confidence"
        elif raw_conf >= 0.48:
            conf_bucket = "mid_confidence"
        else:
            conf_bucket = "low_confidence"

        return {
            "context_bucket": context_bucket,
            "signal_strength": signal_strength,
            "raw_confidence": raw_conf,
            "calibrated_confidence": calibrated_conf,
            "quality_score": quality_score,
            "market_agreement": market_agreement,
            "market_disagreement": market_disagreement,
            "suspicious_market_disagreement": suspicious_market_disagreement,
            "stats_override_signal": stats_override_signal,
            "weak_favorite_flag": weak_favorite_flag,
            "draw_risk_flag": draw_risk_flag,
            "confidence_bucket": conf_bucket,
        }

    def _build_db_audit_summary(self, database: Any) -> dict[str, Any]:
        summary = {
            "total_matches": 0,
            "completed_matches": 0,
            "matches_with_odds": 0,
            "matches_without_odds": 0,
            "team_season_stats_rows": 0,
            "league_season_stats_rows": 0,
            "unique_leagues": 0,
            "unique_teams": 0,
        }
        conn = getattr(database, "conn", None)
        if conn is None:
            return summary

        try:
            summary["total_matches"] = int(conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
            summary["completed_matches"] = int(conn.execute("SELECT COUNT(*) FROM matches WHERE status = 'completed'").fetchone()[0])
            summary["matches_with_odds"] = int(
                conn.execute(
                    "SELECT COUNT(*) FROM matches WHERE odds_ft_1 IS NOT NULL AND odds_ft_x IS NOT NULL AND odds_ft_2 IS NOT NULL"
                ).fetchone()[0]
            )
            summary["matches_without_odds"] = max(summary["total_matches"] - summary["matches_with_odds"], 0)
            summary["team_season_stats_rows"] = int(conn.execute("SELECT COUNT(*) FROM team_season_stats").fetchone()[0])
            summary["league_season_stats_rows"] = int(conn.execute("SELECT COUNT(*) FROM league_season_stats").fetchone()[0])
            summary["unique_leagues"] = int(conn.execute("SELECT COUNT(DISTINCT season_id) FROM matches").fetchone()[0])
            summary["unique_teams"] = int(
                conn.execute(
                    "SELECT COUNT(DISTINCT team_id) FROM (SELECT home_team_id AS team_id FROM matches UNION ALL SELECT away_team_id AS team_id FROM matches)"
                ).fetchone()[0]
            )
        except Exception as exc:
            logger.warning("db_input_summary failed: %s", exc)

        return summary

    def _build_quality_warnings(
        self,
        completed_matches_found: int,
        dataset_rows_raw: int,
        training_rows: int,
        target_distribution: dict[str, int],
        weak_features: list[dict[str, Any]],
        predict_summary: dict[str, Any] | None,
    ) -> list[str]:
        warnings: list[str] = []

        if completed_matches_found < 50:
            warnings.append("dataset is very small (<50 completed matches)")
        elif completed_matches_found < 500:
            warnings.append("dataset is below 500 completed matches; quality estimate is unstable")

        if dataset_rows_raw > 0:
            dropped = max(dataset_rows_raw - training_rows, 0)
            drop_rate = dropped / dataset_rows_raw
            if drop_rate > 0.3:
                warnings.append("high dropped_rows rate (>30%) between raw dataset and training rows")

        total_targets = (
            target_distribution.get("home_win_count", 0)
            + target_distribution.get("draw_count", 0)
            + target_distribution.get("away_win_count", 0)
        )
        if total_targets > 0:
            draw_count = target_distribution.get("draw_count", 0)
            if draw_count == 0:
                warnings.append("draw class is missing in training rows")
            else:
                draw_share = draw_count / total_targets
                if draw_share < 0.08:
                    warnings.append("draw class is underrepresented (<8%)")

            home_share = target_distribution.get("home_win_count", 0) / total_targets
            away_share = target_distribution.get("away_win_count", 0) / total_targets
            if home_share > 0.7 or away_share > 0.7:
                warnings.append("class distribution is heavily imbalanced towards one side")

        if weak_features:
            warnings.append("several features look weak/default-heavy; review weak_features list")
            top = weak_features[0]
            if isinstance(top, dict) and top.get("feature"):
                warnings.append(
                    f"top weak feature: {top.get('feature')} (fallback_rate={top.get('fallback_rate', 0)}, zero_rate={top.get('zero_rate', 0)})"
                )

        predict_quality = self._build_predict_quality_summary(predict_summary)
        usable_rate = float(predict_quality.get("usable_prediction_rate", 0.0) or 0.0)
        total_eval = int(predict_quality.get("total_evaluated", 0) or 0)
        if total_eval > 0 and usable_rate < 0.6:
            warnings.append("usable prediction rate is low (<60%)")

        if completed_matches_found > 0 and training_rows > 0:
            # Honest note for current project stage: some engineered features may still be default-heavy.
            warnings.append("feature contract is satisfied, but part of features may still be default-heavy")

        return warnings

    def _normalize_probs(self, probs: dict[str, Any]) -> dict[str, float]:
        """Нормализовать dict вероятностей к P1/PX/P2 с суммой 1.0."""
        p1 = self._safe_float(probs.get("P1"))
        px = self._safe_float(probs.get("PX"))
        p2 = self._safe_float(probs.get("P2"))

        total = p1 + px + p2
        if total <= 0:
            return {"P1": 1.0 / 3, "PX": 1.0 / 3, "P2": 1.0 / 3}

        return {
            "P1": p1 / total,
            "PX": px / total,
            "P2": p2 / total,
        }

    @staticmethod
    def _safe_float(value: Any) -> float:
        """Безопасно конвертировать любое значение в float, default 0.0."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _refresh_runtime_state(self) -> None:
        self.model_file_exists = self.model_path.exists()
        self.feature_schema_loaded = self.schema_path.exists()
        self.models_loaded = self.predictor.model is not None
        self.predictor_trained = bool(getattr(self.predictor, "predictor_trained", False))

    def get_model_readiness(self) -> dict[str, Any]:
        """Single source of truth for model runtime readiness across UI tabs."""
        self._refresh_runtime_state()
        ready = bool(
            self.model_file_exists
            and self.feature_schema_loaded
            and self.models_loaded
            and self.predictor_trained
        )
        return {
            "model_file_exists": self.model_file_exists,
            "feature_schema_loaded": self.feature_schema_loaded,
            "models_loaded": self.models_loaded,
            "predictor_trained": self.predictor_trained,
            "ready": ready,
        }
