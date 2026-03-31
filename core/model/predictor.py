from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.model.calibrator import ProbabilityCalibrator
from core.model.feature_schema import FeatureSchema
from core.model.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Strict wrapper around trained multiclass model inference."""

    MARKET_FEATURES = {
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
    }
    _CLASS_TO_OUTCOME = {1: "P1", 0: "PX", 2: "P2"}
    DRAW_PROMOTION_MAX_BOOST = 0.16

    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_columns: list[str] | None = None
        self.model_path: Path | None = None
        self.raw_model_path: Path | None = None
        self.calibrated_model_path: Path | None = None
        self.using_calibrated_model: bool = False
        self.trained: bool = False
        self.predictor_trained: bool = False
        self.schema = FeatureSchema()

    @property
    def is_ready(self) -> bool:
        return bool(self.trained and self.model is not None and self.feature_columns is not None)

    def _set_not_ready(self) -> None:
        self.model = None
        self.feature_columns = None
        self.trained = False
        self.predictor_trained = False
        self.using_calibrated_model = False

    def _set_ready(self) -> None:
        self.trained = True
        self.predictor_trained = True

    def train(
        self,
        dataset: list[dict[str, Any]],
        model_path: str | Path = "data/models/model.pkl",
        calibrate: bool = False,
    ) -> Any:
        """Train via existing ModelTrainer and make predictor immediately ready for inference.

        This is a thin compatibility bridge to restore one-entry lifecycle:
        train -> save artifact -> load runtime state -> predict.
        """
        destination = Path(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        self._set_not_ready()

        trainer = ModelTrainer(model_path=str(destination))
        model = trainer.train(dataset)
        trainer.save(model, destination)

        if calibrate:
            x_train, x_valid, y_train, y_valid = trainer.split_chronological(dataset)
            calibrator_path = destination.parent / "calibrated_model.pkl"
            calibrator = ProbabilityCalibrator(model_path=str(calibrator_path))
            try:
                calibrator.fit(model, x_valid, y_valid)
            except Exception as exc:
                # Keep runtime usable with raw model fallback.
                logger.warning("Calibration failed; continuing with raw model fallback. error=%s", exc)

        # Re-load from artifacts to keep a single runtime path and consistent readiness state.
        self.load(destination.parent)
        return self.model

    def load(self, path: str | Path = "data/models") -> None:
        self._set_not_ready()

        path_obj = Path(path)
        model_dir = path_obj if path_obj.is_dir() else path_obj.parent

        raw_model_path = model_dir / "model.pkl"
        calibrated_model_path = model_dir / "calibrated_model.pkl"
        schema_path = model_dir / "feature_schema.json"

        if path_obj.is_file():
            raw_model_path = path_obj
            model_dir = raw_model_path.parent
            calibrated_model_path = model_dir / "calibrated_model.pkl"
            schema_path = model_dir / "feature_schema.json"

        if not raw_model_path.exists():
            msg = f"Model file not found: {raw_model_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        model_to_load = calibrated_model_path if calibrated_model_path.exists() else raw_model_path
        loaded_model = joblib.load(model_to_load)
        loaded_feature_columns = self.schema.load(str(schema_path))

        self.model = loaded_model
        self.feature_columns = loaded_feature_columns
        self.model_path = model_to_load
        self.raw_model_path = raw_model_path
        self.calibrated_model_path = calibrated_model_path
        self.using_calibrated_model = calibrated_model_path.exists()
        self._set_ready()

        logger.info("Using %s model: %s", "calibrated" if self.using_calibrated_model else "raw", model_to_load)

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        if not self.is_ready:
            msg = "Model is not loaded. Call load(path) before predict()."
            logger.error(msg)
            raise RuntimeError(msg)

        self.schema.validate(features, self.feature_columns)

        row = np.array([[float(features[name]) for name in self.feature_columns]], dtype=float)
        if np.isnan(row).any() or np.isinf(row).any():
            msg = "Input features contain NaN or infinite values."
            logger.error(msg)
            raise ValueError(msg)

        probs = self.model.predict_proba(row)
        if probs.ndim != 2 or probs.shape != (1, 3):
            msg = f"Invalid predict_proba shape: {probs.shape}. Expected (1, 3)."
            logger.error(msg)
            raise ValueError(msg)

        values = probs[0]
        if np.isnan(values).any():
            msg = f"Prediction contains NaN values: {values.tolist()}"
            logger.error(msg)
            raise ValueError(msg)

        result = self._map_probabilities_to_outcomes(values)
        result = self._apply_draw_balance_adjustment(features=features, probs=result)
        self._validate_probabilities(result)
        logger.info("Prediction values: %s", result)
        return result

    def _apply_draw_balance_adjustment(
        self,
        *,
        features: dict[str, float],
        probs: dict[str, float],
    ) -> dict[str, float]:
        """Apply a narrow PX uplift only for genuinely draw-like close games.

        This correction is intentionally conservative and is gated by:
        - close top-2 market/model structure (no strong favorite),
        - multiple draw-like signals from existing features,
        - small PX-to-top gap.
        """
        p1 = float(probs.get("P1", 0.0))
        px = float(probs.get("PX", 0.0))
        p2 = float(probs.get("P2", 0.0))

        top_non_draw = max(p1, p2)
        if top_non_draw <= px:
            return probs

        margin_12 = abs(p1 - p2)
        gap_to_top = top_non_draw - px

        market_p1 = self._safe_feature_value(features, "implied_prob_1", default=1.0 / 3.0)
        market_px = self._safe_feature_value(features, "implied_prob_x", default=1.0 / 3.0)
        market_p2 = self._safe_feature_value(features, "implied_prob_2", default=1.0 / 3.0)
        market_total = market_p1 + market_px + market_p2
        if market_total > 0.0:
            market_px /= market_total

        should_promote_draw = bool(
            margin_12 <= 0.10
            and gap_to_top <= 0.14
            and market_px >= 0.26
            and self._safe_feature_value(features, "draw_pct") >= 0.22
            and self._safe_feature_value(features, "entropy") >= 0.72
            and self._safe_feature_value(features, "gap") <= 0.20
            and abs(self._safe_feature_value(features, "ppg_diff")) <= 0.85
            and abs(self._safe_feature_value(features, "xg_diff")) <= 0.65
        )
        if not should_promote_draw:
            return probs

        boost = min(self.DRAW_PROMOTION_MAX_BOOST, max(0.0, (top_non_draw - px) + 0.0040))
        if boost <= 0.0:
            return probs

        adjusted = {
            "P1": p1,
            "PX": px + boost,
            "P2": p2,
        }
        total = adjusted["P1"] + adjusted["PX"] + adjusted["P2"]
        if total <= 0.0:
            return probs
        return {
            "P1": adjusted["P1"] / total,
            "PX": adjusted["PX"] / total,
            "P2": adjusted["P2"] / total,
        }

    def predict_with_diagnostics(
        self,
        features: dict[str, float],
        *,
        allow_no_odds_fallback: bool = True,
        min_non_odds_features: int = 6,
    ) -> dict[str, Any]:
        """Predict with optional no-odds fallback and explicit diagnostics.

        This does not replace strict `predict()`; it provides a safe fallback API
        for callers that want to process rows without bookmaker odds.
        """
        if not self.is_ready:
            return {
                "status": "skipped",
                "source": "predict_error",
                "reason": "model_not_ready",
                "probs": None,
                "no_odds_mode": False,
                "feature_diagnostics": {
                    "available_non_odds_features": 0,
                    "required_non_odds_features": 0,
                    "non_odds_sufficient": False,
                },
            }

        row, mode, diagnostics = self._build_runtime_row(
            features=features,
            allow_no_odds_fallback=allow_no_odds_fallback,
            min_non_odds_features=min_non_odds_features,
        )

        if row is None:
            reason = (
                "Нет кф и недостаточно статистики"
                if diagnostics.get("reason_code") == "no_odds_insufficient_non_odds_features"
                else "Нет кф, fallback-прогноз невозможен"
            )
            return {
                "status": "skipped",
                "source": "no_odds_unavailable" if diagnostics.get("missing_market_features") else "feature_error",
                "reason": reason,
                "reason_code": diagnostics.get("reason_code"),
                "probs": None,
                "no_odds_mode": bool(diagnostics.get("missing_market_features")),
                "feature_diagnostics": diagnostics,
            }

        probs = self._predict_from_row(row)
        if mode == "no_odds_fallback":
            probs = self._apply_no_odds_caution(probs, blend=0.18)
        elif mode == "partial_no_odds_fallback":
            # Stronger caution: fewer features available, predictions less reliable.
            probs = self._apply_no_odds_caution(probs, blend=0.30)

        is_no_odds = mode in ("no_odds_fallback", "partial_no_odds_fallback")
        prediction_quality = self._build_prediction_quality(
            features=features,
            diagnostics=diagnostics,
            probs=probs,
            mode=mode,
        )
        return {
            "status": "predicted",
            "source": "no_odds_fallback" if is_no_odds else "model_runtime",
            "reason": "ok",
            "probs": probs,
            "no_odds_mode": is_no_odds,
            "feature_diagnostics": diagnostics,
            "prediction_quality": prediction_quality,
        }

    def _predict_from_row(self, row: np.ndarray) -> dict[str, float]:
        probs = self.model.predict_proba(row)
        if probs.ndim != 2 or probs.shape != (1, 3):
            msg = f"Invalid predict_proba shape: {probs.shape}. Expected (1, 3)."
            logger.error(msg)
            raise ValueError(msg)

        values = probs[0]
        if np.isnan(values).any():
            msg = f"Prediction contains NaN values: {values.tolist()}"
            logger.error(msg)
            raise ValueError(msg)

        result = self._map_probabilities_to_outcomes(values)
        self._validate_probabilities(result)
        return result

    def _map_probabilities_to_outcomes(self, values: np.ndarray) -> dict[str, float]:
        """Map predict_proba output to P1/PX/P2 using model.classes_.

        Business mapping is fixed by target semantics:
        - class 1 -> P1 (home)
        - class 0 -> PX (draw)
        - class 2 -> P2 (away)
        """
        classes = getattr(self.model, "classes_", None)
        if classes is None:
            msg = "Model does not expose classes_; cannot map probabilities safely."
            logger.error(msg)
            raise ValueError(msg)

        try:
            class_to_index = {int(cls): idx for idx, cls in enumerate(classes)}
        except Exception as exc:
            msg = f"Failed to parse model classes_: {classes}."
            logger.error("%s error=%s", msg, exc)
            raise ValueError(msg) from exc

        missing = [cls for cls in self._CLASS_TO_OUTCOME if cls not in class_to_index]
        if missing:
            msg = f"Model classes_ missing required classes: {missing}. classes_={list(classes)}"
            logger.error(msg)
            raise ValueError(msg)

        return {
            outcome: float(values[class_to_index[cls]])
            for cls, outcome in self._CLASS_TO_OUTCOME.items()
        }

    def _build_runtime_row(
        self,
        features: dict[str, float],
        *,
        allow_no_odds_fallback: bool,
        min_non_odds_features: int,
    ) -> tuple[np.ndarray | None, str, dict[str, Any]]:
        assert self.feature_columns is not None

        diagnostics = self._collect_feature_availability(features, min_non_odds_features=min_non_odds_features)
        missing_market = diagnostics["missing_market_features"]
        missing_non_market = diagnostics["missing_non_odds_features"]
        non_odds_sufficient = bool(diagnostics["non_odds_sufficient"])

        use_no_odds_fallback = bool(
            allow_no_odds_fallback
            and missing_market
            and not missing_non_market
            and non_odds_sufficient
        )

        # Relaxed no-odds fallback: allow prediction when odds are absent but at least 3
        # non-market features (e.g. PPG) are available, even if up to 10 stats are missing.
        # A stronger caution blend (0.30) is applied to counteract the sparse input.
        use_partial_no_odds = bool(
            allow_no_odds_fallback
            and missing_market
            and missing_non_market
            and len(diagnostics["missing_non_odds_features"]) <= 10
            and diagnostics["available_non_odds_features"] >= 3
        )

        if missing_non_market and not use_partial_no_odds:
            diagnostics["reason_code"] = "missing_non_market_features"
            return None, "skipped", diagnostics

        if missing_market and not use_no_odds_fallback and not use_partial_no_odds:
            diagnostics["reason_code"] = "no_odds_insufficient_non_odds_features"
            return None, "skipped", diagnostics

        row_values: list[float] = []
        for name in self.feature_columns:
            raw = features.get(name)
            if raw is None and name in self.MARKET_FEATURES and (use_no_odds_fallback or use_partial_no_odds):
                raw = self._default_market_feature_value(name)

            if raw is None:
                # For partial no-odds, allow missing non-market stats to default to 0.0
                if use_partial_no_odds and name not in self.MARKET_FEATURES:
                    raw = 0.0
                else:
                    diagnostics["reason_code"] = f"missing_required:{name}"
                    return None, "skipped", diagnostics

            try:
                value = float(raw)
            except (TypeError, ValueError):
                diagnostics["reason_code"] = f"non_numeric:{name}"
                return None, "skipped", diagnostics

            if np.isnan(value) or np.isinf(value):
                diagnostics["reason_code"] = f"nan_or_inf:{name}"
                return None, "skipped", diagnostics

            row_values.append(value)

        row = np.array([row_values], dtype=float)
        mode = (
            "no_odds_fallback"
            if use_no_odds_fallback
            else "partial_no_odds_fallback"
            if use_partial_no_odds
            else "market_assisted"
        )
        return row, mode, diagnostics

    def _collect_feature_availability(
        self,
        features: dict[str, float],
        *,
        min_non_odds_features: int,
    ) -> dict[str, Any]:
        assert self.feature_columns is not None

        available_market: list[str] = []
        missing_market: list[str] = []
        available_non_market: list[str] = []
        missing_non_market: list[str] = []

        for name in self.feature_columns:
            value = features.get(name)
            is_available = False
            if value is not None:
                try:
                    value_f = float(value)
                    is_available = not (np.isnan(value_f) or np.isinf(value_f))
                except (TypeError, ValueError):
                    is_available = False

            if name in self.MARKET_FEATURES:
                (available_market if is_available else missing_market).append(name)
            else:
                (available_non_market if is_available else missing_non_market).append(name)

        total_non_market = len(available_non_market) + len(missing_non_market)
        non_market_ratio = (len(available_non_market) / total_non_market) if total_non_market > 0 else 0.0
        context_level = self._classify_feature_context_level(
            has_market_features=bool(available_market),
            total_non_market_count=total_non_market,
            available_non_market_count=len(available_non_market),
        )
        market_heavy_flag = bool(
            total_non_market <= 0 or len(available_non_market) <= max(1, total_non_market // 3)
        )
        non_market_rich_flag = bool(
            total_non_market > 0 and len(available_non_market) >= max(3, int(total_non_market * 0.6))
        )

        return {
            "available_market_features": len(available_market),
            "required_market_features": len(available_market) + len(missing_market),
            "missing_market_features": missing_market,
            "available_non_odds_features": len(available_non_market),
            "required_non_odds_features": len(available_non_market) + len(missing_non_market),
            "missing_non_odds_features": missing_non_market,
            "non_odds_sufficient": len(available_non_market) >= max(1, min_non_odds_features),
            "context_level": context_level,
            "non_market_ratio": non_market_ratio,
            "market_heavy_flag": market_heavy_flag,
            "non_market_rich_flag": non_market_rich_flag,
        }

    @staticmethod
    def _classify_feature_context_level(
        *,
        has_market_features: bool,
        total_non_market_count: int,
        available_non_market_count: int,
    ) -> str:
        if not has_market_features:
            return "degraded_context"
        if total_non_market_count <= 0 or available_non_market_count <= 0:
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
                return {
                    "P1": p1 / total,
                    "PX": px / total,
                    "P2": p2 / total,
                }

        o1 = self._safe_feature_value(features, "odds_ft_1", default=0.0)
        ox = self._safe_feature_value(features, "odds_ft_x", default=0.0)
        o2 = self._safe_feature_value(features, "odds_ft_2", default=0.0)
        if o1 > 1.0 and ox > 1.0 and o2 > 1.0:
            inv_1 = 1.0 / o1
            inv_x = 1.0 / ox
            inv_2 = 1.0 / o2
            total = inv_1 + inv_x + inv_2
            if total > 0.0:
                return {
                    "P1": inv_1 / total,
                    "PX": inv_x / total,
                    "P2": inv_2 / total,
                }

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

    def _build_prediction_quality(
        self,
        *,
        features: dict[str, float],
        diagnostics: dict[str, Any],
        probs: dict[str, float],
        mode: str,
    ) -> dict[str, Any]:
        context_level = str(diagnostics.get("context_level") or "degraded_context")
        non_market_ratio = float(diagnostics.get("non_market_ratio") or 0.0)
        market_heavy_flag = bool(diagnostics.get("market_heavy_flag"))

        base_scores = {
            "full_context": 0.85,
            "partial_context": 0.62,
            "odds_only_context": 0.36,
            "degraded_context": 0.24,
        }
        quality_score = base_scores.get(context_level, 0.24)
        quality_score = (0.70 * quality_score) + (0.25 * non_market_ratio) + (0.05 if diagnostics.get("non_market_rich_flag") else 0.0)
        if market_heavy_flag:
            quality_score -= 0.08
        if mode == "no_odds_fallback":
            quality_score -= 0.10
        elif mode == "partial_no_odds_fallback":
            quality_score -= 0.16
        quality_score = max(0.0, min(1.0, quality_score))

        ordered = sorted(((key, float(value)) for key, value in probs.items()), key=lambda item: item[1], reverse=True)
        raw_confidence = float(ordered[0][1]) if ordered else (1.0 / 3.0)
        prediction_margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) > 1 else 0.0

        confidence_multiplier = {
            "full_context": 1.00,
            "partial_context": 0.94,
            "odds_only_context": 0.88,
            "degraded_context": 0.82,
        }.get(context_level, 0.82)
        if mode == "no_odds_fallback":
            confidence_multiplier *= 0.92
        elif mode == "partial_no_odds_fallback":
            confidence_multiplier *= 0.86
        if market_heavy_flag:
            confidence_multiplier *= 0.95
        calibrated_confidence = max(1.0 / 3.0, min(1.0, raw_confidence * confidence_multiplier))

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

        if quality_score >= 0.72 and calibrated_confidence >= 0.56 and prediction_margin >= 0.10:
            signal_strength = "strong_signal"
        elif quality_score >= 0.45 and calibrated_confidence >= 0.44:
            signal_strength = "medium_signal"
        else:
            signal_strength = "weak_signal"

        return {
            "context_level": context_level,
            "signal_strength": signal_strength,
            "quality_score": round(quality_score, 4),
            "raw_confidence": round(raw_confidence, 4),
            "calibrated_confidence": round(calibrated_confidence, 4),
            "confidence_delta": round(raw_confidence - calibrated_confidence, 4),
            "prediction_margin": round(prediction_margin, 4),
            "market_alignment_score": round(market_alignment_score, 4),
            "market_disagreement_flag": market_disagreement_flag,
            "suspicious_market_disagreement_flag": suspicious_market_disagreement_flag,
            "weak_favorite_flag": weak_favorite_flag,
            "draw_risk_flag": draw_risk_flag,
            "stats_override_signal": stats_override_signal,
            "market_favorite_outcome": market_favorite,
            "stats_context_favorite_outcome": stats_favorite,
        }

    @staticmethod
    def _default_market_feature_value(feature_name: str) -> float:
        if feature_name.startswith("odds_ft_"):
            return 3.0
        # Neutral implied prior in no-odds mode.
        return 1.0 / 3.0

    @staticmethod
    def _apply_no_odds_caution(probs: dict[str, float], blend: float = 0.18) -> dict[str, float]:
        """Blend prediction towards uniform to reduce overconfidence in no-odds mode."""
        blend = min(0.40, max(0.0, float(blend)))
        neutral = 1.0 / 3.0
        adjusted = {
            "P1": (1.0 - blend) * float(probs["P1"]) + blend * neutral,
            "PX": (1.0 - blend) * float(probs["PX"]) + blend * neutral,
            "P2": (1.0 - blend) * float(probs["P2"]) + blend * neutral,
        }
        total = adjusted["P1"] + adjusted["PX"] + adjusted["P2"]
        if total <= 0.0:
            return {"P1": neutral, "PX": neutral, "P2": neutral}
        return {
            "P1": adjusted["P1"] / total,
            "PX": adjusted["PX"] / total,
            "P2": adjusted["P2"] / total,
        }

    @staticmethod
    def _validate_probabilities(result: dict[str, float], tolerance: float = 1e-6) -> None:
        values = np.array(list(result.values()), dtype=float)

        if np.isnan(values).any():
            msg = f"Probability contains NaN: {result}"
            logger.error(msg)
            raise ValueError(msg)

        if np.any(values < 0.0) or np.any(values > 1.0):
            msg = f"Probability value is outside [0, 1]: {result}"
            logger.error(msg)
            raise ValueError(msg)

        total = float(np.sum(values))
        if abs(total - 1.0) > tolerance:
            msg = f"Probability sum is invalid: {total}."
            logger.error(msg)
            raise ValueError(msg)


class Predictor(ModelPredictor):
    """Backward-compatible alias."""
