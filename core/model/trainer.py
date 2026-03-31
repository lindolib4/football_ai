from __future__ import annotations

from collections import Counter
import logging
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, log_loss

from core.features.builder import FeatureBuilder
from core.model.feature_schema import FeatureSchema

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate a single prematch LightGBM multiclass model."""

    BASELINE_FEATURE_CANDIDATES = (
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "ppg_diff",
        "home_home_ppg",
        "away_away_ppg",
        "split_advantage",
        "draw_pct",
        "home_advantage",
        "avg_goals",
        "entropy",
        "gap",
        "volatility",
    )

    def __init__(self, model_path: str = "data/models/model.pkl") -> None:
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        template = FeatureBuilder.build_features({}, {}, {}, {})
        self.all_feature_columns = self._extract_numeric_feature_columns(template)
        self.source_metadata_columns = [name for name in template.keys() if name.startswith("__source_")]
        self.baseline_feature_columns = [
            name for name in self.BASELINE_FEATURE_CANDIDATES if name in self.all_feature_columns
        ]
        self.feature_columns = list(self.all_feature_columns)
        self.last_clean_report: dict[str, Any] = {}

    def clean_data(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop rows with invalid target/feature values and auto-fallback to baseline-safe features."""
        cleaned, report = self.clean_data_with_report(dataset)
        self.last_clean_report = report
        return cleaned

    def clean_data_with_report(self, dataset: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return cleaned rows and a diagnostic report for the train pipeline."""
        cleaned_extended, report_extended = self._clean_data_for_columns(dataset, self.all_feature_columns)
        report_extended["baseline_candidate_rows_after_clean"] = 0
        report_extended["fallback_from_extended"] = False

        if cleaned_extended:
            self.feature_columns = list(self.all_feature_columns)
            report_extended["training_feature_mode"] = "extended"
            report_extended["active_feature_count"] = len(self.feature_columns)
            return cleaned_extended, report_extended

        cleaned_baseline, report_baseline = self._clean_data_for_columns(dataset, self.baseline_feature_columns)
        report_baseline["fallback_from_extended"] = bool(cleaned_baseline)
        report_baseline["baseline_candidate_rows_after_clean"] = len(cleaned_baseline)

        if cleaned_baseline:
            self.feature_columns = list(self.baseline_feature_columns)
            report_baseline["training_feature_mode"] = "baseline"
            report_baseline["active_feature_count"] = len(self.feature_columns)
            report_baseline["extended_candidate_rows_after_clean"] = 0
            logger.warning(
                "Extended feature set produced no clean rows; falling back to baseline-safe feature set. baseline_rows=%s",
                len(cleaned_baseline),
            )
            return cleaned_baseline, report_baseline

        self.feature_columns = list(self.all_feature_columns)
        report_extended["training_feature_mode"] = "extended"
        report_extended["active_feature_count"] = len(self.feature_columns)
        report_extended["baseline_candidate_rows_after_clean"] = len(cleaned_baseline)
        return [], report_extended

    def _clean_data_for_columns(
        self,
        dataset: list[dict[str, Any]],
        feature_columns: list[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        missing_counter: Counter[str] = Counter()
        sample_filtered_rows: list[dict[str, Any]] = []
        dropped_due_to_invalid_target = 0
        dropped_due_to_missing_required = 0
        dropped_due_to_non_numeric = 0
        dropped_due_to_nan = 0
        dropped_due_to_postmatch = 0

        for row in dataset:
            if "target" not in row:
                dropped_due_to_invalid_target += 1
                self._append_filter_sample(sample_filtered_rows, row, "missing_target")
                continue

            target = row.get("target")
            if target is None:
                dropped_due_to_invalid_target += 1
                self._append_filter_sample(sample_filtered_rows, row, "target_is_none")
                continue
            try:
                target_int = int(target)
            except (TypeError, ValueError):
                dropped_due_to_invalid_target += 1
                self._append_filter_sample(sample_filtered_rows, row, f"invalid_target:{target}")
                continue
            if target_int not in (0, 1, 2):
                dropped_due_to_invalid_target += 1
                self._append_filter_sample(sample_filtered_rows, row, f"target_out_of_range:{target_int}")
                continue

            if self._contains_postmatch_signals(row):
                dropped_due_to_postmatch += 1
                self._append_filter_sample(sample_filtered_rows, row, "postmatch_signal_detected")
                continue

            cleaned_row: dict[str, Any] = {"target": target_int}
            invalid_reason: str | None = None

            for feature_name in feature_columns:
                value = row.get(feature_name)
                if value is None:
                    dropped_due_to_missing_required += 1
                    missing_counter[feature_name] += 1
                    invalid_reason = f"missing_required:{feature_name}"
                    break

                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    dropped_due_to_non_numeric += 1
                    invalid_reason = f"non_numeric:{feature_name}"
                    break

                if math.isnan(value_f) or math.isinf(value_f):
                    dropped_due_to_nan += 1
                    invalid_reason = f"nan_or_inf:{feature_name}"
                    break

                cleaned_row[feature_name] = value_f

            if invalid_reason is not None:
                self._append_filter_sample(sample_filtered_rows, row, invalid_reason)
                continue

            if "match_date" in row:
                cleaned_row["match_date"] = row["match_date"]
            elif "timestamp" in row:
                cleaned_row["match_date"] = row["timestamp"]

            cleaned.append(cleaned_row)

        report: dict[str, Any] = {
            "training_rows_before_clean": len(dataset),
            "training_rows_after_clean": len(cleaned),
            "dropped_due_to_invalid_target": dropped_due_to_invalid_target,
            "dropped_due_to_missing_required": dropped_due_to_missing_required,
            "dropped_due_to_non_numeric": dropped_due_to_non_numeric,
            "dropped_due_to_nan": dropped_due_to_nan,
            "dropped_due_to_postmatch": dropped_due_to_postmatch,
            "required_features_missing_top": [
                {"feature": feature, "count": count}
                for feature, count in missing_counter.most_common(10)
            ],
            "sample_filtered_rows_reasons": sample_filtered_rows,
            "candidate_feature_count": len(feature_columns),
        }
        return cleaned, report

    def prepare_dataset(self, dataset: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        """Validate and convert list[dict] dataset into model matrices."""
        cleaned = self.clean_data(dataset)
        dataset_size = len(cleaned)

        if dataset_size == 0:
            raise ValueError("Dataset is empty after cleaning.")

        if dataset_size < 500:
            logger.warning("Dataset is small for stable training: %s matches (<500).", dataset_size)

        x = np.array([[row[name] for name in self.feature_columns] for row in cleaned], dtype=float)
        y = np.array([row["target"] for row in cleaned], dtype=int)

        logger.info("Dataset size: %s", dataset_size)
        logger.info("Feature count: %s", x.shape[1])
        return x, y

    def split_chronological(
        self,
        dataset: list[dict[str, Any]],
        train_ratio: float = 0.8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Chronological split: first part train, latest part validation."""
        cleaned = self.clean_data(dataset)
        if not cleaned:
            raise ValueError("Dataset is empty after cleaning.")

        if all("match_date" in row for row in cleaned):
            cleaned = sorted(cleaned, key=lambda row: str(row["match_date"]))

        split_idx = int(len(cleaned) * train_ratio)
        split_idx = max(1, min(split_idx, len(cleaned) - 1))

        train_rows = cleaned[:split_idx]
        valid_rows = cleaned[split_idx:]

        x_train = np.array([[row[name] for name in self.feature_columns] for row in train_rows], dtype=float)
        y_train = np.array([row["target"] for row in train_rows], dtype=int)
        x_valid = np.array([[row[name] for name in self.feature_columns] for row in valid_rows], dtype=float)
        y_valid = np.array([row["target"] for row in valid_rows], dtype=int)

        logger.info("Train size: %s", len(train_rows))
        logger.info("Valid size: %s", len(valid_rows))
        return x_train, x_valid, y_train, y_valid

    def train(self, dataset: list[dict[str, Any]]) -> LGBMClassifier:
        x_train, x_valid, y_train, y_valid = self.split_chronological(dataset)

        model = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            learning_rate=0.05,
            n_estimators=300,
            max_depth=-1,
            random_state=42,
        )
        model.fit(x_train, y_train)

        metrics = self.evaluate(model, x_valid, y_valid)
        logger.info("Accuracy: %.4f", metrics["accuracy"])
        logger.info("Log loss: %.4f", metrics["log_loss"])

        return model

    def evaluate(self, model: LGBMClassifier, x_valid: np.ndarray, y_valid: np.ndarray) -> dict[str, float | dict[str, int]]:
        probabilities = model.predict_proba(x_valid)
        predictions = np.argmax(probabilities, axis=1)

        metrics: dict[str, float | dict[str, int]] = {
            "accuracy": float(accuracy_score(y_valid, predictions)),
            "log_loss": float(log_loss(y_valid, probabilities, labels=[0, 1, 2])),
            "prediction_distribution": {
                "1": int(np.sum(predictions == 1)),
                "X": int(np.sum(predictions == 0)),
                "2": int(np.sum(predictions == 2)),
            },
        }
        return metrics

    def save(self, model: LGBMClassifier, path: str | Path | None = None) -> None:
        destination = Path(path) if path is not None else self.model_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, destination)
        logger.info("Model saved to %s", destination)

        schema_path = destination.parent / "feature_schema.json"
        FeatureSchema().save(self.feature_columns, str(schema_path))

    @staticmethod
    def _extract_numeric_feature_columns(template: dict[str, Any]) -> list[str]:
        return [name for name in template.keys() if not name.startswith("__source_")]

    @staticmethod
    def _append_filter_sample(samples: list[dict[str, Any]], row: dict[str, Any], reason: str) -> None:
        if len(samples) >= 10:
            return
        samples.append(
            {
                "reason": reason,
                "match_date": row.get("match_date") or row.get("timestamp"),
                "target": row.get("target"),
            }
        )

    @staticmethod
    def _contains_postmatch_signals(row: dict[str, Any]) -> bool:
        banned_fragments = (
            "final_score",
            "full_time_score",
            "result",
            "winner",
            "goals_home_ft",
            "goals_away_ft",
        )
        lowered_keys = [key.lower() for key in row.keys()]
        return any(fragment in key for key in lowered_keys for fragment in banned_fragments)
