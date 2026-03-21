from __future__ import annotations

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

    def __init__(self, model_path: str = "data/models/model.pkl") -> None:
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_columns = list(FeatureBuilder.build_features({}, {}, {}, {}).keys())

    def clean_data(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop rows with invalid target/feature values (None/NaN/inf)."""
        cleaned: list[dict[str, Any]] = []

        for row in dataset:
            if "target" not in row:
                continue

            target = row.get("target")
            if target is None:
                continue
            try:
                target_int = int(target)
            except (TypeError, ValueError):
                continue
            if target_int not in (0, 1, 2):
                continue

            cleaned_row: dict[str, Any] = {"target": target_int}
            invalid = False

            for feature_name in self.feature_columns:
                value = row.get(feature_name)
                if value is None:
                    invalid = True
                    break

                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    invalid = True
                    break

                if math.isnan(value_f) or math.isinf(value_f):
                    invalid = True
                    break

                cleaned_row[feature_name] = value_f

            if invalid:
                continue

            if self._contains_postmatch_signals(row):
                raise ValueError("Dataset contains post-match fields. Only prematch features are allowed.")

            if "match_date" in row:
                cleaned_row["match_date"] = row["match_date"]
            elif "timestamp" in row:
                cleaned_row["match_date"] = row["timestamp"]

            cleaned.append(cleaned_row)

        return cleaned

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
                "1": int(np.sum(predictions == 0)),
                "X": int(np.sum(predictions == 1)),
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
