from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.features.builder import FeatureBuilder

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Strict wrapper around trained multiclass model inference."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_columns: list[str] | None = None
        self.model_path: Path | None = None

    def load(self, path: str | Path) -> None:
        model_path = Path(path)
        if model_path.is_dir():
            model_path = model_path / "model.pkl"

        if not model_path.exists():
            logger.exception("Model file not found: %s", model_path)
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.model_path = model_path
        self.feature_columns = self._load_feature_schema(model_path)

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        if self.model is None:
            logger.exception("Model is not loaded. Call load(path) before predict().")
            raise RuntimeError("Model is not loaded. Call load(path) before predict().")

        expected = self.feature_columns or list(FeatureBuilder.build_features({}, {}, {}, {}).keys())
        self._validate_features(features, expected)

        row = np.array([[float(features[name]) for name in expected]], dtype=float)
        probs = self.model.predict_proba(row)[0]

        result = {"P1": float(probs[0]), "PX": float(probs[1]), "P2": float(probs[2])}
        self._validate_probabilities(result)
        return result

    def _load_feature_schema(self, model_path: Path) -> list[str]:
        schema_json = model_path.with_name("feature_schema.json")
        if schema_json.exists():
            with schema_json.open("r", encoding="utf-8") as fp:
                payload = json.load(fp)

            if isinstance(payload, dict) and isinstance(payload.get("feature_columns"), list):
                return [str(name) for name in payload["feature_columns"]]
            if isinstance(payload, list):
                return [str(name) for name in payload]

        schema_pkl = model_path.with_name("feature_schema.pkl")
        if schema_pkl.exists():
            payload = joblib.load(schema_pkl)
            if isinstance(payload, list):
                return [str(name) for name in payload]
            if isinstance(payload, dict) and isinstance(payload.get("feature_columns"), list):
                return [str(name) for name in payload["feature_columns"]]

        return list(FeatureBuilder.build_features({}, {}, {}, {}).keys())

    @staticmethod
    def _validate_features(features: dict[str, float], expected: list[str]) -> None:
        incoming = set(features.keys())
        expected_set = set(expected)

        missing = sorted(expected_set - incoming)
        extra = sorted(incoming - expected_set)
        if missing or extra:
            msg = f"Feature mismatch. Missing: {missing}. Extra: {extra}."
            logger.exception(msg)
            raise ValueError(msg)

        for name in expected:
            value = features[name]
            value_f = float(value)
            if np.isnan(value_f):
                msg = f"Feature '{name}' contains NaN."
                logger.exception(msg)
                raise ValueError(msg)

    @staticmethod
    def _validate_probabilities(result: dict[str, float], tolerance: float = 1e-6) -> None:
        values = list(result.values())
        if any(value < 0 for value in values):
            msg = f"Negative probability detected: {result}"
            logger.exception(msg)
            raise ValueError(msg)

        total = float(sum(values))
        if abs(total - 1.0) > tolerance:
            msg = f"Probability sum is invalid: {total}."
            logger.exception(msg)
            raise ValueError(msg)


class Predictor(ModelPredictor):
    """Backward-compatible alias."""
