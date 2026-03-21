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
        self.raw_model_path: Path | None = None
        self.calibrated_model_path: Path | None = None
        self.using_calibrated_model: bool = False

    def load(self, path: str | Path = "data/models") -> None:
        path_obj = Path(path)
        model_dir = path_obj if path_obj.is_dir() else path_obj.parent
        raw_model_path = model_dir / "model.pkl"
        calibrated_model_path = model_dir / "calibrated_model.pkl"

        if path_obj.is_file():
            raw_model_path = path_obj
            model_dir = raw_model_path.parent
            calibrated_model_path = model_dir / "calibrated_model.pkl"

        if not raw_model_path.exists():
            msg = f"Model file not found: {raw_model_path}"
            logger.exception(msg)
            raise FileNotFoundError(msg)

        model_to_load = calibrated_model_path if calibrated_model_path.exists() else raw_model_path
        self.model = joblib.load(model_to_load)

        self.model_path = model_to_load
        self.raw_model_path = raw_model_path
        self.calibrated_model_path = calibrated_model_path
        self.using_calibrated_model = calibrated_model_path.exists()

        logger.info(
            "Loaded %s model: %s",
            "calibrated" if self.using_calibrated_model else "raw",
            model_to_load,
        )

        self.feature_columns = self._load_feature_schema(raw_model_path)

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
        if any(value < 0 or value > 1 for value in values):
            msg = f"Probability value is outside [0, 1]: {result}"
            logger.exception(msg)
            raise ValueError(msg)

        total = float(sum(values))
        if abs(total - 1.0) > tolerance:
            msg = f"Probability sum is invalid: {total}."
            logger.exception(msg)
            raise ValueError(msg)


class Predictor(ModelPredictor):
    """Backward-compatible alias."""
