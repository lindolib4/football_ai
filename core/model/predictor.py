from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from core.model.feature_schema import FeatureSchema

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
        self.schema = FeatureSchema()

    def load(self, path: str | Path = "data/models") -> None:
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
        self.model = joblib.load(model_to_load)

        self.model_path = model_to_load
        self.raw_model_path = raw_model_path
        self.calibrated_model_path = calibrated_model_path
        self.using_calibrated_model = calibrated_model_path.exists()
        logger.info("Using %s model: %s", "calibrated" if self.using_calibrated_model else "raw", model_to_load)

        self.feature_columns = self.schema.load(str(schema_path))

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        if self.model is None or self.feature_columns is None:
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

        result = {"P1": float(values[0]), "PX": float(values[1]), "P2": float(values[2])}
        self._validate_probabilities(result)
        logger.info("Prediction values: %s", result)
        return result

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
