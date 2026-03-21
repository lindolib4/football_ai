from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.utils._param_validation import InvalidParameterError
from sklearn.frozen import FrozenEstimator

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Sigmoid probability calibration for a pre-trained multiclass classifier."""

    def __init__(self, model_path: str = "data/models/calibrated_model.pkl") -> None:
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibrated_model: CalibratedClassifierCV | None = None

    def fit(self, model: LGBMClassifier, X_valid: np.ndarray, y_valid: np.ndarray) -> CalibratedClassifierCV:
        """Fit Platt scaling (sigmoid) calibrator on validation split."""
        raw_proba = model.predict_proba(X_valid)
        raw_log_loss = float(log_loss(y_valid, raw_proba, labels=[0, 1, 2]))
        logger.info("Calibration log_loss before: %.6f", raw_log_loss)

        try:
            calibrated = CalibratedClassifierCV(
                base_estimator=model,
                method="sigmoid",
                cv="prefit",
            )
            calibrated.fit(X_valid, y_valid)
        except (TypeError, InvalidParameterError):
            calibrated = CalibratedClassifierCV(
                estimator=FrozenEstimator(model),
                method="sigmoid",
                cv=None,
            )
            calibrated.fit(X_valid, y_valid)

        calibrated_proba = calibrated.predict_proba(X_valid)
        calibrated_log_loss = float(log_loss(y_valid, calibrated_proba, labels=[0, 1, 2]))
        logger.info("Calibration log_loss after: %.6f", calibrated_log_loss)

        self.calibrated_model = calibrated
        self.save(calibrated)
        return calibrated

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities of shape [n, 3] without NaN values."""
        if self.calibrated_model is None:
            raise RuntimeError("Calibrated model is not fitted.")

        probabilities = self.calibrated_model.predict_proba(X)

        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError(f"Expected probabilities of shape [n, 3], got {probabilities.shape}.")

        if np.isnan(probabilities).any():
            raise ValueError("Calibrated probabilities contain NaN values.")

        sums = probabilities.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-6):
            raise ValueError("Calibrated probability sum is invalid.")

        return probabilities

    def save(self, calibrated_model: CalibratedClassifierCV | None = None, path: str | Path | None = None) -> None:
        model_to_save = calibrated_model if calibrated_model is not None else self.calibrated_model
        if model_to_save is None:
            raise RuntimeError("No calibrated model to save.")

        destination = Path(path) if path is not None else self.model_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_to_save, destination)
        logger.info("Calibrated model saved to %s", destination)


__all__ = ["ProbabilityCalibrator"]
