from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from core.model.calibrator import ProbabilityCalibrator
from lightgbm import LGBMClassifier


def _build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = make_classification(
        n_samples=1500,
        n_features=18,
        n_informative=10,
        n_redundant=4,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.2,
        random_state=42,
    )

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def test_calibrator_predict_proba_shape_and_no_nan(tmp_path) -> None:
    x_train, y_train, x_valid, y_valid, x_test, _ = _build_dataset()

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=250,
        learning_rate=0.08,
        random_state=42,
    )
    model.fit(x_train, y_train)

    calibrator = ProbabilityCalibrator(model_path=str(tmp_path / "calibrated_model.pkl"))
    calibrator.fit(model, x_valid, y_valid)

    proba = calibrator.predict_proba(x_test)

    assert proba.shape == (x_test.shape[0], 3)
    assert not np.isnan(proba).any()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_calibration_log_loss_not_much_worse(tmp_path) -> None:
    x_train, y_train, x_valid, y_valid, x_test, y_test = _build_dataset()

    model = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=250,
        learning_rate=0.08,
        random_state=42,
    )
    model.fit(x_train, y_train)

    raw_proba = model.predict_proba(x_test)
    raw_loss = float(log_loss(y_test, raw_proba, labels=[0, 1, 2]))

    calibrator = ProbabilityCalibrator(model_path=str(tmp_path / "calibrated_model.pkl"))
    calibrator.fit(model, x_valid, y_valid)
    calibrated_proba = calibrator.predict_proba(x_test)
    calibrated_loss = float(log_loss(y_test, calibrated_proba, labels=[0, 1, 2]))

    assert calibrated_loss <= raw_loss + 0.03
