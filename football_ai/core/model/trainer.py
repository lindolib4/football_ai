from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split


class ModelTrainer:
    def __init__(self, model_path: str = "football_ai/data/models/model.pkl") -> None:
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
        base_model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=250, learning_rate=0.05)
        base_model.fit(x_train, y_train)
        cv_scores = cross_val_score(base_model, x_train, y_train, cv=3, scoring="accuracy")

        calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
        calibrated.fit(x_train, y_train)
        pred = calibrated.predict(x_test)
        acc = accuracy_score(y_test, pred)

        with self.model_path.open("wb") as fp:
            pickle.dump(calibrated, fp)

        return {"test_accuracy": float(acc), "cv_accuracy": float(cv_scores.mean())}
