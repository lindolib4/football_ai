from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class Predictor:
    def __init__(self, model_path: str = "data/models/model.pkl") -> None:
        self.model_path = Path(model_path)

    def predict_proba(self, feature_row: dict[str, float]) -> dict[str, float]:
        with self.model_path.open("rb") as fp:
            model = pickle.load(fp)
        row = np.array([list(feature_row.values())], dtype=float)
        probs = model.predict_proba(row)[0]
        return {"P1": float(probs[0]), "PX": float(probs[1]), "P2": float(probs[2])}
