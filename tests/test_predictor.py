from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from core.model.predictor import ModelPredictor


class DummyModel:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.5, 0.3, 0.2]], dtype=float), (rows, 1))


class BadSumModel:
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.9, 0.2, 0.1]], dtype=float), (rows, 1))


def _feature_dict(names: list[str]) -> dict[str, float]:
    return {name: float(idx + 1) for idx, name in enumerate(names)}


def test_predictor_predict_returns_p1_px_p2(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    schema_path = tmp_path / "feature_schema.json"
    feature_names = ["f1", "f2", "f3"]

    joblib.dump(DummyModel(), model_path)
    schema_path.write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    result = predictor.predict(_feature_dict(feature_names))

    assert set(result.keys()) == {"P1", "PX", "P2"}
    assert result["P1"] == pytest.approx(0.5)
    assert result["PX"] == pytest.approx(0.3)
    assert result["P2"] == pytest.approx(0.2)


def test_predictor_raises_when_model_missing(tmp_path: pytest.TempPathFactory) -> None:
    predictor = ModelPredictor()

    with pytest.raises(FileNotFoundError):
        predictor.load(tmp_path / "missing.pkl")


def test_predictor_raises_on_feature_mismatch(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2"]

    joblib.dump(DummyModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    with pytest.raises(ValueError, match="Feature mismatch"):
        predictor.predict({"f1": 1.0})


def test_predictor_raises_on_nan_feature(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2"]

    joblib.dump(DummyModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    with pytest.raises(ValueError, match="contains NaN"):
        predictor.predict({"f1": float("nan"), "f2": 1.0})


def test_predictor_raises_when_probability_sum_is_invalid(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2"]

    joblib.dump(BadSumModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    with pytest.raises(ValueError, match="Probability sum is invalid"):
        predictor.predict(_feature_dict(feature_names))
