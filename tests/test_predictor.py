from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from core.decision.decision_engine import DecisionEngine
from core.model.feature_schema import FeatureSchema
from core.model.predictor import ModelPredictor


class DummyModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.5, 0.3, 0.2]], dtype=float), (rows, 1))


class CalibratedDummyModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.4, 0.35, 0.25]], dtype=float), (rows, 1))


class BadSumModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.9, 0.2, 0.1]], dtype=float), (rows, 1))


class BadRangeModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[1.2, -0.1, -0.1]], dtype=float), (rows, 1))


class BadNaNModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.5, np.nan, 0.5]], dtype=float), (rows, 1))


class ReorderedClassesModel:
    classes_ = np.array([2, 0, 1], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        # probs correspond to classes_ order: [class 2, class 0, class 1]
        return np.tile(np.array([[0.7, 0.2, 0.1]], dtype=float), (rows, 1))


class MissingClassModel:
    classes_ = np.array([0, 1, 3], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.5, 0.3, 0.2]], dtype=float), (rows, 1))


class DrawHeavyModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        # class 0(draw) is highest; class-aware mapping must produce PX as max.
        return np.tile(np.array([[0.62, 0.20, 0.18]], dtype=float), (rows, 1))


class CloseNonDrawModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        # PX is close to top but not highest.
        return np.tile(np.array([[0.31, 0.35, 0.34]], dtype=float), (rows, 1))


class StrongFavoriteModel:
    classes_ = np.array([0, 1, 2], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        rows = x.shape[0]
        return np.tile(np.array([[0.22, 0.58, 0.20]], dtype=float), (rows, 1))


def _feature_dict(names: list[str]) -> dict[str, float]:
    return {name: float(idx + 1) for idx, name in enumerate(names)}


def test_feature_schema_save_load_and_validate(tmp_path: pytest.TempPathFactory) -> None:
    schema = FeatureSchema()
    schema_path = tmp_path / "feature_schema.json"
    feature_names = ["f1", "f2", "f3"]

    schema.save(feature_names, str(schema_path))
    loaded = schema.load(str(schema_path))

    assert loaded == feature_names
    schema.validate({"f1": 1.0, "f2": 2.0, "f3": 3.0}, loaded)

    with pytest.raises(ValueError, match="Feature schema mismatch"):
        schema.validate({"f1": 1.0, "f2": 2.0}, loaded)


def test_predictor_uses_calibrated_model_when_available(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    calibrated_path = tmp_path / "calibrated_model.pkl"
    schema_path = tmp_path / "feature_schema.json"
    feature_names = ["f1", "f2", "f3"]

    joblib.dump(DummyModel(), model_path)
    joblib.dump(CalibratedDummyModel(), calibrated_path)
    schema_path.write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(tmp_path)

    result = predictor.predict(_feature_dict(feature_names))

    assert predictor.using_calibrated_model is True
    assert result == {"P1": pytest.approx(0.35), "PX": pytest.approx(0.4), "P2": pytest.approx(0.25)}


def test_predictor_falls_back_to_raw_model(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2", "f3"]

    joblib.dump(DummyModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(tmp_path)

    result = predictor.predict(_feature_dict(feature_names))

    assert predictor.using_calibrated_model is False
    assert result == {"P1": pytest.approx(0.3), "PX": pytest.approx(0.5), "P2": pytest.approx(0.2)}


def test_predictor_maps_probabilities_by_model_classes_order(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2", "f3"]

    joblib.dump(ReorderedClassesModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(tmp_path)

    result = predictor.predict(_feature_dict(feature_names))

    # classes_ = [2, 0, 1] with proba [0.7, 0.2, 0.1]
    # class 1 -> P1, class 0 -> PX, class 2 -> P2
    assert result == {"P1": pytest.approx(0.1), "PX": pytest.approx(0.2), "P2": pytest.approx(0.7)}


def test_predictor_raises_when_required_class_is_missing(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2", "f3"]

    joblib.dump(MissingClassModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(tmp_path)

    with pytest.raises(ValueError, match="missing required classes"):
        predictor.predict(_feature_dict(feature_names))


def test_decision_engine_uses_corrected_mapping_from_predictor(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2", "f3"]

    joblib.dump(DrawHeavyModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(tmp_path)
    probs = predictor.predict(_feature_dict(feature_names))
    assert probs == {"P1": pytest.approx(0.20), "PX": pytest.approx(0.62), "P2": pytest.approx(0.18)}

    result = DecisionEngine().decide(probs=probs, features={})
    assert result["decision"] == "X"


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

    with pytest.raises(ValueError, match="Feature schema mismatch"):
        predictor.predict({"f1": 1.0})


def test_predictor_raises_on_nan_feature(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2"]

    joblib.dump(DummyModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    with pytest.raises(ValueError, match="NaN or infinite"):
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


def test_predictor_raises_when_probability_value_out_of_range(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2"]

    joblib.dump(BadRangeModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        predictor.predict(_feature_dict(feature_names))


def test_predictor_raises_when_probability_contains_nan(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = ["f1", "f2"]

    joblib.dump(BadNaNModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    with pytest.raises(ValueError, match="contains NaN"):
        predictor.predict(_feature_dict(feature_names))


def test_predict_with_diagnostics_adds_calibrated_confidence_and_signal_quality(
    tmp_path: pytest.TempPathFactory,
) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = [
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "ppg_diff",
        "split_advantage",
        "draw_pct",
    ]

    joblib.dump(DummyModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    result = predictor.predict_with_diagnostics(
        {
            "ppg_diff": 0.8,
            "split_advantage": 0.6,
            "draw_pct": 0.29,
        },
        allow_no_odds_fallback=True,
        min_non_odds_features=3,
    )

    assert result["status"] == "predicted"
    assert result["no_odds_mode"] is True
    quality = result.get("prediction_quality")
    assert isinstance(quality, dict)
    assert quality["signal_strength"] == "weak_signal"
    assert quality["calibrated_confidence"] <= quality["raw_confidence"]
    assert quality["context_level"] == "degraded_context"


def test_predictor_applies_narrow_draw_promotion_in_draw_like_close_case(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = [
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "draw_pct",
        "entropy",
        "gap",
        "ppg_diff",
        "xg_diff",
        "goals_diff",
    ]

    joblib.dump(CloseNonDrawModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    result = predictor.predict(
        {
            "implied_prob_1": 0.35,
            "implied_prob_x": 0.30,
            "implied_prob_2": 0.35,
            "draw_pct": 0.31,
            "entropy": 0.95,
            "gap": 0.06,
            "ppg_diff": 0.08,
            "xg_diff": 0.10,
            "goals_diff": 0.12,
        }
    )

    assert result["PX"] > 0.31
    assert result["PX"] >= (result["P2"] - 1e-12)


def test_predictor_does_not_promote_draw_for_clear_favorite(tmp_path: pytest.TempPathFactory) -> None:
    model_path = tmp_path / "model.pkl"
    feature_names = [
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "draw_pct",
        "entropy",
        "gap",
        "ppg_diff",
        "xg_diff",
        "goals_diff",
    ]

    joblib.dump(StrongFavoriteModel(), model_path)
    (tmp_path / "feature_schema.json").write_text(json.dumps(feature_names), encoding="utf-8")

    predictor = ModelPredictor()
    predictor.load(model_path)

    result = predictor.predict(
        {
            "implied_prob_1": 0.60,
            "implied_prob_x": 0.19,
            "implied_prob_2": 0.21,
            "draw_pct": 0.33,
            "entropy": 0.90,
            "gap": 0.09,
            "ppg_diff": 0.05,
            "xg_diff": 0.08,
            "goals_diff": 0.06,
        }
    )

    assert result["P1"] == pytest.approx(0.58)
    assert result["PX"] == pytest.approx(0.22)
    assert result["P2"] == pytest.approx(0.20)
