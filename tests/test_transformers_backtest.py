from core.evaluation.backtest import BacktestEngine
from core.features.transformers import handle_missing, normalize_features


def test_transformers_handle_missing_and_normalize() -> None:
    data = {"a": 1, "b": None, "c": 3}
    cleaned = handle_missing(data)
    assert cleaned["b"] == 0.0

    scaled = normalize_features(cleaned)
    assert 0.0 <= scaled["a"] <= 1.0
    assert 0.0 <= scaled["c"] <= 1.0


def test_backtest_engine_evaluate() -> None:
    engine = BacktestEngine()
    metrics = engine.evaluate(["1", "X", "2"], ["1", "2", "2"])
    assert metrics["total"] == 3.0
    assert metrics["correct"] == 2.0
