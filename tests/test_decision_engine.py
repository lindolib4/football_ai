from core.decision.decision_engine import DecisionEngine


def test_decision_engine_strong_favorite_returns_single_outcome() -> None:
    result = DecisionEngine().decide(
        probs={"P1": 0.65, "PX": 0.20, "P2": 0.15},
        features={},
    )
    assert result["decision"] == "1"
    assert result["confidence"] == 0.65


def test_decision_engine_medium_uncertainty_returns_double_outcome() -> None:
    result = DecisionEngine().decide(
        probs={"P1": 0.52, "PX": 0.08, "P2": 0.40},
        features={},
    )
    assert result["decision"] == "1X"


def test_decision_engine_high_uncertainty_returns_12() -> None:
    result = DecisionEngine().decide(
        probs={"P1": 0.34, "PX": 0.33, "P2": 0.33},
        features={},
    )
    assert result["decision"] == "12"


def test_decision_engine_draw_zone_includes_x() -> None:
    result = DecisionEngine().decide(
        probs={"P1": 0.48, "PX": 0.44, "P2": 0.08},
        features={},
    )
    assert result["decision"] == "1X"


def test_decision_engine_invalid_probabilities_raise_error() -> None:
    engine = DecisionEngine()

    try:
        engine.decide(
            probs={"P1": 0.5, "PX": 0.4, "P2": float("nan")},
            features={},
        )
        assert False, "Expected ValueError for NaN probabilities"
    except ValueError as exc:
        assert "NaN" in str(exc)

    try:
        engine.decide(
            probs={"P1": 0.5, "PX": 0.4, "P2": 0.2},
            features={},
        )
        assert False, "Expected ValueError for invalid probability sum"
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
