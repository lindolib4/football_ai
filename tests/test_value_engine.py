import pytest

from core.value.value_engine import ValueEngine


def test_value_engine_calculates_ev_correctly() -> None:
    result = ValueEngine().calculate(
        probs={"P1": 0.50, "PX": 0.30, "P2": 0.20},
        odds={"O1": 2.20, "OX": 3.00, "O2": 4.00},
    )

    assert result["EV1"] == pytest.approx(0.10)
    assert result["EVX"] == pytest.approx(-0.10)
    assert result["EV2"] == pytest.approx(-0.20)


def test_value_engine_selects_best_bet_by_max_ev() -> None:
    result = ValueEngine().calculate(
        probs={"P1": 0.40, "PX": 0.25, "P2": 0.35},
        odds={"O1": 2.20, "OX": 4.00, "O2": 3.60},
    )

    assert result["best_bet"] == "2"


def test_value_engine_sets_value_flag_when_best_ev_above_threshold() -> None:
    positive_result = ValueEngine().calculate(
        probs={"P1": 0.55, "PX": 0.25, "P2": 0.20},
        odds={"O1": 2.00, "OX": 3.20, "O2": 4.20},
    )
    assert positive_result["value_flag"] is True

    neutral_result = ValueEngine().calculate(
        probs={"P1": 0.50, "PX": 0.25, "P2": 0.25},
        odds={"O1": 2.00, "OX": 2.80, "O2": 3.00},
    )
    assert neutral_result["value_flag"] is False


def test_value_engine_rejects_invalid_inputs() -> None:
    engine = ValueEngine()

    try:
        engine.calculate(
            probs={"P1": 0.5, "PX": 0.4, "P2": 0.2},
            odds={"O1": 2.0, "OX": 3.0, "O2": 4.0},
        )
        assert False, "Expected ValueError for invalid probability sum"
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)

    try:
        engine.calculate(
            probs={"P1": 0.5, "PX": 0.3, "P2": 0.2},
            odds={"O1": 1.0, "OX": 3.0, "O2": 4.0},
        )
        assert False, "Expected ValueError for invalid odds"
    except ValueError as exc:
        assert "greater than 1" in str(exc)
