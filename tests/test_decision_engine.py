from core.decision.decision_engine import DecisionEngine


def test_decision_engine_home_win_threshold() -> None:
    decision, _ = DecisionEngine().decide(0.60, 0.20, 0.20)
    assert decision == "1"


def test_decision_engine_double_when_close() -> None:
    decision, _ = DecisionEngine().decide(0.36, 0.25, 0.34)
    assert decision == "12"
