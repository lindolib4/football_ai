from core.decision.final_decision_engine import FinalDecisionEngine


def test_final_decision_engine_match_between_decision_and_value() -> None:
    result = FinalDecisionEngine().decide(
        probs={"P1": 0.65, "PX": 0.20, "P2": 0.15},
        features={},
        odds={"O1": 2.0, "OX": 3.0, "O2": 4.0},
    )

    assert result["decision"] == "1"
    assert result["value_bet"] == "1"
    assert result["final_bet"] == "1"
    assert result["value_flag"] is True


def test_final_decision_engine_conflict_uses_ev_threshold() -> None:
    engine = FinalDecisionEngine()

    low_ev_result = engine.decide(
        probs={"P1": 0.62, "PX": 0.18, "P2": 0.20},
        features={},
        odds={"O1": 1.5, "OX": 2.0, "O2": 5.45},
    )
    assert low_ev_result["decision"] == "1"
    assert low_ev_result["value_bet"] == "2"
    assert low_ev_result["EV"] < 0.10
    assert low_ev_result["final_bet"] == "1"

    high_ev_result = engine.decide(
        probs={"P1": 0.62, "PX": 0.18, "P2": 0.20},
        features={},
        odds={"O1": 1.5, "OX": 2.0, "O2": 5.7},
    )
    assert high_ev_result["decision"] == "1"
    assert high_ev_result["value_bet"] == "2"
    assert high_ev_result["EV"] >= 0.10
    assert high_ev_result["final_bet"] == "2"


def test_final_decision_engine_returns_none_when_no_value() -> None:
    result = FinalDecisionEngine().decide(
        probs={"P1": 0.50, "PX": 0.25, "P2": 0.25},
        features={},
        odds={"O1": 2.0, "OX": 2.8, "O2": 3.0},
    )

    assert result["value_flag"] is False
    assert result["final_bet"] is None
