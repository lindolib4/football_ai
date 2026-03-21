from __future__ import annotations

import logging

from core.decision.decision_engine import DecisionEngine
from core.value.value_engine import ValueEngine


LOGGER = logging.getLogger(__name__)


class FinalDecisionEngine:
    def __init__(self) -> None:
        self.decision_engine = DecisionEngine()
        self.value_engine = ValueEngine()

    def decide(self, probs: dict, features: dict, odds: dict) -> dict:
        decision_result = self.decision_engine.decide(probs=probs, features=features)
        value_result = self.value_engine.calculate(probs=probs, odds=odds)

        decision = decision_result["decision"]
        value_bet = value_result["best_bet"]
        value_flag = bool(value_result["value_flag"])

        ev_map = {
            "1": float(value_result["EV1"]),
            "X": float(value_result["EVX"]),
            "2": float(value_result["EV2"]),
        }
        ev = ev_map[value_bet]

        conflict = False

        if not value_flag:
            final_bet = None
        elif len(decision) == 2 and value_bet in decision:
            final_bet = value_bet
        elif value_bet == decision:
            final_bet = value_bet
        elif len(decision) == 2:
            conflict = True
            final_bet = decision
        else:
            conflict = True
            final_bet = value_bet if ev >= 0.10 else decision

        LOGGER.info(
            "FinalDecisionEngine decision=%s value=%s conflict=%s final_bet=%s",
            decision,
            value_bet,
            conflict,
            final_bet,
        )

        return {
            "decision": decision,
            "value_bet": value_bet,
            "final_bet": final_bet,
            "confidence": float(decision_result["confidence"]),
            "EV": ev,
            "value_flag": value_flag,
        }
