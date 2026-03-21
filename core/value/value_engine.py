from __future__ import annotations

import logging
import math


LOGGER = logging.getLogger(__name__)


class ValueEngine:
    def calculate(self, probs: dict, odds: dict) -> dict:
        required_prob_keys = ("P1", "PX", "P2")
        required_odds_keys = ("O1", "OX", "O2")

        missing_prob_keys = [key for key in required_prob_keys if key not in probs]
        if missing_prob_keys:
            raise ValueError(f"Missing probability keys: {missing_prob_keys}")

        missing_odds_keys = [key for key in required_odds_keys if key not in odds]
        if missing_odds_keys:
            raise ValueError(f"Missing odds keys: {missing_odds_keys}")

        p1 = float(probs["P1"])
        px = float(probs["PX"])
        p2 = float(probs["P2"])

        o1 = float(odds["O1"])
        ox = float(odds["OX"])
        o2 = float(odds["O2"])

        probabilities = (p1, px, p2)
        odds_values = (o1, ox, o2)

        if any(math.isnan(value) or math.isinf(value) for value in probabilities):
            raise ValueError("Probabilities must contain finite values")

        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("Probabilities must be in [0, 1]")

        total_probability = sum(probabilities)
        if not math.isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to 1.0")

        if any(math.isnan(value) or math.isinf(value) for value in odds_values):
            raise ValueError("Odds must contain finite values")

        if any(value <= 1.0 for value in odds_values):
            raise ValueError("Odds must be greater than 1")

        ev1 = p1 * o1 - 1.0
        evx = px * ox - 1.0
        ev2 = p2 * o2 - 1.0

        ev_map = {"1": ev1, "X": evx, "2": ev2}
        best_bet = max(ev_map, key=ev_map.get)
        value_flag = ev_map[best_bet] > 0.05

        LOGGER.info(
            "ValueEngine probs=%s odds=%s ev=%s best_bet=%s value_flag=%s",
            {"P1": p1, "PX": px, "P2": p2},
            {"O1": o1, "OX": ox, "O2": o2},
            {"EV1": ev1, "EVX": evx, "EV2": ev2},
            best_bet,
            value_flag,
        )

        return {
            "EV1": ev1,
            "EVX": evx,
            "EV2": ev2,
            "best_bet": best_bet,
            "value_flag": value_flag,
        }
