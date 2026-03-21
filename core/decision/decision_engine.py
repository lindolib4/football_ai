from __future__ import annotations

import logging
import math


LOGGER = logging.getLogger(__name__)


class DecisionEngine:
    def decide(self, probs: dict, features: dict) -> dict:
        _ = features

        required_keys = ("P1", "PX", "P2")
        missing_keys = [key for key in required_keys if key not in probs]
        if missing_keys:
            raise ValueError(f"Missing probability keys: {missing_keys}")

        p1 = float(probs["P1"])
        px = float(probs["PX"])
        p2 = float(probs["P2"])

        probabilities = (p1, px, p2)
        if any(math.isnan(value) for value in probabilities):
            raise ValueError("Probabilities must not contain NaN")

        total_probability = sum(probabilities)
        if not math.isclose(total_probability, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("Probabilities must sum to 1.0")

        max_entropy = math.log(3)
        entropy_raw = -sum(value * math.log(value + 1e-12) for value in probabilities)
        entropy = min(max(entropy_raw / max_entropy, 0.0), 1.0)

        gap = abs(p1 - p2)
        confidence = max(probabilities)

        max_key = max(probs, key=probs.get)

        draw_zone = px >= 0.30 and abs(px - confidence) <= 0.05

        if confidence >= 0.60 and entropy < 0.85:
            if max_key == "P1":
                decision = "1"
            elif max_key == "PX":
                decision = "X"
            else:
                decision = "2"
        elif 0.40 <= confidence <= 0.60:
            if draw_zone:
                decision = "1X" if p1 >= p2 else "X2"
            else:
                decision = "1X" if p1 > p2 else "X2"
        elif entropy >= 0.90 or gap < 0.15:
            decision = "12"
        elif draw_zone:
            decision = "1X" if p1 >= p2 else "X2"
        else:
            decision = "1X" if p1 >= p2 else "X2"

        LOGGER.info(
            "DecisionEngine probs=%s entropy=%.4f gap=%.4f decision=%s",
            {"P1": p1, "PX": px, "P2": p2},
            entropy,
            gap,
            decision,
        )

        return {
            "decision": decision,
            "confidence": confidence,
            "entropy": entropy,
            "gap": gap,
        }
