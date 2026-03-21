from __future__ import annotations

import math


class DecisionEngine:
    def decide(self, p1: float, px: float, p2: float) -> tuple[str, float]:
        entropy = -sum(p * math.log(p + 1e-12) for p in (p1, px, p2))
        confidence = max(p1, px, p2)

        if p1 >= 0.55:
            return "1", confidence
        if p2 >= 0.55:
            return "2", confidence
        if abs(p1 - p2) < 0.10 and px < 0.28:
            return "12", confidence
        if px >= 0.30:
            return ("1X" if p1 >= p2 else "X2"), confidence
        if confidence < 0.45 or entropy > 1.0:
            return ("1X" if p1 >= p2 else "X2"), confidence
        return ("1" if p1 >= p2 else "2"), confidence
