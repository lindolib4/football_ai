from __future__ import annotations


def needs_calibration(probabilities: dict[str, float], min_confidence: float = 0.45) -> bool:
    return max(probabilities.values()) < min_confidence
