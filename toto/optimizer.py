from __future__ import annotations


def score_coupon(coupon: list[str]) -> float:
    double_count = sum(1 for item in coupon if len(item) == 2)
    return double_count / max(1, len(coupon))
