from __future__ import annotations

from collections import Counter


class TotoBacktest:
    """Simple backtest for toto coupons against real outcomes."""

    PAYOUTS = {
        15: 100000,
        14: 5000,
        13: 500,
    }

    @staticmethod
    def _is_hit(prediction: str, result: str) -> bool:
        if prediction == "1":
            return result == "1"
        if prediction == "X":
            return result == "X"
        if prediction == "2":
            return result == "2"
        if prediction == "1X":
            return result in {"1", "X"}
        if prediction == "X2":
            return result in {"X", "2"}
        if prediction == "12":
            return result in {"1", "2"}
        raise ValueError(f"Unsupported prediction: {prediction}")

    def evaluate(self, coupons: list, results: list) -> dict:
        if not coupons:
            return {
                "max_hits": 0,
                "avg_hits": 0.0,
                "distribution": {},
                "ROI": 0.0,
            }

        hits_per_coupon: list[int] = []
        total_win = 0.0

        for coupon in coupons:
            count_correct = sum(
                1 for prediction, result in zip(coupon, results) if self._is_hit(prediction, result)
            )
            hits_per_coupon.append(count_correct)
            total_win += float(self.PAYOUTS.get(count_correct, 0))

        distribution = dict(sorted(Counter(hits_per_coupon).items(), reverse=True))
        max_hits = max(hits_per_coupon)
        avg_hits = float(sum(hits_per_coupon) / len(hits_per_coupon))

        stake = len(coupons)
        roi = float(total_win / stake) if stake else 0.0

        return {
            "max_hits": max_hits,
            "avg_hits": avg_hits,
            "distribution": distribution,
            "ROI": roi,
        }
