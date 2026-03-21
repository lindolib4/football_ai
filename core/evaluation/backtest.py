from __future__ import annotations

from collections.abc import Sequence


class BacktestEngine:
    """Minimal backtesting engine for quick quality checks."""

    @staticmethod
    def accuracy(correct: int, total: int) -> float:
        return 0.0 if total == 0 else float(correct / total)

    @staticmethod
    def roi(stake: float, payout: float) -> float:
        return 0.0 if stake == 0 else float((payout - stake) / stake)

    def evaluate(self, predictions: Sequence[str], results: Sequence[str]) -> dict[str, float]:
        """Evaluate prediction quality against final outcomes."""
        total = min(len(predictions), len(results))
        if total == 0:
            return {"total": 0.0, "correct": 0.0, "accuracy": 0.0}

        correct = sum(1 for pred, res in zip(predictions[:total], results[:total]) if pred == res)
        return {
            "total": float(total),
            "correct": float(correct),
            "accuracy": self.accuracy(correct, total),
        }
