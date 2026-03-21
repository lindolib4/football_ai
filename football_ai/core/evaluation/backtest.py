from __future__ import annotations


def accuracy(correct: int, total: int) -> float:
    return 0.0 if total == 0 else correct / total


def roi(stake: float, payout: float) -> float:
    return 0.0 if stake == 0 else (payout - stake) / stake
