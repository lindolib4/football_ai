from __future__ import annotations

import logging
from collections.abc import Sequence

from core.decision.final_decision_engine import FinalDecisionEngine


LOGGER = logging.getLogger(__name__)


class BacktestEngine:
    """Evaluate FinalDecisionEngine profitability on historical matches."""

    def __init__(self) -> None:
        self.final_decision_engine = FinalDecisionEngine()

    @staticmethod
    def _is_win(bet: str, result: str) -> bool:
        if bet in {"1", "X", "2"}:
            return bet == result
        if bet == "1X":
            return result in {"1", "X"}
        if bet == "X2":
            return result in {"X", "2"}
        if bet == "12":
            return result in {"1", "2"}
        raise ValueError(f"Unsupported bet type: {bet}")

    @staticmethod
    def _odds_for_bet(odds: dict, bet: str) -> float:
        key_variants = (
            f"O{bet}",
            bet,
        )
        for key in key_variants:
            if key in odds:
                return float(odds[key])
        raise ValueError(f"Missing odds for bet '{bet}'")

    def evaluate(self, matches: Sequence[dict] | Sequence[str], results: Sequence[str]) -> dict[str, float]:
        if matches and isinstance(matches[0], str):
            total = min(len(matches), len(results))
            correct = sum(
                1
                for pred, res in zip(matches[:total], results[:total])
                if pred == res
            )
            return {
                "total": float(total),
                "correct": float(correct),
                "accuracy": float(correct / total) if total else 0.0,
            }

        total_bets = 0
        wins = 0
        losses = 0
        total_profit = 0.0

        for match, result in zip(matches, results):
            final = self.final_decision_engine.decide(
                probs=match["probs"],
                features=match["features"],
                odds=match["odds"],
            )
            final_bet = final["final_bet"]

            if final_bet is None:
                continue

            total_bets += 1
            bet_odds = self._odds_for_bet(match["odds"], final_bet)

            if self._is_win(final_bet, result):
                wins += 1
                total_profit += bet_odds - 1.0
            else:
                losses += 1
                total_profit -= 1.0

        accuracy = float(wins / total_bets) if total_bets else 0.0
        roi = float(total_profit / total_bets) if total_bets else 0.0

        LOGGER.info("Backtest bets=%s wins=%s roi=%.4f", total_bets, wins, roi)

        return {
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "accuracy": accuracy,
            "ROI": roi,
        }
