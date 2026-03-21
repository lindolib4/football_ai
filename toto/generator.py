from __future__ import annotations

import itertools

from core.decision.final_decision_engine import FinalDecisionEngine

ALLOWED_BETS = {"1", "X", "2", "1X", "X2", "12"}


class TotoGenerator:
    def __init__(self) -> None:
        self.final_engine = FinalDecisionEngine()

    def generate(self, matches: list, mode: str) -> list[list[str]]:
        if mode not in {"16", "32"}:
            raise ValueError("mode must be '16' or '32'")

        target_lines = 16 if mode == "16" else 32
        toggle_limit = 4 if mode == "16" else 5

        analyzed_matches = [self._analyze_match(match) for match in matches]

        toggle_indexes = self._select_toggle_indexes(
            analyzed_matches=analyzed_matches,
            limit=toggle_limit,
        )

        pools: list[list[str]] = []
        for index, analyzed in enumerate(analyzed_matches):
            if index in toggle_indexes:
                pools.append(analyzed["toggle_options"])
            else:
                pools.append([analyzed["single_bet"]])

        lines = [list(row) for row in itertools.product(*pools)]
        unique_lines = self._deduplicate(lines)
        coupons = unique_lines[:target_lines]

        for coupon in coupons:
            self._validate_coupon(coupon, expected_len=len(matches))

        return coupons

    def _analyze_match(self, match: dict) -> dict:
        probs = match["probs"]
        features = match["features"]
        odds = match["odds"]

        result = self.final_engine.decide(probs=probs, features=features, odds=odds)

        final_bet = result["final_bet"] or result["decision"]
        confidence = float(result["confidence"])
        single_bet = self._single_from_bet(final_bet=final_bet, probs=probs)
        toggle_options = self._build_toggle_options(final_bet=final_bet, probs=probs)

        is_safe = len(final_bet) == 1 and confidence > 0.55

        return {
            "single_bet": single_bet,
            "toggle_options": toggle_options,
            "is_safe": is_safe,
            "confidence": confidence,
        }

    def _build_toggle_options(self, final_bet: str, probs: dict) -> list[str]:
        if len(final_bet) == 2:
            options = [final_bet[0], final_bet[1]]
        else:
            sorted_outcomes = [
                outcome
                for outcome, _ in sorted(
                    (("1", probs["P1"]), ("X", probs["PX"]), ("2", probs["P2"])),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
            ]
            primary = final_bet
            secondary = next(outcome for outcome in sorted_outcomes if outcome != primary)
            options = [primary, secondary]
        return options

    def _single_from_bet(self, final_bet: str, probs: dict) -> str:
        if len(final_bet) == 1:
            return final_bet
        key_map = {"1": "P1", "X": "PX", "2": "P2"}
        return max(final_bet, key=lambda outcome: probs[key_map[outcome]])

    def _select_toggle_indexes(self, analyzed_matches: list[dict], limit: int) -> set[int]:
        risky_indexes = [idx for idx, item in enumerate(analyzed_matches) if not item["is_safe"]]
        if len(risky_indexes) >= limit:
            return set(risky_indexes[:limit])

        safe_indexes = [idx for idx, item in enumerate(analyzed_matches) if item["is_safe"]]
        safe_indexes = sorted(safe_indexes, key=lambda idx: analyzed_matches[idx]["confidence"])
        needed = limit - len(risky_indexes)
        return set(risky_indexes + safe_indexes[:needed])

    def _deduplicate(self, lines: list[list[str]]) -> list[list[str]]:
        seen: set[tuple[str, ...]] = set()
        unique: list[list[str]] = []
        for line in lines:
            key = tuple(line)
            if key in seen:
                continue
            seen.add(key)
            unique.append(line)
        return unique

    def _validate_coupon(self, coupon: list[str], expected_len: int) -> None:
        if len(coupon) != expected_len:
            raise ValueError("Coupon length is invalid")
        if any(not bet for bet in coupon):
            raise ValueError("Coupon contains empty values")
        if any(bet not in ALLOWED_BETS for bet in coupon):
            raise ValueError("Coupon contains unsupported value")


def generate_coupons(decisions: list[str], limit: int = 16) -> list[list[str]]:
    pools: list[list[str]] = []
    for decision in decisions:
        if len(decision) == 2:
            pools.append([decision[0], decision[1]])
        else:
            pools.append([decision])

    lines = [list(row) for row in itertools.product(*pools)]
    return lines[:limit]
