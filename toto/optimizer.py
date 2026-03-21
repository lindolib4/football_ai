from __future__ import annotations

import itertools
import math

ALLOWED_OUTCOMES = {"1", "X", "2"}
_PROB_KEY = {"1": "P1", "X": "PX", "2": "P2"}


class TotoOptimizer:
    def optimize(self, matches: list[dict], mode: str) -> list[list[str]]:
        if mode not in {"16", "32"}:
            raise ValueError("mode must be '16' or '32'")

        if not matches:
            return []

        target_lines = 16 if mode == "16" else 32
        risk_count = 4 if mode == "16" else 5

        sorted_indexes = self._sort_indexes_by_risk(matches=matches)
        risk_indexes = sorted_indexes[: min(risk_count, len(matches))]

        pools: list[list[str]] = []
        for idx, match in enumerate(matches):
            decision = str(match.get("decision", "")).strip()
            if idx in risk_indexes:
                pools.append(self._options_from_decision(decision=decision, probs=match["probs"]))
            else:
                pools.append([self._single_from_decision(decision=decision, probs=match["probs"])])

        coupons = [list(line) for line in itertools.product(*pools)]
        coupons.sort(key=lambda coupon: self._coupon_weight(coupon=coupon, matches=matches), reverse=True)

        unique_coupons: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for coupon in coupons:
            key = tuple(coupon)
            if key in seen:
                continue
            seen.add(key)
            unique_coupons.append(coupon)
            if len(unique_coupons) >= target_lines:
                break

        return unique_coupons

    def coverage_score(self, coupons: list[list[str]], matches: list[dict]) -> float:
        if not coupons or not matches:
            return 0.0

        total = 0.0
        for idx, match in enumerate(matches):
            outcomes = {coupon[idx] for coupon in coupons}
            covered = sum(match["probs"][_PROB_KEY[outcome]] for outcome in outcomes)
            total += covered
        return total / len(matches)

    def _sort_indexes_by_risk(self, matches: list[dict]) -> list[int]:
        def risk(index: int) -> float:
            probs = matches[index]["probs"]
            entropy = 0.0
            for value in (probs["P1"], probs["PX"], probs["P2"]):
                if value > 0:
                    entropy -= value * math.log(value, 2)
            return entropy

        return sorted(range(len(matches)), key=risk, reverse=True)

    def _options_from_decision(self, decision: str, probs: dict) -> list[str]:
        candidates = [outcome for outcome in decision if outcome in ALLOWED_OUTCOMES]
        if not candidates:
            return [self._single_from_decision(decision=decision, probs=probs)]

        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) == 1:
            return unique_candidates

        return sorted(unique_candidates, key=lambda outcome: probs[_PROB_KEY[outcome]], reverse=True)

    def _single_from_decision(self, decision: str, probs: dict) -> str:
        candidates = [outcome for outcome in decision if outcome in ALLOWED_OUTCOMES]
        if not candidates:
            return max(("1", "X", "2"), key=lambda outcome: probs[_PROB_KEY[outcome]])

        return max(candidates, key=lambda outcome: probs[_PROB_KEY[outcome]])

    def _coupon_weight(self, coupon: list[str], matches: list[dict]) -> float:
        weight = 1.0
        for idx, outcome in enumerate(coupon):
            weight *= matches[idx]["probs"][_PROB_KEY[outcome]]
        return weight
