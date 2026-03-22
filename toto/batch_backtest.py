from __future__ import annotations

import json
from pathlib import Path

from api.toto_api import TotoAPI
from toto.backtest import TotoBacktest
from toto.optimizer import TotoOptimizer


class TotoBatchBacktest:
    """Run Toto optimization + backtest for multiple draws from API pages."""

    def __init__(
        self,
        api: TotoAPI | None = None,
        optimizer: TotoOptimizer | None = None,
        backtest: TotoBacktest | None = None,
        mode: str = "16",
        output_path: str | Path = "data/backtest_results.json",
    ) -> None:
        self.api = api or TotoAPI()
        self.optimizer = optimizer or TotoOptimizer()
        self.backtest = backtest or TotoBacktest()
        self.mode = mode
        self.output_path = Path(output_path)

    def run(self, draw_name: str, pages: int) -> dict:
        draw_ids = self._collect_draw_ids(draw_name=draw_name, pages=pages)
        if not draw_ids:
            summary = {
                "draws": 0,
                "total_profit": 0.0,
                "ROI": 0.0,
                "avg_hits": 0.0,
                "max_hits": 0,
                "distribution": {13: 0, 14: 0, 15: 0},
            }
            self._save(summary)
            return summary

        total_profit = 0.0
        total_stake = 0.0
        max_hits = 0
        avg_hits_acc = 0.0
        total_draws = 0
        distribution = {13: 0, 14: 0, 15: 0}

        for draw_id in draw_ids:
            draw = self.api.get_draw(int(draw_id))
            raw_matches = draw.get("matches", [])
            matches = [self._to_optimizer_match(match) for match in raw_matches]
            results = [str(match.get("result", "")) for match in raw_matches]
            payouts = draw.get("payouts")

            if not matches or not results:
                continue

            coupons = self.optimizer.optimize(matches=matches, mode=self.mode)
            report = self.backtest.evaluate(coupons=coupons, results=results, payouts=payouts)

            stake = float(len(coupons))
            draw_profit = float(report["ROI"]) * stake - stake

            total_profit += draw_profit
            total_stake += stake
            max_hits = max(max_hits, int(report["max_hits"]))
            avg_hits_acc += float(report["avg_hits"])
            total_draws += 1

            draw_distribution = report.get("distribution", {})
            for hits in (13, 14, 15):
                distribution[hits] += int(draw_distribution.get(hits, 0))

        roi = float(total_profit / total_stake) if total_stake else 0.0
        avg_hits = float(avg_hits_acc / total_draws) if total_draws else 0.0

        summary = {
            "draws": total_draws,
            "total_profit": total_profit,
            "ROI": roi,
            "avg_hits": avg_hits,
            "max_hits": max_hits,
            "distribution": distribution,
        }
        self._save(summary)
        return summary

    def _collect_draw_ids(self, draw_name: str, pages: int) -> list[int]:
        unique_ids: list[int] = []
        seen: set[int] = set()

        safe_pages = max(0, int(pages))
        for page in range(1, safe_pages + 1):
            draws = self.api.get_draws(name=draw_name, page=page)
            for draw in draws:
                draw_id = int(draw.get("id", 0))
                if draw_id <= 0 or draw_id in seen:
                    continue
                seen.add(draw_id)
                unique_ids.append(draw_id)

        return unique_ids

    def _to_optimizer_match(self, match: dict) -> dict:
        probs = match.get("pool_probs") or {}
        p1 = float(probs.get("P1", 0.0))
        px = float(probs.get("PX", 0.0))
        p2 = float(probs.get("P2", 0.0))

        ordered = sorted((("1", p1), ("X", px), ("2", p2)), key=lambda item: item[1], reverse=True)
        top, second = ordered[0], ordered[1]

        if top[1] - second[1] <= 0.08:
            decision = f"{top[0]}{second[0]}"
        else:
            decision = top[0]

        return {
            "probs": {"P1": p1, "PX": px, "P2": p2},
            "decision": decision,
        }

    def _save(self, payload: dict) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
