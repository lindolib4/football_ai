from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from api.toto_api import TotoAPI
from core.decision.final_decision_engine import FinalDecisionEngine
from toto.backtest import TotoBacktest
from toto.optimizer import TotoOptimizer


logger = logging.getLogger("toto")


class TotoPipeline:
    """End-to-end orchestration for Toto draw processing."""

    def __init__(
        self,
        api: TotoAPI | None = None,
        optimizer: TotoOptimizer | None = None,
        backtest: TotoBacktest | None = None,
        decision_engine: FinalDecisionEngine | None = None,
        output_path: str | Path = "data/last_run.json",
    ) -> None:
        self.api = api or TotoAPI()
        self.optimizer = optimizer or TotoOptimizer()
        self.backtest = backtest or TotoBacktest()
        self.decision_engine = decision_engine or FinalDecisionEngine()
        self.output_path = Path(output_path)

    @staticmethod
    def _safe_probabilities(raw_probs: dict[str, Any] | None) -> dict[str, float]:
        raw_probs = raw_probs or {}
        p1 = float(raw_probs.get("P1", 0.0))
        px = float(raw_probs.get("PX", 0.0))
        p2 = float(raw_probs.get("P2", 0.0))

        total = p1 + px + p2
        if total <= 0:
            return {"P1": 1 / 3, "PX": 1 / 3, "P2": 1 / 3}

        return {
            "P1": p1 / total,
            "PX": px / total,
            "P2": p2 / total,
        }

    @staticmethod
    def _odds_from_match(match: dict[str, Any], probs: dict[str, float]) -> dict[str, float]:
        odds = match.get("odds")
        if isinstance(odds, dict) and {"O1", "OX", "O2"}.issubset(odds.keys()):
            return {
                "O1": float(odds["O1"]),
                "OX": float(odds["OX"]),
                "O2": float(odds["O2"]),
            }

        # Fallback to fair odds derived from probs.
        return {
            "O1": 1.0 / max(probs["P1"], 1e-9),
            "OX": 1.0 / max(probs["PX"], 1e-9),
            "O2": 1.0 / max(probs["P2"], 1e-9),
        }

    def _prepare_matches(self, draw: dict[str, Any]) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []

        for match in draw.get("matches", []):
            probs = self._safe_probabilities(match.get("pool_probs"))
            features = match.get("features") if isinstance(match.get("features"), dict) else {}
            odds = self._odds_from_match(match=match, probs=probs)

            decision_result = self.decision_engine.decide(probs=probs, features=features, odds=odds)
            decision = decision_result["final_bet"] or decision_result["decision"]

            prepared.append(
                {
                    "name": match.get("name", ""),
                    "probs": probs,
                    "features": features,
                    "decision": decision,
                }
            )

        return prepared

    @staticmethod
    def _extract_results(draw: dict[str, Any]) -> list[str]:
        results = draw.get("results")
        if isinstance(results, list):
            return [str(item) for item in results]

        extracted: list[str] = []
        for match in draw.get("matches", []):
            result = match.get("result")
            if result in {"1", "X", "2"}:
                extracted.append(str(result))
        return extracted

    def run_draw(self, draw_id: int, mode: str) -> dict[str, Any]:
        draw = self.api.get_draw(draw_id)

        matches = self._prepare_matches(draw=draw)
        coupons = self.optimizer.optimize(matches=matches, mode=mode)

        result = self.backtest.evaluate(
            coupons=coupons,
            results=self._extract_results(draw),
            payouts=draw.get("payouts"),
        )

        payload = {
            "draw_id": int(draw_id),
            "coupons": coupons,
            "result": result,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "toto_pipeline_run draw_id=%s roi=%.4f max_hits=%s",
            draw_id,
            float(result.get("ROI", 0.0)),
            result.get("max_hits", 0),
        )

        return payload
