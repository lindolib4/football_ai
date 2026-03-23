from __future__ import annotations

import json

from toto.pipeline import TotoPipeline


class _FakeAPI:
    def get_draw(self, draw_id: int) -> dict:
        assert draw_id == 101
        return {
            "draw_id": draw_id,
            "matches": [
                {
                    "name": "A-B",
                    "pool_probs": {"P1": 0.5, "PX": 0.3, "P2": 0.2},
                    "result": "1",
                },
                {
                    "name": "C-D",
                    "pool_probs": {"P1": 0.2, "PX": 0.3, "P2": 0.5},
                    "result": "2",
                },
            ],
            "payouts": {2: 120},
        }


class _FakeOptimizer:
    def optimize(self, matches: list[dict], mode: str) -> list[list[str]]:
        assert mode == "16"
        assert len(matches) == 2
        assert all("probs" in item for item in matches)
        return [["1", "2"], ["1", "X"]]


class _FakeBacktest:
    def evaluate(self, coupons: list, results: list, payouts: dict | None = None) -> dict:
        assert coupons == [["1", "2"], ["1", "X"]]
        assert results == ["1", "2"]
        assert payouts == {2: 120}
        return {
            "max_hits": 2,
            "avg_hits": 1.5,
            "distribution": {2: 1, 1: 1},
            "ROI": 60.0,
        }


class _FakeDecisionEngine:
    def decide(self, probs: dict, features: dict, odds: dict) -> dict:
        _ = probs, features, odds
        return {
            "decision": "1X",
            "value_bet": "1",
            "final_bet": None,
            "confidence": 0.5,
            "EV": 0.01,
            "value_flag": False,
        }


def test_pipeline_run_draw_returns_payload_and_writes_last_run(tmp_path) -> None:
    output_path = tmp_path / "last_run.json"
    pipeline = TotoPipeline(
        api=_FakeAPI(),
        optimizer=_FakeOptimizer(),
        backtest=_FakeBacktest(),
        decision_engine=_FakeDecisionEngine(),
        output_path=output_path,
    )

    response = pipeline.run_draw(draw_id=101, mode="16")

    assert response["draw_id"] == 101
    assert response["coupons"] == [["1", "2"], ["1", "X"]]
    assert response["result"]["max_hits"] == 2

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["draw_id"] == response["draw_id"]
    assert saved["coupons"] == response["coupons"]
    assert saved["result"]["max_hits"] == response["result"]["max_hits"]
    assert saved["result"]["ROI"] == response["result"]["ROI"]


def test_pipeline_extracts_results_from_matches_when_results_field_missing(tmp_path) -> None:
    class _ApiWithoutResults(_FakeAPI):
        def get_draw(self, draw_id: int) -> dict:
            draw = super().get_draw(draw_id)
            draw.pop("payouts")
            return draw

    class _BacktestAssertsExtractedResults:
        def evaluate(self, coupons: list, results: list, payouts: dict | None = None) -> dict:
            assert coupons == [["1", "2"], ["1", "X"]]
            assert results == ["1", "2"]
            assert payouts is None
            return {"max_hits": 2, "avg_hits": 2.0, "distribution": {2: 2}, "ROI": 0.0}

    pipeline = TotoPipeline(
        api=_ApiWithoutResults(),
        optimizer=_FakeOptimizer(),
        backtest=_BacktestAssertsExtractedResults(),
        decision_engine=_FakeDecisionEngine(),
        output_path=tmp_path / "last_run.json",
    )

    response = pipeline.run_draw(draw_id=101, mode="16")
    assert response["result"]["max_hits"] == 2
