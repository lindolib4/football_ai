from __future__ import annotations

import json

from toto.batch_backtest import TotoBatchBacktest


class FakeAPI:
    def get_draw(self, draw_id: int) -> dict:
        draws = {
            101: {
                "draw_id": 101,
                "matches": [
                    {"pool_probs": {"P1": 0.60, "PX": 0.25, "P2": 0.15}, "result": "1"},
                    {"pool_probs": {"P1": 0.34, "PX": 0.33, "P2": 0.33}, "result": "X"},
                ],
            },
            202: {
                "draw_id": 202,
                "matches": [
                    {"pool_probs": {"P1": 0.35, "PX": 0.36, "P2": 0.29}, "result": "X"},
                    {"pool_probs": {"P1": 0.20, "PX": 0.25, "P2": 0.55}, "result": "2"},
                ],
            },
        }
        return draws[draw_id]


class FakeOptimizer:
    def optimize(self, matches: list[dict], mode: str) -> list[list[str]]:
        assert mode == "16"
        assert all("probs" in m and "decision" in m for m in matches)
        return [["1", "X"], ["X", "2"]]


class FakeBacktest:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, coupons: list, results: list, payouts: dict | None = None) -> dict:
        self.calls += 1
        if self.calls == 1:
            return {
                "max_hits": 14,
                "avg_hits": 13.0,
                "distribution": {14: 1, 13: 1},
                "ROI": 5.0,
            }
        return {
            "max_hits": 15,
            "avg_hits": 14.0,
            "distribution": {15: 1, 14: 1},
            "ROI": 4.0,
        }


def test_run_aggregates_draws_and_saves_json(tmp_path) -> None:
    output_path = tmp_path / "backtest_results.json"

    service = TotoBatchBacktest(
        api=FakeAPI(),
        optimizer=FakeOptimizer(),
        backtest=FakeBacktest(),
        mode="16",
        output_path=output_path,
    )

    result = service.run(draw_ids=[101, 202])

    assert result["draws"] == 2
    assert result["total_profit"] == 14.0
    assert result["ROI"] == 7.0
    assert result["avg_hits"] == 13.5
    assert result["max_hits"] == 15
    assert result["distribution"] == {13: 1, 14: 2, 15: 1}

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == {
        **result,
        "distribution": {"13": 1, "14": 2, "15": 1},
    }


def test_run_with_empty_draws_returns_zeroes(tmp_path) -> None:
    output_path = tmp_path / "backtest_results.json"
    service = TotoBatchBacktest(output_path=output_path)

    result = service.run(draw_ids=[])

    assert result == {
        "draws": 0,
        "total_profit": 0.0,
        "ROI": 0.0,
        "avg_hits": 0.0,
        "max_hits": 0,
        "distribution": {13: 0, 14: 0, 15: 0},
    }
    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == {
        "draws": 0,
        "total_profit": 0.0,
        "ROI": 0.0,
        "avg_hits": 0.0,
        "max_hits": 0,
        "distribution": {"13": 0, "14": 0, "15": 0},
    }
