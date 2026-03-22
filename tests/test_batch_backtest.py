from __future__ import annotations

import json

from toto.batch_backtest import TotoBatchBacktest


class FakeAPI:
    def get_draws(self, name: str, page: int = 1) -> list[dict]:
        assert name == "sportprognosis"
        if page == 1:
            return [{"id": 101}, {"id": 202}]
        if page == 2:
            return [{"id": 202}, {"id": 303}]
        return []

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
                "payouts": {13: 350, 14: 3200, 15: 120000},
            },
            303: {
                "draw_id": 303,
                "matches": [
                    {"pool_probs": {"P1": 0.5, "PX": 0.3, "P2": 0.2}, "result": "1"},
                    {"pool_probs": {"P1": 0.3, "PX": 0.4, "P2": 0.3}, "result": "X"},
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
        self.received_payouts: list[dict | None] = []

    def evaluate(self, coupons: list, results: list, payouts: dict | None = None) -> dict:
        self.calls += 1
        self.received_payouts.append(payouts)
        if self.calls == 1:
            return {
                "max_hits": 14,
                "avg_hits": 13.0,
                "distribution": {14: 1, 13: 1},
                "ROI": 5.0,
            }
        if self.calls == 2:
            return {
                "max_hits": 15,
                "avg_hits": 14.0,
                "distribution": {15: 1, 14: 1},
                "ROI": 4.0,
            }
        return {
            "max_hits": 13,
            "avg_hits": 12.0,
            "distribution": {13: 1},
            "ROI": 3.0,
        }


def test_run_aggregates_pages_draws_and_saves_json(tmp_path) -> None:
    output_path = tmp_path / "backtest_results.json"
    fake_backtest = FakeBacktest()

    service = TotoBatchBacktest(
        api=FakeAPI(),
        optimizer=FakeOptimizer(),
        backtest=fake_backtest,
        mode="16",
        output_path=output_path,
    )

    result = service.run(draw_name="sportprognosis", pages=2)

    assert result["draws"] == 3
    assert result["total_profit"] == 18.0
    assert result["ROI"] == 3.0
    assert result["avg_hits"] == 13.0
    assert result["max_hits"] == 15
    assert result["distribution"] == {13: 2, 14: 2, 15: 1}
    assert fake_backtest.received_payouts == [None, {13: 350, 14: 3200, 15: 120000}, None]

    on_disk = json.loads(output_path.read_text(encoding="utf-8"))
    assert on_disk == {
        **result,
        "distribution": {"13": 2, "14": 2, "15": 1},
    }


def test_run_with_empty_pages_returns_zeroes(tmp_path) -> None:
    output_path = tmp_path / "backtest_results.json"
    service = TotoBatchBacktest(output_path=output_path)

    result = service.run(draw_name="sportprognosis", pages=0)

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
