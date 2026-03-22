from __future__ import annotations

import json
from pathlib import Path

from api.client import HttpResponse
from api.toto_api import TotoAPI


class _FakeHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def get(self, url: str, params: dict | None = None) -> HttpResponse:
        self.calls.append((url, params))
        return HttpResponse(payload=self.payload, status_code=200)


def test_get_draw_normalizes_and_saves_raw_payload(tmp_path: Path) -> None:
    api = TotoAPI(base_url="https://example.test", data_dir=tmp_path)
    fake = _FakeHttpClient(
        {
            "draw": {
                "draw_id": 12345,
                "matches": [{"id": 1, "home": "A", "away": "B"}],
                "results": ["1", "X", "2"],
                "payouts": {"15": 111111, "14": 2222, "13": 333},
            }
        }
    )
    api.http = fake

    draw = api.get_draw(12345)

    assert draw == {
        "draw_id": 12345,
        "matches": [{"id": 1, "home": "A", "away": "B"}],
        "results": ["1", "X", "2"],
        "payouts": {15: 111111, 14: 2222, 13: 333},
    }
    assert fake.calls == [("https://example.test/draws/12345", None)]

    raw_path = tmp_path / "12345.json"
    assert raw_path.exists()
    saved = json.loads(raw_path.read_text(encoding="utf-8"))
    assert saved["draw_id"] == 12345


def test_get_draws_normalizes_list_and_saves_each_raw_payload(tmp_path: Path) -> None:
    api = TotoAPI(base_url="https://example.test", data_dir=tmp_path)
    fake = _FakeHttpClient(
        {
            "draws": [
                {"id": 1, "matches": [], "results": ["1"], "payouts": {"15": 10}},
                {"drawId": 2, "matches": [], "results": ["X"], "payouts": {14: 20}},
            ]
        }
    )
    api.http = fake

    draws = api.get_draws(date_from="2026-01-01", date_to="2026-01-31")

    assert draws == [
        {"draw_id": 1, "matches": [], "results": ["1"], "payouts": {15: 10}},
        {"draw_id": 2, "matches": [], "results": ["X"], "payouts": {14: 20}},
    ]
    assert fake.calls == [
        (
            "https://example.test/draws",
            {"date_from": "2026-01-01", "date_to": "2026-01-31"},
        )
    ]
    assert (tmp_path / "1.json").exists()
    assert (tmp_path / "2.json").exists()
