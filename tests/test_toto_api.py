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


def test_get_draw_normalizes_matches_pool_probs_and_saves_raw_payload(tmp_path: Path) -> None:
    api = TotoAPI(base_url="https://example.test", data_dir=tmp_path)
    fake = _FakeHttpClient(
        {
            "drawing": {
                "matches": [
                    {
                        "name": "Team A - Team B",
                        "result": "1",
                        "quotes": {"pool_win_1": 51.2, "pool_draw": 27.3, "pool_win_2": 21.5},
                    },
                    {
                        "home": "Team C",
                        "away": "Team D",
                        "outcome": "X",
                        "quotes": {"pool_win_1": 40, "pool_draw": 35, "pool_win_2": 25},
                    },
                ],
                "payouts": {"13": "500.4", "14": 5000, "15": "100000", "bad": "x"},
            }
        }
    )
    api.http = fake

    draw = api.get_draw(12345)

    assert draw == {
        "draw_id": 12345,
        "matches": [
            {
                "name": "Team A - Team B",
                "result": "1",
                "pool_probs": {"P1": 0.512, "PX": 0.273, "P2": 0.215},
            },
            {
                "name": "Team C - Team D",
                "result": "X",
                "pool_probs": {"P1": 0.4, "PX": 0.35, "P2": 0.25},
            },
        ],
        "payouts": {13: 500, 14: 5000, 15: 100000},
    }
    assert fake.calls == [("https://example.test/api/v1/community/drawing-info/12345", None)]

    raw_path = tmp_path / "12345.json"
    assert raw_path.exists()
    saved = json.loads(raw_path.read_text(encoding="utf-8"))
    assert "matches" in saved


def test_get_draws_returns_brief_draw_list(tmp_path: Path) -> None:
    api = TotoAPI(base_url="https://example.test", data_dir=tmp_path)
    fake = _FakeHttpClient(
        {
            "drawings": [
                {"id": 10, "number": "5123", "ended_at": "2026-03-20T20:00:00Z", "ignored": 1},
                {"id": "11", "number": "5124", "ended_at": "2026-03-21T20:00:00Z"},
            ]
        }
    )
    api.http = fake

    draws = api.get_draws(name="sportprognosis", page=3)

    assert draws == [
        {"id": 10, "number": "5123", "ended_at": "2026-03-20T20:00:00Z"},
        {"id": 11, "number": "5124", "ended_at": "2026-03-21T20:00:00Z"},
    ]
    assert fake.calls == [
        (
            "https://example.test/api/v1/community/sportprognosis/drawings",
            {"page": 3},
        )
    ]
