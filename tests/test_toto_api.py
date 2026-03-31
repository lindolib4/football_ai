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


def test_get_draw_unwraps_top_level_data_payload(tmp_path: Path) -> None:
    api = TotoAPI(base_url="https://example.test", data_dir=tmp_path)
    fake = _FakeHttpClient(
        {
            "data": {
                "id": 6001,
                "number": "6001",
                "status": "finished",
                "events": [
                    {
                        "name": "Team E - Team F",
                        "order": 0,
                        "result": "X",
                        "quotes": {"pool_win_1": 41, "pool_draw": 33, "pool_win_2": 26},
                        "bookmaker_quotes": {"bk_win_1": 39, "bk_draw": 31, "bk_win_2": 30},
                    }
                ],
            }
        }
    )
    api.http = fake

    draw = api.get_draw(6001)

    assert draw.get("draw_id") == 6001
    assert draw.get("status") == "finished"
    matches = draw.get("matches", [])
    assert isinstance(matches, list)
    assert len(matches) == 1
    row = matches[0]
    assert isinstance(row, dict)
    assert row.get("result") == "X"
    assert isinstance(row.get("pool_probs"), dict)


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


def test_get_cached_draw_history_reads_db_snapshot(tmp_path: Path) -> None:
    api = TotoAPI(base_url="https://example.test", data_dir=tmp_path)
    payload = {
        "drawing_name": "baltbet-main",
        "provider": "totobrief",
        "source": "draw_history_api",
        "history_state": "full_history_ready",
        "history_empty_reason": None,
        "history_stats_ready": True,
        "history_draws_loaded_count": 1,
        "history_events_loaded_count": 1,
        "pages_loaded": 1,
        "draws": [{"id": 9001, "number": "X1", "ended_at": "2026-03-20T20:00:00Z"}],
        "events": [
            {
                "drawing_id": 9001,
                "draw_number": "X1",
                "order": 1,
                "name": "A vs B",
                "result": "1",
                "score": "1:0",
                "pool_win_1": 52.0,
                "pool_draw": 27.0,
                "pool_win_2": 21.0,
                "bk_win_1": 49.0,
                "bk_draw": 28.0,
                "bk_win_2": 23.0,
            }
        ],
        "stats": {"events_count": 1.0},
    }

    assert api._save_history_snapshot_to_db(name="baltbet-main", payload=payload) is True

    cached = api.get_cached_draw_history(name="baltbet-main")
    assert isinstance(cached, dict)
    assert int(cached.get("history_events_loaded_count", 0) or 0) == 1
    assert int(cached.get("history_draws_loaded_count", 0) or 0) == 1


def test_get_draw_history_merges_with_cached_when_live_events_empty(tmp_path: Path) -> None:
    api = TotoAPI(base_url="https://example.test", data_dir=tmp_path)

    cached_payload = {
        "drawing_name": "baltbet-main",
        "provider": "totobrief",
        "source": "draw_history_api",
        "history_state": "full_history_ready",
        "history_empty_reason": None,
        "history_stats_ready": True,
        "history_draws_loaded_count": 1,
        "history_events_loaded_count": 1,
        "pages_loaded": 1,
        "draws": [{"id": 500, "number": "500", "ended_at": "2026-03-20T20:00:00Z"}],
        "events": [
            {
                "drawing_id": 500,
                "draw_number": "500",
                "order": 1,
                "name": "Old A vs Old B",
                "result": "X",
                "score": "1:1",
                "pool_win_1": 45.0,
                "pool_draw": 32.0,
                "pool_win_2": 23.0,
                "bk_win_1": 48.0,
                "bk_draw": 29.0,
                "bk_win_2": 23.0,
            }
        ],
        "stats": {"events_count": 1.0},
    }
    assert api._save_history_snapshot_to_db(name="baltbet-main", payload=cached_payload) is True

    api.get_draws = lambda name, page=1: [{"id": 501, "number": "501", "ended_at": "2026-03-21T20:00:00Z"}] if page == 1 else []  # type: ignore[method-assign]
    api.get_draw = lambda draw_id: {"draw_id": int(draw_id), "matches": []}  # type: ignore[method-assign]

    merged = api.get_draw_history(
        name="baltbet-main",
        start_page=1,
        max_pages=1,
        max_draws=5,
        include_draw_payload=False,
        expand_events=True,
        persist_snapshot=True,
    )

    assert int(merged.get("history_draws_loaded_count", 0) or 0) >= 1
    assert int(merged.get("history_events_loaded_count", 0) or 0) >= 1
    assert bool(merged.get("history_db_saved", False)) is True
