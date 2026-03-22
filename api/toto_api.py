from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from api.client import HttpClient
from config import settings

logger = logging.getLogger("toto")


class TotoAPI:
    """Client for Totobrief drawings and drawing details endpoints."""

    def __init__(self, base_url: str | None = None, data_dir: str | Path = "data/toto_draws") -> None:
        self.http = HttpClient(timeout_sec=settings.request_timeout_sec, retries=settings.request_retries)
        self.base_url = (base_url or os.getenv("TOTO_API_BASE_URL", "")).rstrip("/")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _extract_pool_probs(cls, payload: dict[str, Any]) -> dict[str, float]:
        quotes = payload.get("quotes")
        if not isinstance(quotes, dict):
            quotes = {}

        return {
            "P1": cls._safe_float(quotes.get("pool_win_1")) / 100,
            "PX": cls._safe_float(quotes.get("pool_draw")) / 100,
            "P2": cls._safe_float(quotes.get("pool_win_2")) / 100,
        }

    @staticmethod
    def _extract_match_name(payload: dict[str, Any]) -> str:
        for key in ("name", "match_name", "label", "title"):
            value = payload.get(key)
            if value:
                return str(value)

        home = payload.get("home")
        away = payload.get("away")
        if home and away:
            return f"{home} - {away}"
        return ""

    @staticmethod
    def _extract_match_result(payload: dict[str, Any]) -> str:
        for key in ("result", "outcome", "final_result", "sign"):
            value = payload.get(key)
            if value in {"1", "X", "2"}:
                return str(value)
        return ""

    def _normalize_match(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": self._extract_match_name(payload),
            "result": self._extract_match_result(payload),
            "pool_probs": self._extract_pool_probs(payload),
        }

    def _save_raw_draw(self, draw_id: int, payload: dict[str, Any]) -> None:
        path = self.data_dir / f"{draw_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalize_draw(self, draw_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        matches_payload = payload.get("matches", [])
        if not isinstance(matches_payload, list):
            matches_payload = []

        draw = {
            "draw_id": int(draw_id),
            "matches": [self._normalize_match(match) for match in matches_payload if isinstance(match, dict)],
        }
        logger.info("toto_draw draw_id=%s matches=%s", draw["draw_id"], len(draw["matches"]))
        return draw

    def get_draws(self, name: str, page: int = 1) -> list[dict[str, Any]]:
        if not self.base_url:
            raise ValueError("TOTO_API_BASE_URL is not configured")

        response = self.http.get(
            url=f"{self.base_url}/api/v1/community/{name}/drawings",
            params={"page": page},
        )
        payload = response.payload or {}

        rows = payload.get("drawings", payload.get("data", payload))
        if not isinstance(rows, list):
            rows = []

        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            normalized.append(
                {
                    "id": int(row.get("id", 0)),
                    "number": row.get("number"),
                    "ended_at": row.get("ended_at"),
                }
            )
        return normalized

    def get_draw(self, draw_id: int) -> dict[str, Any]:
        if not self.base_url:
            raise ValueError("TOTO_API_BASE_URL is not configured")

        response = self.http.get(url=f"{self.base_url}/api/v1/community/drawing-info/{draw_id}")
        payload = response.payload or {}

        raw_draw = payload.get("drawing") if isinstance(payload.get("drawing"), dict) else payload
        draw = self._normalize_draw(draw_id, raw_draw)
        self._save_raw_draw(draw_id, raw_draw)
        return draw
