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
    """Client for TOTO draws API with normalization and raw payload persistence."""

    def __init__(self, base_url: str | None = None, data_dir: str | Path = "data/toto_draws") -> None:
        self.http = HttpClient(timeout_sec=settings.request_timeout_sec, retries=settings.request_retries)
        self.base_url = (base_url or os.getenv("TOTO_API_BASE_URL", "")).rstrip("/")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _extract_results(payload: dict[str, Any]) -> list[str]:
        values = payload.get("results", [])
        if not isinstance(values, list):
            return []
        return [str(v) for v in values]

    @staticmethod
    def _extract_matches(payload: dict[str, Any]) -> list[dict[str, Any]]:
        values = payload.get("matches", [])
        if not isinstance(values, list):
            return []
        return [row for row in values if isinstance(row, dict)]

    @staticmethod
    def _extract_draw_id(payload: dict[str, Any]) -> int:
        for key in ("draw_id", "id", "drawId"):
            value = payload.get(key)
            if value is not None:
                return int(value)
        raise ValueError("draw_id is missing in payload")

    @staticmethod
    def _extract_payouts(payload: dict[str, Any]) -> dict[int, int]:
        raw = payload.get("payouts", {})
        if not isinstance(raw, dict):
            return {}

        parsed: dict[int, int] = {}
        for target_hits in (15, 14, 13):
            value = raw.get(target_hits)
            if value is None:
                value = raw.get(str(target_hits))
            if value is None:
                continue
            parsed[target_hits] = int(value)
        return parsed

    def _save_raw_draw(self, draw_id: int, payload: dict[str, Any]) -> None:
        path = self.data_dir / f"{draw_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _normalize_draw(self, payload: dict[str, Any]) -> dict[str, Any]:
        draw = {
            "draw_id": self._extract_draw_id(payload),
            "matches": self._extract_matches(payload),
            "results": self._extract_results(payload),
            "payouts": self._extract_payouts(payload),
        }
        logger.info("toto_draw draw_id=%s payouts=%s", draw["draw_id"], draw["payouts"])
        return draw

    def get_draw(self, draw_id: int) -> dict[str, Any]:
        if not self.base_url:
            raise ValueError("TOTO_API_BASE_URL is not configured")

        response = self.http.get(url=f"{self.base_url}/draws/{draw_id}")
        payload = response.payload or {}
        raw_draw = payload.get("draw") if isinstance(payload.get("draw"), dict) else payload
        draw = self._normalize_draw(raw_draw)
        self._save_raw_draw(draw["draw_id"], raw_draw)
        return draw

    def get_draws(self, date_from: str, date_to: str) -> list[dict[str, Any]]:
        if not self.base_url:
            raise ValueError("TOTO_API_BASE_URL is not configured")

        response = self.http.get(
            url=f"{self.base_url}/draws",
            params={"date_from": date_from, "date_to": date_to},
        )
        payload = response.payload or {}

        if isinstance(payload.get("draws"), list):
            rows = payload["draws"]
        elif isinstance(payload.get("data"), list):
            rows = payload["data"]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []

        normalized: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            draw = self._normalize_draw(row)
            self._save_raw_draw(draw["draw_id"], row)
            normalized.append(draw)
        return normalized
