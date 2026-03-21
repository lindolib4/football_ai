from __future__ import annotations

import logging
from typing import Any

from football_ai.api.cache import JsonFileCache
from football_ai.api.client import HttpClient
from football_ai.config import settings

logger = logging.getLogger(__name__)


class FootyStatsClient:
    def __init__(self) -> None:
        self.client = HttpClient(timeout_sec=settings.request_timeout_sec, retries=settings.request_retries)
        self.cache = JsonFileCache(ttl_sec=settings.cache_ttl_sec)
        self.base_url = settings.footystats_base_url.rstrip("/")
        self.api_key = settings.footystats_api_key

    def get_todays_matches(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}/today"
        data = self.client.get(url, params={"key": self.api_key})
        if not data:
            logger.warning("Using cached today matches fallback")
            cached = self.cache.get("today_matches")
            return cached if isinstance(cached, list) else []
        matches = data.get("matches", [])
        self.cache.set("today_matches", matches)
        return matches

    def get_match_details(self, match_id: str | int) -> dict[str, Any] | None:
        cache_key = f"match_{match_id}"
        url = f"{self.base_url}/match"
        data = self.client.get(url, params={"key": self.api_key, "match_id": match_id})
        if data:
            self.cache.set(cache_key, data)
            return data
        logger.warning("Using cached fallback for match_id=%s", match_id)
        return self.cache.get(cache_key)
