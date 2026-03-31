from __future__ import annotations

import logging
from typing import Any

from api.cache import JsonFileCache
from api.client import HttpClient
from config import settings

logger = logging.getLogger("api")


class FootyStatsClient:
    def __init__(self) -> None:
        self.http = HttpClient(timeout_sec=settings.request_timeout_sec, retries=settings.request_retries)
        self.cache = JsonFileCache(ttl_sec=settings.cache_ttl_sec)
        self.base_url = settings.footystats_base_url.rstrip("/")
        self.api_key = settings.footystats_api_key

    @staticmethod
    def _safe_params(params: dict[str, Any]) -> dict[str, Any]:
        return {key: ("***" if key.lower() == "key" else value) for key, value in params.items()}

    @staticmethod
    def _response_error_name(response: Any) -> str:
        error_type = getattr(response, "error_type", None)
        if error_type is None:
            return "unknown"
        return str(getattr(error_type, "value", error_type)).lower()

    @staticmethod
    def _response_terminal(response: Any) -> bool:
        return bool(getattr(response, "terminal", False))

    @staticmethod
    def _can_use_stale_cache(error_name: str, terminal: bool) -> bool:
        if terminal:
            return False
        return error_name in {"rate_limited", "server_error", "network_error"}

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        force_refresh: bool = False,
    ) -> tuple[dict[str, Any] | None, bool]:
        req_params = {"key": self.api_key, **(params or {})}
        safe_params = self._safe_params(req_params)

        if not self.api_key:
            logger.error(
                "endpoint=%s params=%s error=missing_api_key terminal=true",
                endpoint,
                safe_params,
            )
            return None, False

        cache_key = self.cache.make_key(endpoint=endpoint, params=req_params)

        ttl_override_sec: int | None = None
        if endpoint == "/todays-matches":
            ttl_override_sec = settings.live_cache_ttl_sec

        if not force_refresh:
            cached_payload, is_fresh = self.cache.get(cache_key, ttl_sec=ttl_override_sec)
            if cached_payload is not None and is_fresh:
                logger.info("endpoint=%s params=%s status=cache cache=hit freshness=fresh", endpoint, safe_params)
                return cached_payload, True

        url = f"{self.base_url}{endpoint}"
        response = self.http.get(url=url, params=req_params)
        if response.payload is not None:
            self.cache.set(cache_key, response.payload)
            return response.payload, False

        error_name = self._response_error_name(response)
        terminal = self._response_terminal(response)
        status_code = getattr(response, "status_code", None)

        logger.warning(
            "endpoint=%s params=%s status=%s error=%s terminal=%s cache=miss",
            endpoint,
            safe_params,
            status_code,
            error_name,
            terminal,
        )

        if not self._can_use_stale_cache(error_name=error_name, terminal=terminal):
            if error_name == "auth_failed":
                logger.error(
                    "endpoint=%s params=%s fallback=disabled reason=auth_failed",
                    endpoint,
                    safe_params,
                )
            return None, False

        stale_payload, _ = self.cache.get(cache_key, allow_stale=True, ttl_sec=ttl_override_sec)
        if stale_payload is not None:
            logger.warning(
                "endpoint=%s params=%s fallback=stale_cache reason=%s degraded_mode=true",
                endpoint,
                safe_params,
                error_name,
            )
            return stale_payload, True

        logger.warning(
            "endpoint=%s params=%s fallback=unavailable reason=%s degraded_mode=false",
            endpoint,
            safe_params,
            error_name,
        )

        return None, False

    @staticmethod
    def _extract_rows(data: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not data:
            return []
        for key in ("data", "matches", "teams", "leagues", "countries"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return []

    def get_country_list(self) -> list[dict[str, Any]]:
        data, _ = self._request("/country-list")
        return self._extract_rows(data)

    def get_league_list(self, chosen_leagues_only: bool = True, country_id: int | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"chosen_leagues_only": str(chosen_leagues_only).lower()}
        if country_id is not None:
            params["country_id"] = country_id
        data, _ = self._request("/league-list", params=params)
        return self._extract_rows(data)

    def get_todays_matches(
        self,
        date: str | None = None,
        timezone: str | None = None,
        page: int = 1,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"page": page}
        if date:
            params["date"] = date
        params["timezone"] = timezone or settings.app_timezone
        data, _ = self._request("/todays-matches", params=params, force_refresh=force_refresh)
        return self._extract_rows(data)

    def get_all_todays_matches(
        self,
        date: str | None = None,
        timezone: str | None = None,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        page = 1
        rows: list[dict[str, Any]] = []
        while True:
            chunk = self.get_todays_matches(
                date=date,
                timezone=timezone,
                page=page,
                force_refresh=force_refresh,
            )
            if not chunk:
                break
            rows.extend(chunk)
            if len(chunk) < 200:
                break
            page += 1
        return rows

    def get_league_season(self, season_id: int, max_time: int | None = None) -> dict[str, Any] | None:
        params: dict[str, Any] = {"season_id": season_id}
        if max_time is not None:
            params["max_time"] = max_time
        data, _ = self._request("/league-season", params=params)
        if not data:
            return None
        if isinstance(data.get("data"), dict):
            return data["data"]
        return data

    def get_league_matches(
        self,
        season_id: int,
        page: int = 1,
        max_per_page: int = 1000,
        max_time: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"season_id": season_id, "page": page, "max_per_page": max_per_page}
        if max_time is not None:
            params["max_time"] = max_time
        data, _ = self._request("/league-matches", params=params)
        return self._extract_rows(data)

    def get_all_league_matches(
        self,
        season_id: int,
        max_per_page: int = 1000,
        max_time: int | None = None,
    ) -> list[dict[str, Any]]:
        page = 1
        rows: list[dict[str, Any]] = []
        while True:
            chunk = self.get_league_matches(season_id=season_id, page=page, max_per_page=max_per_page, max_time=max_time)
            if not chunk:
                break
            rows.extend(chunk)
            if len(chunk) < max_per_page:
                break
            page += 1
        return rows

    def get_league_teams(
        self,
        season_id: int,
        include_stats: bool = True,
        page: int = 1,
        max_time: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "season_id": season_id,
            "page": page,
            "include": "stats" if include_stats else "",
        }
        if max_time is not None:
            params["max_time"] = max_time
        data, _ = self._request("/league-teams", params=params)
        return self._extract_rows(data)

    def get_all_league_teams(
        self,
        season_id: int,
        include_stats: bool = True,
        max_time: int | None = None,
    ) -> list[dict[str, Any]]:
        page = 1
        rows: list[dict[str, Any]] = []
        while True:
            chunk = self.get_league_teams(season_id=season_id, include_stats=include_stats, page=page, max_time=max_time)
            if not chunk:
                break
            rows.extend(chunk)
            if len(chunk) < 1000:
                break
            page += 1
        return rows
