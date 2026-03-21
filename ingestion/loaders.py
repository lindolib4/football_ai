from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from api.footystats import FootyStatsClient
from config import settings
from database.db import Database
from ingestion.normalizers import (
    normalize_country,
    normalize_league,
    normalize_league_season_stats,
    normalize_match,
    normalize_team,
)

logger = logging.getLogger("api")


class IngestionLoader:
    def __init__(self, api: FootyStatsClient | None = None, db: Database | None = None) -> None:
        self.api = api or FootyStatsClient()
        self.db = db or Database()
        self.raw_dir = settings.raw_data_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _save_raw(self, bucket: str, name: str, payload: Any) -> None:
        target_dir = self.raw_dir / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load_countries(self) -> int:
        rows = self.api.get_country_list()
        normalized = [normalize_country(row) for row in rows]
        self._save_raw("countries", "country_list", rows)
        self.db.upsert_countries(normalized)
        logger.info("countries loaded=%s", len(normalized))
        return len(normalized)

    def load_leagues(self, chosen_leagues_only: bool = True, country_id: int | None = None) -> int:
        rows = self.api.get_league_list(chosen_leagues_only=chosen_leagues_only, country_id=country_id)
        normalized = [normalize_league(row) for row in rows]
        self._save_raw("leagues", f"league_list_country_{country_id or 'all'}", rows)
        self.db.upsert_leagues(normalized)
        logger.info("leagues loaded=%s", len(normalized))
        return len(normalized)

    def load_todays_matches(self, date: str | None = None, timezone: str | None = None) -> int:
        rows = self.api.get_all_todays_matches(date=date, timezone=timezone)
        normalized = [normalize_match(row) for row in rows]
        self._save_raw("matches", f"todays_matches_{date or 'today'}", rows)
        self.db.upsert_matches(normalized)
        logger.info("todays matches loaded=%s", len(normalized))
        return len(normalized)

    def load_league_season(self, season_id: int, max_time: int | None = None) -> bool:
        row = self.api.get_league_season(season_id=season_id, max_time=max_time)
        if not row:
            return False
        normalized = normalize_league_season_stats(row=row, season_id=season_id)
        self._save_raw("seasons", f"season_{season_id}", row)
        self.db.upsert_league_season_stats(normalized)
        logger.info("league season loaded season_id=%s", season_id)
        return True

    def load_league_teams(self, season_id: int, max_time: int | None = None) -> int:
        rows = self.api.get_all_league_teams(season_id=season_id, include_stats=True, max_time=max_time)
        normalized = [normalize_team(row=row, season_id=season_id) for row in rows]
        self._save_raw("teams", f"league_teams_{season_id}", rows)
        self.db.upsert_teams(normalized)
        logger.info("league teams loaded=%s season_id=%s", len(normalized), season_id)
        return len(normalized)

    def load_league_matches(self, season_id: int, max_time: int | None = None) -> int:
        rows = self.api.get_all_league_matches(season_id=season_id, max_per_page=1000, max_time=max_time)
        normalized = [normalize_match(row) for row in rows]
        self._save_raw("matches", f"league_matches_{season_id}", rows)
        self.db.upsert_matches(normalized)
        logger.info("league matches loaded=%s season_id=%s", len(normalized), season_id)
        return len(normalized)
