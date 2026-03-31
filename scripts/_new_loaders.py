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
    _COMPLETED_STATUSES = {
        "complete",
        "completed",
        "finished",
        "full-time",
        "full_time",
        "ft",
        "aet",
        "after extra time",
    }

    def __init__(self, api: FootyStatsClient | None = None, db: Database | None = None) -> None:
        self.api = api or FootyStatsClient()
        self.db = db or Database()
        # Make storage path operational on a clean environment without manual schema step.
        self.db.initialize()
        self.raw_dir = settings.raw_data_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def _save_raw(self, bucket: str, name: str, payload: Any) -> None:
        target_dir = self.raw_dir / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def _is_completed_status(cls, status: Any) -> bool:
        return str(status or "").strip().lower() in cls._COMPLETED_STATUSES

    @classmethod
    def _to_completed_status(cls, status: Any) -> str | None:
        if cls._is_completed_status(status):
            return "completed"
        normalized = str(status or "").strip().lower()
        return normalized or None

    @staticmethod
    def _match_has_result(match: dict[str, Any]) -> bool:
        return (
            match.get("winning_team_id") is not None
            or (match.get("home_goals") is not None and match.get("away_goals") is not None)
        )

    def _normalize_history_matches(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            norm = normalize_match(row)
            norm["status"] = self._to_completed_status(norm.get("status"))
            normalized_rows.append(norm)
        return normalized_rows

    def _extract_candidate_seasons(self) -> list[dict[str, Any]]:
        league_rows = self.api.get_league_list(chosen_leagues_only=True)
        candidates: list[dict[str, Any]] = []
        for league in league_rows:
            if not isinstance(league, dict):
                continue
            seasons = league.get("season")
            if not isinstance(seasons, list):
                continue
            for season in seasons:
                if not isinstance(season, dict):
                    continue
                season_id = season.get("id")
                year = season.get("year")
                try:
                    season_id_int = int(season_id)
                except (TypeError, ValueError):
                    continue
                try:
                    year_int = int(year)
                except (TypeError, ValueError):
                    year_int = 0
                candidates.append(
                    {
                        "season_id": season_id_int,
                        "year": year_int,
                        "league": str(league.get("name") or ""),
                        "country": str(league.get("country") or ""),
                    }
                )

        deduped: dict[int, dict[str, Any]] = {}
        for row in sorted(candidates, key=lambda item: (item["year"], item["season_id"]), reverse=True):
            sid = int(row["season_id"])
            if sid not in deduped:
                deduped[sid] = row
        return list(deduped.values())

    def load_historical_completed_matches(
        self,
        season_ids: list[int] | None = None,
        max_seasons: int = 3,
        min_year: int | None = None,
        max_time: int | None = None,
    ) -> dict[str, Any]:
        """
        Load historical league matches and write completed rows to SQLite.

        For each processed season this also calls load_league_teams() and
        load_league_season() so that team_season_stats and league_season_stats
        are populated.  Without those two calls all team-stat-based features
        (goals_diff, shots_diff, possession_diff, xg_diff, draw_pct,
        home_advantage, avg_goals) fall back to 0.0 at training time.

        Returns a summary dict suitable for UI/runtime status blocks.
        """
        target_seasons: list[dict[str, Any]] = []
        if season_ids:
            target_seasons = [{"season_id": int(sid), "year": 0, "league": "", "country": ""} for sid in season_ids]
        else:
            candidates = self._extract_candidate_seasons()
            if min_year is not None:
                candidates = [row for row in candidates if int(row.get("year", 0)) >= int(min_year)]
            target_seasons = candidates[:max(1, int(max_seasons))]

        summary: dict[str, Any] = {
            "status": "empty",
            "seasons_requested": len(target_seasons),
            "seasons_processed": 0,
            "matches_fetched": 0,
            "completed_matches_found": 0,
            "completed_matches_with_result": 0,
            "db_added_estimate": 0,
            "db_updated_estimate": 0,
            "db_total_matches": 0,
            "teams_loaded": 0,
            "league_seasons_loaded": 0,
            "season_summaries": [],
            "message": "No historical seasons selected.",
        }

        if not target_seasons:
            return summary

        for season in target_seasons:
            season_id = int(season["season_id"])
            rows = self.api.get_all_league_matches(season_id=season_id, max_per_page=1000, max_time=max_time)
            normalized = self._normalize_history_matches(rows)

            completed_rows = [row for row in normalized if self._is_completed_status(row.get("status"))]
            trainable_rows = [row for row in completed_rows if self._match_has_result(row)]

            existing_ids = {
                int(r[0])
                for r in self.db.conn.execute("SELECT match_id FROM matches WHERE season_id = ?", (season_id,)).fetchall()
                if r[0] is not None
            }
            current_ids = {
                int(row["match_id"])
                for row in completed_rows
                if row.get("match_id") is not None
            }
            updated_estimate = len(existing_ids.intersection(current_ids))
            added_estimate = max(len(current_ids) - updated_estimate, 0)

            if completed_rows:
                self.db.upsert_matches(completed_rows)

            self._save_raw("matches", f"historical_matches_{season_id}", rows)

            # Load team season stats (shotsAVG, possessionAVG, scoredAVG, xg etc.) for this season.
            # Required for goals_diff, shots_diff, possession_diff, xg_diff — without this call
            # all team-stats-based features fall back to 0.0 at training time.
            teams_loaded = self.load_league_teams(season_id=season_id, max_time=max_time)

            # Load league season stats (draw_pct, home_advantage, avg_goals etc.)
            league_loaded = self.load_league_season(season_id=season_id, max_time=max_time)

            summary["seasons_processed"] += 1
            summary["matches_fetched"] += len(normalized)
            summary["completed_matches_found"] += len(completed_rows)
            summary["completed_matches_with_result"] += len(trainable_rows)
            summary["db_added_estimate"] += added_estimate
            summary["db_updated_estimate"] += updated_estimate
            summary["teams_loaded"] += teams_loaded
            summary["league_seasons_loaded"] += int(league_loaded)
            summary["season_summaries"].append(
                {
                    "season_id": season_id,
                    "year": int(season.get("year", 0) or 0),
                    "league": str(season.get("league", "")),
                    "country": str(season.get("country", "")),
                    "matches_fetched": len(normalized),
                    "completed_matches_found": len(completed_rows),
                    "completed_matches_with_result": len(trainable_rows),
                    "db_added_estimate": added_estimate,
                    "db_updated_estimate": updated_estimate,
                    "teams_loaded": teams_loaded,
                    "league_season_loaded": league_loaded,
                }
            )

        summary["db_total_matches"] = int(self.db.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])

        if summary["completed_matches_found"] <= 0:
            summary["status"] = "empty"
            summary["message"] = "Historical ingest finished, but no completed matches were returned by API."
        elif summary["completed_matches_with_result"] <= 0:
            summary["status"] = "partial"
            summary["message"] = "Completed matches were stored, but none had enough result fields for training."
        else:
            summary["status"] = "ok"
            summary["message"] = "Historical completed matches loaded into SQLite."

        logger.info(
            "historical_completed_ingest status=%s seasons=%s fetched=%s completed=%s trainable=%s "
            "added~=%s updated~=%s teams=%s league_seasons=%s",
            summary["status"],
            summary["seasons_processed"],
            summary["matches_fetched"],
            summary["completed_matches_found"],
            summary["completed_matches_with_result"],
            summary["db_added_estimate"],
            summary["db_updated_estimate"],
            summary["teams_loaded"],
            summary["league_seasons_loaded"],
        )
        return summary

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
