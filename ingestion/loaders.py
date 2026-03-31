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

    def _normalize_history_matches(self, rows: list[dict[str, Any]], season_id: int | None = None) -> list[dict[str, Any]]:
        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            norm = normalize_match(row, season_id=season_id)
            norm["status"] = self._to_completed_status(norm.get("status"))
            normalized_rows.append(norm)
        return normalized_rows

    @staticmethod
    def _extract_match_season_ids(rows: list[dict[str, Any]]) -> list[int]:
        season_ids: set[int] = set()
        for row in rows:
            value = row.get("season_id")
            try:
                season_id = int(value)
            except (TypeError, ValueError):
                continue
            season_ids.add(season_id)
        return sorted(season_ids)

    @staticmethod
    def _append_stats_error(summary: dict[str, Any], season_id: int, stage: str, reason: str, details: str | None = None) -> None:
        summary["stats_load_errors"] += 1
        summary.setdefault("stats_load_error_details", []).append(
            {
                "season_id": season_id,
                "stage": stage,
                "reason": reason,
                "details": details,
            }
        )

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
            "stats_seasons_targeted": 0,
            "targeted_season_ids": [],
            "targeted_seasons_count": 0,
            "matches_fetched": 0,
            "rows_found": 0,
            "completed_matches_found": 0,
            "completed_rows": 0,
            "completed_matches_with_result": 0,
            "matches_with_season_id_count": 0,
            "completed_matches_with_season_id_count": 0,
            "db_added_estimate": 0,
            "db_updated_estimate": 0,
            "db_total_matches": 0,
            "teams_loaded": 0,
            "teams_with_usable_stats_loaded": 0,
            "league_seasons_loaded": 0,
            "team_stats_rows_loaded": 0,
            "league_stats_rows_loaded": 0,
            "stats_load_errors": 0,
            "stats_load_error_details": [],
            "season_summaries": [],
            "message": "No historical seasons selected.",
        }

        if not target_seasons:
            return summary

        target_seasons_by_id: dict[int, dict[str, Any]] = {
            int(season["season_id"]): season for season in target_seasons
        }
        season_ids_from_completed_matches: set[int] = set()

        for season in target_seasons:
            season_id = int(season["season_id"])
            rows = self.api.get_all_league_matches(season_id=season_id, max_per_page=1000, max_time=max_time)
            normalized = self._normalize_history_matches(rows, season_id=season_id)

            completed_rows = [row for row in normalized if self._is_completed_status(row.get("status"))]
            trainable_rows = [row for row in completed_rows if self._match_has_result(row)]
            normalized_season_ids = self._extract_match_season_ids(normalized)
            completed_season_ids = self._extract_match_season_ids(completed_rows)
            season_ids_from_completed_matches.update(completed_season_ids)

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
                # Ensure season id is always persisted even when API payload omits it.
                for row in completed_rows:
                    row["season_id"] = int(season_id)
                self.db.upsert_matches(completed_rows)

            self._save_raw("matches", f"historical_matches_{season_id}", rows)

            summary["seasons_processed"] += 1
            summary["matches_fetched"] += len(normalized)
            summary["rows_found"] += len(normalized)
            summary["completed_matches_found"] += len(completed_rows)
            summary["completed_rows"] += len(completed_rows)
            summary["completed_matches_with_result"] += len(trainable_rows)
            summary["matches_with_season_id_count"] += len([row for row in normalized if row.get("season_id") is not None])
            summary["completed_matches_with_season_id_count"] += len(
                [row for row in completed_rows if row.get("season_id") is not None]
            )
            summary["db_added_estimate"] += added_estimate
            summary["db_updated_estimate"] += updated_estimate
            summary["season_summaries"].append(
                {
                    "season_id": season_id,
                    "year": int(season.get("year", 0) or 0),
                    "league": str(season.get("league", "")),
                    "country": str(season.get("country", "")),
                    "matches_fetched": len(normalized),
                    "completed_matches_found": len(completed_rows),
                    "completed_matches_with_result": len(trainable_rows),
                    "matches_with_season_id_count": len([row for row in normalized if row.get("season_id") is not None]),
                    "completed_matches_with_season_id_count": len(
                        [row for row in completed_rows if row.get("season_id") is not None]
                    ),
                    "normalized_season_ids": normalized_season_ids,
                    "completed_season_ids": completed_season_ids,
                    "db_added_estimate": added_estimate,
                    "db_updated_estimate": updated_estimate,
                }
            )

        targeted_season_ids = sorted(season_ids_from_completed_matches)
        if not targeted_season_ids:
            fallback_targeted = sorted(target_seasons_by_id.keys())
            targeted_season_ids = fallback_targeted
            for fallback_sid in fallback_targeted:
                self._append_stats_error(
                    summary,
                    season_id=fallback_sid,
                    stage="season_targeting",
                    reason="no_completed_match_season_ids",
                    details="No season_id found in normalized completed matches; using requested season_id fallback.",
                )

        summary["targeted_season_ids"] = targeted_season_ids
        summary["targeted_seasons_count"] = len(targeted_season_ids)
        summary["stats_seasons_targeted"] = len(targeted_season_ids)

        for targeted_season_id in targeted_season_ids:
            team_stats_before = int(
                self.db.conn.execute(
                    "SELECT COUNT(*) FROM team_season_stats WHERE season_id = ?",
                    (targeted_season_id,),
                ).fetchone()[0]
            )
            league_stats_before = int(
                self.db.conn.execute(
                    "SELECT COUNT(*) FROM league_season_stats WHERE season_id = ?",
                    (targeted_season_id,),
                ).fetchone()[0]
            )

            teams_loaded = 0
            teams_with_usable_stats = 0
            league_loaded = False

            try:
                team_diag = self._load_league_teams_with_diagnostics(season_id=targeted_season_id, max_time=max_time)
                teams_loaded = int(team_diag["teams_loaded"])
                teams_with_usable_stats = int(team_diag["teams_with_usable_stats"])
                if team_diag["error"] is not None:
                    self._append_stats_error(
                        summary,
                        season_id=targeted_season_id,
                        stage="team_stats",
                        reason=str(team_diag["error"]),
                        details=str(team_diag.get("details") or ""),
                    )
            except Exception as exc:
                self._append_stats_error(
                    summary,
                    season_id=targeted_season_id,
                    stage="team_stats",
                    reason="exception",
                    details=str(exc),
                )
                logger.exception("historical team stats load failed season_id=%s", targeted_season_id)

            try:
                league_diag = self._load_league_season_with_diagnostics(season_id=targeted_season_id, max_time=max_time)
                league_loaded = bool(league_diag["league_loaded"])
                if league_diag["error"] is not None:
                    self._append_stats_error(
                        summary,
                        season_id=targeted_season_id,
                        stage="league_stats",
                        reason=str(league_diag["error"]),
                        details=str(league_diag.get("details") or ""),
                    )
            except Exception as exc:
                self._append_stats_error(
                    summary,
                    season_id=targeted_season_id,
                    stage="league_stats",
                    reason="exception",
                    details=str(exc),
                )
                logger.exception("historical league stats load failed season_id=%s", targeted_season_id)

            team_stats_after = int(
                self.db.conn.execute(
                    "SELECT COUNT(*) FROM team_season_stats WHERE season_id = ?",
                    (targeted_season_id,),
                ).fetchone()[0]
            )
            league_stats_after = int(
                self.db.conn.execute(
                    "SELECT COUNT(*) FROM league_season_stats WHERE season_id = ?",
                    (targeted_season_id,),
                ).fetchone()[0]
            )

            team_stats_delta = max(team_stats_after - team_stats_before, 0)
            league_stats_delta = max(league_stats_after - league_stats_before, 0)

            summary["teams_loaded"] += teams_loaded
            summary["teams_with_usable_stats_loaded"] += teams_with_usable_stats
            summary["league_seasons_loaded"] += int(league_loaded)
            summary["team_stats_rows_loaded"] += team_stats_delta
            summary["league_stats_rows_loaded"] += league_stats_delta

            for season_summary in summary["season_summaries"]:
                if int(season_summary.get("season_id", 0) or 0) != targeted_season_id:
                    continue
                season_summary["teams_loaded"] = teams_loaded
                season_summary["teams_with_usable_stats_loaded"] = teams_with_usable_stats
                season_summary["league_season_loaded"] = league_loaded
                season_summary["team_stats_rows_loaded"] = team_stats_delta
                season_summary["league_stats_rows_loaded"] = league_stats_delta
                break

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
            "historical_completed_ingest status=%s seasons=%s stats_targeted=%s targeted_ids=%s fetched=%s completed=%s "
            "with_sid=%s completed_with_sid=%s trainable=%s added~=%s updated~=%s teams=%s usable_team_stats=%s "
            "league_seasons=%s team_stats_rows=%s league_stats_rows=%s stats_errors=%s",
            summary["status"],
            summary["seasons_processed"],
            summary["stats_seasons_targeted"],
            summary["targeted_season_ids"],
            summary["matches_fetched"],
            summary["completed_matches_found"],
            summary["matches_with_season_id_count"],
            summary["completed_matches_with_season_id_count"],
            summary["completed_matches_with_result"],
            summary["db_added_estimate"],
            summary["db_updated_estimate"],
            summary["teams_loaded"],
            summary["teams_with_usable_stats_loaded"],
            summary["league_seasons_loaded"],
            summary["team_stats_rows_loaded"],
            summary["league_stats_rows_loaded"],
            summary["stats_load_errors"],
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
        diagnostics = self._load_league_season_with_diagnostics(season_id=season_id, max_time=max_time)
        if diagnostics["error"] is not None:
            logger.warning(
                "league season stats unavailable season_id=%s reason=%s details=%s",
                season_id,
                diagnostics["error"],
                diagnostics.get("details") or "",
            )
        return bool(diagnostics["league_loaded"])

    def load_league_teams(self, season_id: int, max_time: int | None = None) -> int:
        diagnostics = self._load_league_teams_with_diagnostics(season_id=season_id, max_time=max_time)
        if diagnostics["error"] is not None:
            logger.warning(
                "league teams stats unavailable season_id=%s reason=%s details=%s",
                season_id,
                diagnostics["error"],
                diagnostics.get("details") or "",
            )
        return int(diagnostics["teams_loaded"])

    def _load_league_season_with_diagnostics(self, season_id: int, max_time: int | None = None) -> dict[str, Any]:
        row = self.api.get_league_season(season_id=season_id, max_time=max_time)
        if not row:
            return {
                "season_id": season_id,
                "league_loaded": False,
                "error": "empty_api_payload",
                "details": "league-season endpoint returned no data",
            }

        normalized = normalize_league_season_stats(row=row, season_id=season_id)
        self._save_raw("seasons", f"season_{season_id}", row)
        self.db.upsert_league_season_stats(normalized)
        logger.info("league season loaded season_id=%s", season_id)
        return {
            "season_id": season_id,
            "league_loaded": True,
            "error": None,
            "details": None,
        }

    def _load_league_teams_with_diagnostics(self, season_id: int, max_time: int | None = None) -> dict[str, Any]:
        rows = self.api.get_all_league_teams(season_id=season_id, include_stats=True, max_time=max_time)
        if not rows:
            return {
                "season_id": season_id,
                "teams_loaded": 0,
                "teams_with_usable_stats": 0,
                "error": "empty_api_payload",
                "details": "league-teams endpoint returned no rows (include=stats)",
            }

        normalized = [normalize_team(row=row, season_id=season_id) for row in rows if isinstance(row, dict)]
        if not normalized:
            return {
                "season_id": season_id,
                "teams_loaded": 0,
                "teams_with_usable_stats": 0,
                "error": "no_valid_rows_after_normalization",
                "details": "rows existed but none were valid dict payloads for team normalization",
            }

        teams_with_usable_stats = 0
        for team in normalized:
            stats = team.get("stats") if isinstance(team.get("stats"), dict) else {}
            if any(value is not None for value in stats.values()):
                teams_with_usable_stats += 1

        self._save_raw("teams", f"league_teams_{season_id}", rows)
        self.db.upsert_teams(normalized)
        logger.info(
            "league teams loaded=%s usable_stats=%s season_id=%s",
            len(normalized),
            teams_with_usable_stats,
            season_id,
        )
        error = None
        details = None
        if teams_with_usable_stats <= 0:
            error = "stats_missing_or_empty"
            details = "teams were loaded but stats payload is empty or all-null"

        return {
            "season_id": season_id,
            "teams_loaded": len(normalized),
            "teams_with_usable_stats": teams_with_usable_stats,
            "error": error,
            "details": details,
        }

    def load_league_matches(self, season_id: int, max_time: int | None = None) -> int:
        rows = self.api.get_all_league_matches(season_id=season_id, max_per_page=1000, max_time=max_time)
        normalized = [normalize_match(row, season_id=season_id) for row in rows]
        self._save_raw("matches", f"league_matches_{season_id}", rows)
        self.db.upsert_matches(normalized)
        logger.info("league matches loaded=%s season_id=%s", len(normalized), season_id)
        return len(normalized)
