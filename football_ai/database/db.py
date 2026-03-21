from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from football_ai.config import settings

logger = logging.getLogger("database")


class Database:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def initialize(self, schema_path: str = "football_ai/database/schema.sql") -> None:
        schema = Path(schema_path).read_text(encoding="utf-8")
        self.conn.executescript(schema)
        self.conn.commit()
        logger.info("schema initialized path=%s", self.db_path)

    def _upsert(self, query: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.conn.executemany(query, rows)
        self.conn.commit()

    def upsert_countries(self, countries: list[dict[str, Any]]) -> None:
        rows = [
            {"id": row["id"], "name": row.get("name", ""), "raw_json": json.dumps(row, ensure_ascii=False)}
            for row in countries
            if row.get("id") is not None
        ]
        self._upsert(
            """
            INSERT INTO countries (id, name, raw_json, updated_at)
            VALUES (:id, :name, :raw_json, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        logger.info("upserted countries=%s", len(rows))

    def upsert_leagues(self, leagues: list[dict[str, Any]]) -> None:
        rows = [
            {
                "season_id": row["season_id"],
                "league_name": row.get("league_name") or row.get("name", ""),
                "country_name": row.get("country_name", ""),
                "season_label": row.get("season_name") or row.get("season_label", ""),
                "chosen_flag": int(bool(row.get("chosen_flag", True))),
                "raw_json": json.dumps(row, ensure_ascii=False),
            }
            for row in leagues
            if row.get("season_id") is not None
        ]
        self._upsert(
            """
            INSERT INTO leagues (season_id, league_name, country_name, season_label, chosen_flag, raw_json, updated_at)
            VALUES (:season_id, :league_name, :country_name, :season_label, :chosen_flag, :raw_json, CURRENT_TIMESTAMP)
            ON CONFLICT(season_id) DO UPDATE SET
                league_name=excluded.league_name,
                country_name=excluded.country_name,
                season_label=excluded.season_label,
                chosen_flag=excluded.chosen_flag,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        logger.info("upserted leagues=%s", len(rows))

    def upsert_league_season_stats(self, stat: dict[str, Any]) -> None:
        if not stat.get("season_id"):
            return
        row = {
            "season_id": stat["season_id"],
            "progress": stat.get("progress"),
            "total_matches": stat.get("total_matches"),
            "matches_completed": stat.get("matches_completed"),
            "home_win_pct": stat.get("home_win_pct"),
            "draw_pct": stat.get("draw_pct"),
            "away_win_pct": stat.get("away_win_pct"),
            "btts_pct": stat.get("btts_pct"),
            "season_avg_goals": stat.get("season_avg_goals"),
            "home_advantage": stat.get("home_advantage"),
            "corners_avg": stat.get("corners_avg"),
            "cards_avg": stat.get("cards_avg"),
            "raw_json": json.dumps(stat, ensure_ascii=False),
        }
        self.conn.execute(
            """
            INSERT INTO league_season_stats (
                season_id, progress, total_matches, matches_completed, home_win_pct, draw_pct, away_win_pct,
                btts_pct, season_avg_goals, home_advantage, corners_avg, cards_avg, raw_json, updated_at
            ) VALUES (
                :season_id, :progress, :total_matches, :matches_completed, :home_win_pct, :draw_pct, :away_win_pct,
                :btts_pct, :season_avg_goals, :home_advantage, :corners_avg, :cards_avg, :raw_json, CURRENT_TIMESTAMP
            )
            ON CONFLICT(season_id) DO UPDATE SET
                progress=excluded.progress,
                total_matches=excluded.total_matches,
                matches_completed=excluded.matches_completed,
                home_win_pct=excluded.home_win_pct,
                draw_pct=excluded.draw_pct,
                away_win_pct=excluded.away_win_pct,
                btts_pct=excluded.btts_pct,
                season_avg_goals=excluded.season_avg_goals,
                home_advantage=excluded.home_advantage,
                corners_avg=excluded.corners_avg,
                cards_avg=excluded.cards_avg,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            row,
        )
        self.conn.commit()
        logger.info("upserted league season stats season_id=%s", row["season_id"])

    def upsert_teams(self, teams: list[dict[str, Any]]) -> None:
        if not teams:
            return
        team_rows: list[dict[str, Any]] = []
        stat_rows: list[dict[str, Any]] = []
        for team in teams:
            if team.get("team_id") is None or team.get("season_id") is None:
                continue
            team_rows.append(
                {
                    "team_id": team["team_id"],
                    "season_id": team["season_id"],
                    "name": team.get("name", ""),
                    "clean_name": team.get("clean_name", ""),
                    "country": team.get("country", ""),
                    "table_position": team.get("table_position"),
                    "performance_rank": team.get("performance_rank"),
                    "prediction_risk": team.get("prediction_risk"),
                    "raw_json": json.dumps(team.get("raw", team), ensure_ascii=False),
                }
            )
            stats = team.get("stats", {})
            stat_rows.append(
                {
                    "team_id": team["team_id"],
                    "season_id": team["season_id"],
                    "season_ppg_overall": stats.get("season_ppg_overall"),
                    "season_ppg_home": stats.get("season_ppg_home"),
                    "season_ppg_away": stats.get("season_ppg_away"),
                    "win_pct_overall": stats.get("win_pct_overall"),
                    "win_pct_home": stats.get("win_pct_home"),
                    "win_pct_away": stats.get("win_pct_away"),
                    "draw_pct_overall": stats.get("draw_pct_overall"),
                    "draw_pct_home": stats.get("draw_pct_home"),
                    "draw_pct_away": stats.get("draw_pct_away"),
                    "lose_pct_overall": stats.get("lose_pct_overall"),
                    "lose_pct_home": stats.get("lose_pct_home"),
                    "lose_pct_away": stats.get("lose_pct_away"),
                    "goals_for_avg_overall": stats.get("goals_for_avg_overall"),
                    "goals_for_avg_home": stats.get("goals_for_avg_home"),
                    "goals_for_avg_away": stats.get("goals_for_avg_away"),
                    "goals_against_avg_overall": stats.get("goals_against_avg_overall"),
                    "goals_against_avg_home": stats.get("goals_against_avg_home"),
                    "goals_against_avg_away": stats.get("goals_against_avg_away"),
                    "btts_pct_overall": stats.get("btts_pct_overall"),
                    "btts_pct_home": stats.get("btts_pct_home"),
                    "btts_pct_away": stats.get("btts_pct_away"),
                    "over25_pct_overall": stats.get("over25_pct_overall"),
                    "over25_pct_home": stats.get("over25_pct_home"),
                    "over25_pct_away": stats.get("over25_pct_away"),
                    "corners_avg_overall": stats.get("corners_avg_overall"),
                    "corners_avg_home": stats.get("corners_avg_home"),
                    "corners_avg_away": stats.get("corners_avg_away"),
                    "cards_avg_overall": stats.get("cards_avg_overall"),
                    "cards_avg_home": stats.get("cards_avg_home"),
                    "cards_avg_away": stats.get("cards_avg_away"),
                    "shots_avg_overall": stats.get("shots_avg_overall"),
                    "shots_avg_home": stats.get("shots_avg_home"),
                    "shots_avg_away": stats.get("shots_avg_away"),
                    "shots_on_target_avg_overall": stats.get("shots_on_target_avg_overall"),
                    "shots_on_target_avg_home": stats.get("shots_on_target_avg_home"),
                    "shots_on_target_avg_away": stats.get("shots_on_target_avg_away"),
                    "possession_avg_overall": stats.get("possession_avg_overall"),
                    "possession_avg_home": stats.get("possession_avg_home"),
                    "possession_avg_away": stats.get("possession_avg_away"),
                    "xg_for_avg_overall": stats.get("xg_for_avg_overall"),
                    "xg_for_avg_home": stats.get("xg_for_avg_home"),
                    "xg_for_avg_away": stats.get("xg_for_avg_away"),
                    "xg_against_avg_overall": stats.get("xg_against_avg_overall"),
                    "xg_against_avg_home": stats.get("xg_against_avg_home"),
                    "xg_against_avg_away": stats.get("xg_against_avg_away"),
                    "raw_json": json.dumps(stats, ensure_ascii=False),
                }
            )

        self._upsert(
            """
            INSERT INTO teams (
                team_id, season_id, name, clean_name, country, table_position, performance_rank,
                prediction_risk, raw_json, updated_at
            ) VALUES (
                :team_id, :season_id, :name, :clean_name, :country, :table_position, :performance_rank,
                :prediction_risk, :raw_json, CURRENT_TIMESTAMP
            )
            ON CONFLICT(team_id, season_id) DO UPDATE SET
                name=excluded.name,
                clean_name=excluded.clean_name,
                country=excluded.country,
                table_position=excluded.table_position,
                performance_rank=excluded.performance_rank,
                prediction_risk=excluded.prediction_risk,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            team_rows,
        )

        self._upsert(
            """
            INSERT INTO team_season_stats (
                team_id, season_id, season_ppg_overall, season_ppg_home, season_ppg_away,
                win_pct_overall, win_pct_home, win_pct_away, draw_pct_overall, draw_pct_home, draw_pct_away,
                lose_pct_overall, lose_pct_home, lose_pct_away, goals_for_avg_overall, goals_for_avg_home,
                goals_for_avg_away, goals_against_avg_overall, goals_against_avg_home, goals_against_avg_away,
                btts_pct_overall, btts_pct_home, btts_pct_away, over25_pct_overall, over25_pct_home,
                over25_pct_away, corners_avg_overall, corners_avg_home, corners_avg_away, cards_avg_overall,
                cards_avg_home, cards_avg_away, shots_avg_overall, shots_avg_home, shots_avg_away,
                shots_on_target_avg_overall, shots_on_target_avg_home, shots_on_target_avg_away,
                possession_avg_overall, possession_avg_home, possession_avg_away, xg_for_avg_overall,
                xg_for_avg_home, xg_for_avg_away, xg_against_avg_overall, xg_against_avg_home,
                xg_against_avg_away, raw_json, updated_at
            ) VALUES (
                :team_id, :season_id, :season_ppg_overall, :season_ppg_home, :season_ppg_away,
                :win_pct_overall, :win_pct_home, :win_pct_away, :draw_pct_overall, :draw_pct_home, :draw_pct_away,
                :lose_pct_overall, :lose_pct_home, :lose_pct_away, :goals_for_avg_overall, :goals_for_avg_home,
                :goals_for_avg_away, :goals_against_avg_overall, :goals_against_avg_home, :goals_against_avg_away,
                :btts_pct_overall, :btts_pct_home, :btts_pct_away, :over25_pct_overall, :over25_pct_home,
                :over25_pct_away, :corners_avg_overall, :corners_avg_home, :corners_avg_away, :cards_avg_overall,
                :cards_avg_home, :cards_avg_away, :shots_avg_overall, :shots_avg_home, :shots_avg_away,
                :shots_on_target_avg_overall, :shots_on_target_avg_home, :shots_on_target_avg_away,
                :possession_avg_overall, :possession_avg_home, :possession_avg_away, :xg_for_avg_overall,
                :xg_for_avg_home, :xg_for_avg_away, :xg_against_avg_overall, :xg_against_avg_home,
                :xg_against_avg_away, :raw_json, CURRENT_TIMESTAMP
            )
            ON CONFLICT(team_id, season_id) DO UPDATE SET
                season_ppg_overall=excluded.season_ppg_overall,
                season_ppg_home=excluded.season_ppg_home,
                season_ppg_away=excluded.season_ppg_away,
                win_pct_overall=excluded.win_pct_overall,
                win_pct_home=excluded.win_pct_home,
                win_pct_away=excluded.win_pct_away,
                draw_pct_overall=excluded.draw_pct_overall,
                draw_pct_home=excluded.draw_pct_home,
                draw_pct_away=excluded.draw_pct_away,
                lose_pct_overall=excluded.lose_pct_overall,
                lose_pct_home=excluded.lose_pct_home,
                lose_pct_away=excluded.lose_pct_away,
                goals_for_avg_overall=excluded.goals_for_avg_overall,
                goals_for_avg_home=excluded.goals_for_avg_home,
                goals_for_avg_away=excluded.goals_for_avg_away,
                goals_against_avg_overall=excluded.goals_against_avg_overall,
                goals_against_avg_home=excluded.goals_against_avg_home,
                goals_against_avg_away=excluded.goals_against_avg_away,
                btts_pct_overall=excluded.btts_pct_overall,
                btts_pct_home=excluded.btts_pct_home,
                btts_pct_away=excluded.btts_pct_away,
                over25_pct_overall=excluded.over25_pct_overall,
                over25_pct_home=excluded.over25_pct_home,
                over25_pct_away=excluded.over25_pct_away,
                corners_avg_overall=excluded.corners_avg_overall,
                corners_avg_home=excluded.corners_avg_home,
                corners_avg_away=excluded.corners_avg_away,
                cards_avg_overall=excluded.cards_avg_overall,
                cards_avg_home=excluded.cards_avg_home,
                cards_avg_away=excluded.cards_avg_away,
                shots_avg_overall=excluded.shots_avg_overall,
                shots_avg_home=excluded.shots_avg_home,
                shots_avg_away=excluded.shots_avg_away,
                shots_on_target_avg_overall=excluded.shots_on_target_avg_overall,
                shots_on_target_avg_home=excluded.shots_on_target_avg_home,
                shots_on_target_avg_away=excluded.shots_on_target_avg_away,
                possession_avg_overall=excluded.possession_avg_overall,
                possession_avg_home=excluded.possession_avg_home,
                possession_avg_away=excluded.possession_avg_away,
                xg_for_avg_overall=excluded.xg_for_avg_overall,
                xg_for_avg_home=excluded.xg_for_avg_home,
                xg_for_avg_away=excluded.xg_for_avg_away,
                xg_against_avg_overall=excluded.xg_against_avg_overall,
                xg_against_avg_home=excluded.xg_against_avg_home,
                xg_against_avg_away=excluded.xg_against_avg_away,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            stat_rows,
        )
        logger.info("upserted teams=%s", len(team_rows))

    def upsert_matches(self, matches: list[dict[str, Any]]) -> None:
        rows = [
            {
                "match_id": match["match_id"],
                "season_id": match.get("season_id"),
                "date_unix": match.get("date_unix"),
                "match_date_iso": match.get("match_date_iso"),
                "status": match.get("status"),
                "home_team_id": match.get("home_team_id"),
                "away_team_id": match.get("away_team_id"),
                "home_team_name": match.get("home_team_name"),
                "away_team_name": match.get("away_team_name"),
                "home_goals": match.get("home_goals"),
                "away_goals": match.get("away_goals"),
                "winning_team_id": match.get("winning_team_id"),
                "odds_ft_1": match.get("odds_ft_1"),
                "odds_ft_x": match.get("odds_ft_x"),
                "odds_ft_2": match.get("odds_ft_2"),
                "btts_potential": match.get("btts_potential"),
                "o15_potential": match.get("o15_potential"),
                "o25_potential": match.get("o25_potential"),
                "o35_potential": match.get("o35_potential"),
                "o45_potential": match.get("o45_potential"),
                "corners_potential": match.get("corners_potential"),
                "cards_potential": match.get("cards_potential"),
                "avg_potential": match.get("avg_potential"),
                "home_ppg": match.get("home_ppg"),
                "away_ppg": match.get("away_ppg"),
                "pre_match_home_ppg": match.get("pre_match_home_ppg"),
                "pre_match_away_ppg": match.get("pre_match_away_ppg"),
                "raw_json": json.dumps(match.get("raw", match), ensure_ascii=False),
            }
            for match in matches
            if match.get("match_id") is not None
        ]
        self._upsert(
            """
            INSERT INTO matches (
                match_id, season_id, date_unix, match_date_iso, status, home_team_id, away_team_id,
                home_team_name, away_team_name, home_goals, away_goals, winning_team_id,
                odds_ft_1, odds_ft_x, odds_ft_2, btts_potential, o15_potential, o25_potential,
                o35_potential, o45_potential, corners_potential, cards_potential, avg_potential,
                home_ppg, away_ppg, pre_match_home_ppg, pre_match_away_ppg, raw_json, updated_at
            ) VALUES (
                :match_id, :season_id, :date_unix, :match_date_iso, :status, :home_team_id, :away_team_id,
                :home_team_name, :away_team_name, :home_goals, :away_goals, :winning_team_id,
                :odds_ft_1, :odds_ft_x, :odds_ft_2, :btts_potential, :o15_potential, :o25_potential,
                :o35_potential, :o45_potential, :corners_potential, :cards_potential, :avg_potential,
                :home_ppg, :away_ppg, :pre_match_home_ppg, :pre_match_away_ppg, :raw_json, CURRENT_TIMESTAMP
            )
            ON CONFLICT(match_id) DO UPDATE SET
                season_id=excluded.season_id,
                date_unix=excluded.date_unix,
                match_date_iso=excluded.match_date_iso,
                status=excluded.status,
                home_team_id=excluded.home_team_id,
                away_team_id=excluded.away_team_id,
                home_team_name=excluded.home_team_name,
                away_team_name=excluded.away_team_name,
                home_goals=excluded.home_goals,
                away_goals=excluded.away_goals,
                winning_team_id=excluded.winning_team_id,
                odds_ft_1=excluded.odds_ft_1,
                odds_ft_x=excluded.odds_ft_x,
                odds_ft_2=excluded.odds_ft_2,
                btts_potential=excluded.btts_potential,
                o15_potential=excluded.o15_potential,
                o25_potential=excluded.o25_potential,
                o35_potential=excluded.o35_potential,
                o45_potential=excluded.o45_potential,
                corners_potential=excluded.corners_potential,
                cards_potential=excluded.cards_potential,
                avg_potential=excluded.avg_potential,
                home_ppg=excluded.home_ppg,
                away_ppg=excluded.away_ppg,
                pre_match_home_ppg=excluded.pre_match_home_ppg,
                pre_match_away_ppg=excluded.pre_match_away_ppg,
                raw_json=excluded.raw_json,
                updated_at=CURRENT_TIMESTAMP
            """,
            rows,
        )
        logger.info("upserted matches=%s", len(rows))

    def close(self) -> None:
        self.conn.close()
