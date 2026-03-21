from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ingestion.validators import normalize_missing


def _to_int(value: Any) -> int | None:
    value = normalize_missing(value)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    value = normalize_missing(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_iso_from_unix(value: Any) -> str | None:
    timestamp = _to_int(value)
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def normalize_country(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _to_int(row.get("id") or row.get("country_id")),
        "name": row.get("name") or row.get("country_name") or "",
        "raw": row,
    }


def normalize_league(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "season_id": _to_int(row.get("season_id")),
        "league_name": row.get("league_name") or row.get("name") or "",
        "country_name": row.get("country_name") or row.get("country") or "",
        "season_name": row.get("season_name") or row.get("season") or "",
        "chosen_flag": bool(row.get("chosen_leagues_only", True)),
        "raw": row,
    }


def normalize_league_season_stats(row: dict[str, Any], season_id: int) -> dict[str, Any]:
    return {
        "season_id": season_id,
        "progress": _to_float(row.get("progress")),
        "total_matches": _to_int(row.get("total_matches")),
        "matches_completed": _to_int(row.get("matches_completed")),
        "home_win_pct": _to_float(row.get("home_win_percentage") or row.get("home_win_pct")),
        "draw_pct": _to_float(row.get("draw_percentage") or row.get("draw_pct")),
        "away_win_pct": _to_float(row.get("away_win_percentage") or row.get("away_win_pct")),
        "btts_pct": _to_float(row.get("btts_percentage") or row.get("btts_pct")),
        "season_avg_goals": _to_float(row.get("seasonAVG") or row.get("season_avg_goals")),
        "home_advantage": _to_float(row.get("home_advantage")),
        "corners_avg": _to_float(row.get("corners_average") or row.get("corners_avg")),
        "cards_avg": _to_float(row.get("cards_average") or row.get("cards_avg")),
        "raw": row,
    }


def normalize_team(row: dict[str, Any], season_id: int) -> dict[str, Any]:
    stats = row.get("stats", {}) if isinstance(row.get("stats"), dict) else {}

    def stat(*keys: str) -> float | None:
        for key in keys:
            if key in stats:
                return _to_float(stats.get(key))
        return None

    return {
        "team_id": _to_int(row.get("id") or row.get("team_id")),
        "season_id": season_id,
        "name": row.get("name") or "",
        "clean_name": row.get("cleanName") or row.get("clean_name") or row.get("name") or "",
        "country": row.get("country") or "",
        "table_position": _to_int(row.get("table_position") or row.get("position")),
        "performance_rank": _to_float(row.get("performance_rank")),
        "prediction_risk": _to_float(row.get("prediction_risk")),
        "stats": {
            "season_ppg_overall": stat("ppg_overall", "season_ppg_overall"),
            "season_ppg_home": stat("ppg_home", "season_ppg_home"),
            "season_ppg_away": stat("ppg_away", "season_ppg_away"),
            "win_pct_overall": stat("win_percentage_overall", "win_pct_overall"),
            "win_pct_home": stat("win_percentage_home", "win_pct_home"),
            "win_pct_away": stat("win_percentage_away", "win_pct_away"),
            "draw_pct_overall": stat("draw_percentage_overall", "draw_pct_overall"),
            "draw_pct_home": stat("draw_percentage_home", "draw_pct_home"),
            "draw_pct_away": stat("draw_percentage_away", "draw_pct_away"),
            "lose_pct_overall": stat("lose_percentage_overall", "lose_pct_overall"),
            "lose_pct_home": stat("lose_percentage_home", "lose_pct_home"),
            "lose_pct_away": stat("lose_percentage_away", "lose_pct_away"),
            "goals_for_avg_overall": stat("scoredAVG_overall", "goals_for_avg_overall"),
            "goals_for_avg_home": stat("scoredAVG_home", "goals_for_avg_home"),
            "goals_for_avg_away": stat("scoredAVG_away", "goals_for_avg_away"),
            "goals_against_avg_overall": stat("concededAVG_overall", "goals_against_avg_overall"),
            "goals_against_avg_home": stat("concededAVG_home", "goals_against_avg_home"),
            "goals_against_avg_away": stat("concededAVG_away", "goals_against_avg_away"),
            "btts_pct_overall": stat("btts_percentage_overall", "btts_pct_overall"),
            "btts_pct_home": stat("btts_percentage_home", "btts_pct_home"),
            "btts_pct_away": stat("btts_percentage_away", "btts_pct_away"),
            "over25_pct_overall": stat("over25_percentage_overall", "over25_pct_overall"),
            "over25_pct_home": stat("over25_percentage_home", "over25_pct_home"),
            "over25_pct_away": stat("over25_percentage_away", "over25_pct_away"),
            "corners_avg_overall": stat("cornersAVG_overall", "corners_avg_overall"),
            "corners_avg_home": stat("cornersAVG_home", "corners_avg_home"),
            "corners_avg_away": stat("cornersAVG_away", "corners_avg_away"),
            "cards_avg_overall": stat("cardsAVG_overall", "cards_avg_overall"),
            "cards_avg_home": stat("cardsAVG_home", "cards_avg_home"),
            "cards_avg_away": stat("cardsAVG_away", "cards_avg_away"),
            "shots_avg_overall": stat("shotsAVG_overall", "shots_avg_overall"),
            "shots_avg_home": stat("shotsAVG_home", "shots_avg_home"),
            "shots_avg_away": stat("shotsAVG_away", "shots_avg_away"),
            "shots_on_target_avg_overall": stat("shotsOnTargetAVG_overall", "shots_on_target_avg_overall"),
            "shots_on_target_avg_home": stat("shotsOnTargetAVG_home", "shots_on_target_avg_home"),
            "shots_on_target_avg_away": stat("shotsOnTargetAVG_away", "shots_on_target_avg_away"),
            "possession_avg_overall": stat("possessionAVG_overall", "possession_avg_overall"),
            "possession_avg_home": stat("possessionAVG_home", "possession_avg_home"),
            "possession_avg_away": stat("possessionAVG_away", "possession_avg_away"),
            "xg_for_avg_overall": stat("xg_for_avg_overall"),
            "xg_for_avg_home": stat("xg_for_avg_home"),
            "xg_for_avg_away": stat("xg_for_avg_away"),
            "xg_against_avg_overall": stat("xg_against_avg_overall"),
            "xg_against_avg_home": stat("xg_against_avg_home"),
            "xg_against_avg_away": stat("xg_against_avg_away"),
        },
        "raw": row,
    }


def normalize_match(row: dict[str, Any]) -> dict[str, Any]:
    home_id = _to_int(row.get("homeID") or row.get("home_team_id"))
    away_id = _to_int(row.get("awayID") or row.get("away_team_id"))
    winning_id = _to_int(row.get("winningTeam") or row.get("winning_team_id"))
    return {
        "match_id": _to_int(row.get("id") or row.get("match_id")),
        "season_id": _to_int(row.get("season_id")),
        "date_unix": _to_int(row.get("date_unix") or row.get("date")),
        "match_date_iso": row.get("match_date_iso") or _to_iso_from_unix(row.get("date_unix") or row.get("date")),
        "status": str(row.get("status", "")).lower() or None,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "home_team_name": row.get("home_name") or row.get("home_team_name") or "",
        "away_team_name": row.get("away_name") or row.get("away_team_name") or "",
        "home_goals": _to_int(row.get("homeGoalCount") or row.get("home_goals")),
        "away_goals": _to_int(row.get("awayGoalCount") or row.get("away_goals")),
        "winning_team_id": winning_id,
        "odds_ft_1": _to_float(row.get("odds_ft_1")),
        "odds_ft_x": _to_float(row.get("odds_ft_x")),
        "odds_ft_2": _to_float(row.get("odds_ft_2")),
        "btts_potential": _to_float(row.get("btts_potential")),
        "o15_potential": _to_float(row.get("o15_potential")),
        "o25_potential": _to_float(row.get("o25_potential")),
        "o35_potential": _to_float(row.get("o35_potential")),
        "o45_potential": _to_float(row.get("o45_potential")),
        "corners_potential": _to_float(row.get("corners_potential")),
        "cards_potential": _to_float(row.get("cards_potential")),
        "avg_potential": _to_float(row.get("average_potential") or row.get("avg_potential")),
        "home_ppg": _to_float(row.get("home_ppg")),
        "away_ppg": _to_float(row.get("away_ppg")),
        "pre_match_home_ppg": _to_float(row.get("pre_match_home_ppg")),
        "pre_match_away_ppg": _to_float(row.get("pre_match_away_ppg")),
        "raw": row,
    }
