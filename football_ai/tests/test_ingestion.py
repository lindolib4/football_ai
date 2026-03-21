from __future__ import annotations

from football_ai.ingestion.normalizers import normalize_match, normalize_team
from football_ai.ingestion.validators import normalize_missing


def test_missing_values_converted_to_none() -> None:
    assert normalize_missing(-1) is None
    assert normalize_missing(-2) is None
    assert normalize_missing("-1") is None


def test_normalize_match_handles_missing_stats() -> None:
    raw = {
        "id": 10,
        "status": "complete",
        "homeID": 100,
        "awayID": 200,
        "shots_home": -1,
        "odds_ft_1": "1.95",
        "odds_ft_x": -2,
        "odds_ft_2": "3.50",
    }
    normalized = normalize_match(raw)
    assert normalized["match_id"] == 10
    assert normalized["odds_ft_1"] == 1.95
    assert normalized["odds_ft_x"] is None


def test_normalize_team_extracts_stats() -> None:
    raw = {
        "id": 1,
        "name": "Team A",
        "stats": {"ppg_overall": "1.8", "shotsAVG_home": -1},
    }
    normalized = normalize_team(raw, season_id=55)
    assert normalized["team_id"] == 1
    assert normalized["stats"]["season_ppg_overall"] == 1.8
    assert normalized["stats"]["shots_avg_home"] is None
