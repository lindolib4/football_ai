from __future__ import annotations

from typing import Any

import pytest

from ingestion.normalizers import normalize_match
from scheduler.auto_train import AutoTrainer


def test_runtime_feature_snapshot_odds_only_context() -> None:
    trainer = AutoTrainer()
    required_columns = [
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "goals_diff",
        "ppg_diff",
    ]
    match = {
        "odds_ft_1": 2.1,
        "odds_ft_x": 3.3,
        "odds_ft_2": 3.8,
    }

    snapshot = trainer.build_runtime_feature_snapshot(match=match, required_columns=required_columns)

    assert snapshot["status"] == "ok"
    assert snapshot["context_level"] == "odds_only_context"
    assert snapshot["feature_context_level"] == "odds_only_context"
    assert snapshot["market_only_or_market_heavy_flag"] is True
    assert snapshot["degraded_context_flag"] is False
    assert snapshot["missing_feature_count"] >= 1
    features = snapshot["features"]
    assert isinstance(features, dict)
    assert list(features.keys()) == required_columns
    assert features["goals_diff"] == 0.0
    assert features["ppg_diff"] == 0.0


def test_runtime_feature_snapshot_partial_context_from_payload() -> None:
    trainer = AutoTrainer()
    required_columns = [
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "goals_diff",
        "ppg_diff",
        "shots_diff",
    ]
    match = {
        "odds_ft_1": 1.95,
        "odds_ft_x": 3.2,
        "odds_ft_2": 4.4,
        "features": {
            "goals_diff": 0.42,
            "ppg_diff": 0.31,
        },
    }

    snapshot = trainer.build_runtime_feature_snapshot(match=match, required_columns=required_columns)

    assert snapshot["status"] == "ok"
    assert snapshot["context_level"] == "partial_context"
    assert snapshot["feature_context_level"] == "partial_context"
    assert snapshot["missing_non_market_count"] >= 1
    assert "goals_diff" in snapshot["available_feature_names"]
    features = snapshot["features"]
    assert features["goals_diff"] == 0.42
    assert features["ppg_diff"] == 0.31
    assert features["shots_diff"] == 0.0


def test_runtime_feature_snapshot_derives_ppg_features_from_match_context() -> None:
    trainer = AutoTrainer()
    required_columns = [
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "ppg_diff",
        "home_home_ppg",
        "away_away_ppg",
        "split_advantage",
    ]
    match = {
        "odds_ft_1": 2.02,
        "odds_ft_x": 3.20,
        "odds_ft_2": 3.95,
        "home_ppg": 1.8,
        "away_ppg": 1.2,
        "pre_match_home_ppg": 2.0,
        "pre_match_away_ppg": 1.1,
    }

    snapshot = trainer.build_runtime_feature_snapshot(match=match, required_columns=required_columns)

    assert snapshot["status"] == "ok"
    assert snapshot["context_level"] in {"partial_context", "full_context"}
    features = snapshot["features"]
    assert features["ppg_diff"] == pytest.approx(0.6)
    assert features["home_home_ppg"] == pytest.approx(1.8)
    assert features["away_away_ppg"] == pytest.approx(1.2)
    assert features["split_advantage"] == pytest.approx(0.9)
    assert "ppg_diff" in snapshot["available_feature_names"]
    assert snapshot["missing_non_market_count"] <= 1


def test_prepare_toto_match_for_inference_exposes_runtime_features() -> None:
    trainer = AutoTrainer()
    match = {
        "odds_ft_1": 2.0,
        "odds_ft_x": 3.1,
        "odds_ft_2": 3.7,
        "features": {
            "goals_diff": 0.2,
            "ppg_diff": 0.1,
        },
        "pool_probs": {"P1": 0.43, "PX": 0.29, "P2": 0.28},
    }

    result = trainer.prepare_toto_match_for_inference(match=match)

    runtime_features = result.get("runtime_features")
    assert isinstance(runtime_features, dict)
    assert len(runtime_features) > 0
    assert result.get("feature_context_level") in {
        "full_context",
        "partial_context",
        "odds_only_context",
        "degraded_context",
    }


def test_normalize_match_preserves_home_away_and_odds_1_2_mapping() -> None:
    raw = {
        "id": 123,
        "homeID": 11,
        "awayID": 22,
        "home_name": "Home FC",
        "away_name": "Away FC",
        "odds_ft_1": 1.72,
        "odds_ft_x": 3.45,
        "odds_ft_2": 4.85,
        "date_unix": 1774742400,
        "status": "incomplete",
    }

    normalized = normalize_match(raw)

    assert normalized["home_team_name"] == "Home FC"
    assert normalized["away_team_name"] == "Away FC"
    assert normalized["home_team_id"] == 11
    assert normalized["away_team_id"] == 22
    assert normalized["odds_ft_1"] == pytest.approx(1.72)
    assert normalized["odds_ft_2"] == pytest.approx(4.85)
    assert normalized["odds_ft_1"] < normalized["odds_ft_2"]


def test_prepare_toto_match_for_inference_keeps_implied_prob_1_vs_2_orientation() -> None:
    trainer = AutoTrainer()
    match = {
        "odds_ft_1": 1.50,
        "odds_ft_x": 4.20,
        "odds_ft_2": 6.50,
        "pool_probs": {"P1": 0.33, "PX": 0.34, "P2": 0.33},
    }

    result = trainer.prepare_toto_match_for_inference(match=match)
    implied = result.get("implied_probs")
    assert isinstance(implied, dict)

    p1 = float(implied["P1"])
    px = float(implied["PX"])
    p2 = float(implied["P2"])

    # Lower odds for outcome "1" must produce the highest implied probability.
    assert p1 > px
    assert p1 > p2
    assert p2 < px


def test_build_runtime_features_for_diagnostics_derives_ppg_from_home_away_ppg() -> None:
    """_build_runtime_features_for_diagnostics must derive PPG/split features from
    home_ppg / away_ppg / pre_match_home_ppg keys, not only exact column names."""
    trainer = AutoTrainer()
    required_columns = [
        "ppg_diff",
        "home_home_ppg",
        "away_away_ppg",
        "split_advantage",
        "avg_goals",
    ]
    # Match uses API-level key names, not internal feature column names
    match: dict[str, Any] = {
        "home_ppg": 1.9,
        "away_ppg": 1.3,
        "pre_match_home_ppg": 2.1,
        "pre_match_away_ppg": 1.0,
        "avg_potential": 2.4,
    }
    market_inputs: dict[str, Any] = {"has_odds": False}

    result = trainer._build_runtime_features_for_diagnostics(
        match=match,
        required_columns=required_columns,
        market_inputs=market_inputs,
        implied_probs=None,
    )

    assert result.get("home_home_ppg") == pytest.approx(1.9)
    assert result.get("away_away_ppg") == pytest.approx(1.3)
    assert result.get("ppg_diff") == pytest.approx(0.6)
    # split uses pre_match_home_ppg / pre_match_away_ppg when available
    assert result.get("split_advantage") == pytest.approx(1.1)
    assert result.get("avg_goals") == pytest.approx(2.4)


def test_build_runtime_features_for_diagnostics_uses_raw_payload_keys() -> None:
    """PPG keys inside the 'raw' sub-dict must also be found by the diagnostics builder."""
    trainer = AutoTrainer()
    required_columns = ["home_home_ppg", "away_away_ppg", "ppg_diff"]
    match: dict[str, Any] = {
        "raw": {
            "home_ppg": 1.5,
            "away_ppg": 1.1,
        }
    }
    market_inputs: dict[str, Any] = {"has_odds": False}

    result = trainer._build_runtime_features_for_diagnostics(
        match=match,
        required_columns=required_columns,
        market_inputs=market_inputs,
        implied_probs=None,
    )

    assert result.get("home_home_ppg") == pytest.approx(1.5)
    assert result.get("away_away_ppg") == pytest.approx(1.1)
    assert result.get("ppg_diff") == pytest.approx(0.4)


def test_build_match_features_with_meta_uses_league_season_stats_for_draw_pct(tmp_path: Any) -> None:
    """_build_match_features_with_meta must prefer league_season_stats for draw_pct/home_advantage."""
    import sqlite3

    db_path = tmp_path / "test.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE league_season_stats (
            season_id INTEGER PRIMARY KEY,
            draw_pct REAL, home_win_pct REAL, away_win_pct REAL,
            home_advantage REAL, season_avg_goals REAL,
            raw_json TEXT DEFAULT '{}', updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    conn.execute(
        "INSERT INTO league_season_stats (season_id, draw_pct, home_advantage, season_avg_goals) VALUES (?, ?, ?, ?)",
        (42, 0.28, 0.12, 2.55),
    )
    conn.execute(
        "CREATE TABLE team_season_stats (team_id INTEGER, season_id INTEGER, season_ppg_overall REAL, season_ppg_home REAL, season_ppg_away REAL, goals_for_avg_overall REAL, goals_for_avg_home REAL, goals_for_avg_away REAL, shots_avg_overall REAL, shots_avg_home REAL, shots_avg_away REAL, possession_avg_overall REAL, possession_avg_home REAL, possession_avg_away REAL, xg_for_avg_overall REAL, xg_for_avg_home REAL, xg_for_avg_away REAL, raw_json TEXT DEFAULT '{}', PRIMARY KEY (team_id, season_id))"
    )
    conn.execute(
        "CREATE TABLE matches (match_id INTEGER PRIMARY KEY, season_id INTEGER, status TEXT, home_team_id INTEGER, away_team_id INTEGER, winning_team_id INTEGER, home_goals INTEGER, away_goals INTEGER)"
    )
    conn.commit()

    class FakeDB:
        pass

    fake_db = FakeDB()
    fake_db.conn = conn  # type: ignore[attr-defined]

    trainer = AutoTrainer()
    trainer.db = fake_db

    required_columns = [
        "odds_ft_1", "odds_ft_x", "odds_ft_2",
        "implied_prob_1", "implied_prob_x", "implied_prob_2",
        "entropy", "gap", "volatility",
        "ppg_diff", "home_home_ppg", "away_away_ppg", "split_advantage",
        "goals_diff", "xg_diff", "shots_diff", "possession_diff",
        "draw_pct", "home_advantage", "avg_goals",
    ]
    match: dict[str, Any] = {
        "odds_ft_1": 2.0,
        "odds_ft_x": 3.2,
        "odds_ft_2": 4.0,
        "season_id": 42,
        "home_team_id": 1,
        "away_team_id": 2,
    }

    features, reason, source_meta = trainer._build_match_features_with_meta(
        match=match,
        required_columns=required_columns,
        pool_probs={"P1": 0.33, "PX": 0.34, "P2": 0.33},
    )

    conn.close()
    assert reason is None
    assert features is not None
    # draw_pct and home_advantage must come from league_season_stats
    assert features["draw_pct"] == pytest.approx(0.28)
    assert features["home_advantage"] == pytest.approx(0.12)
    assert features["avg_goals"] == pytest.approx(2.55)
    assert source_meta.get("draw_pct") == "db_lookup"
    assert source_meta.get("home_advantage") == "db_lookup"
    assert source_meta.get("avg_goals") == "db_lookup"


def test_build_match_features_with_meta_uses_team_stats_for_xg_shots_possession(tmp_path: Any) -> None:
    """DB lookup for xg_diff, shots_diff, possession_diff from team_season_stats."""
    import sqlite3

    db_path = tmp_path / "test2.sqlite3"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE league_season_stats (season_id INTEGER PRIMARY KEY, draw_pct REAL, home_win_pct REAL, away_win_pct REAL, home_advantage REAL, season_avg_goals REAL, raw_json TEXT DEFAULT '{}', updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        """CREATE TABLE team_season_stats (
            team_id INTEGER, season_id INTEGER,
            season_ppg_overall REAL, season_ppg_home REAL, season_ppg_away REAL,
            goals_for_avg_overall REAL, goals_for_avg_home REAL, goals_for_avg_away REAL,
            shots_avg_overall REAL, shots_avg_home REAL, shots_avg_away REAL,
            possession_avg_overall REAL, possession_avg_home REAL, possession_avg_away REAL,
            xg_for_avg_overall REAL, xg_for_avg_home REAL, xg_for_avg_away REAL,
            raw_json TEXT DEFAULT '{}',
            PRIMARY KEY (team_id, season_id)
        )"""
    )
    conn.execute(
        "CREATE TABLE matches (match_id INTEGER PRIMARY KEY, season_id INTEGER, status TEXT, home_team_id INTEGER, away_team_id INTEGER, winning_team_id INTEGER, home_goals INTEGER, away_goals INTEGER)"
    )
    # Home team: xg 0.9, shots_home 12.0, possession_home 55.0
    conn.execute(
        "INSERT INTO team_season_stats (team_id, season_id, xg_for_avg_overall, shots_avg_home, shots_avg_overall, possession_avg_home, possession_avg_overall) VALUES (1, 10, 0.9, 12.0, 11.0, 55.0, 52.0)"
    )
    # Away team: xg 0.6, shots_away 9.0, possession_away 48.0
    conn.execute(
        "INSERT INTO team_season_stats (team_id, season_id, xg_for_avg_overall, shots_avg_away, shots_avg_overall, possession_avg_away, possession_avg_overall) VALUES (2, 10, 0.6, 9.0, 10.0, 48.0, 49.0)"
    )
    conn.commit()

    class FakeDB:
        pass

    fake_db = FakeDB()
    fake_db.conn = conn  # type: ignore[attr-defined]

    trainer = AutoTrainer()
    trainer.db = fake_db

    required_columns = [
        "odds_ft_1", "odds_ft_x", "odds_ft_2",
        "implied_prob_1", "implied_prob_x", "implied_prob_2",
        "entropy", "gap", "volatility",
        "ppg_diff", "home_home_ppg", "away_away_ppg", "split_advantage",
        "goals_diff", "xg_diff", "shots_diff", "possession_diff",
        "draw_pct", "home_advantage", "avg_goals",
    ]
    match: dict[str, Any] = {
        "odds_ft_1": 1.9,
        "odds_ft_x": 3.4,
        "odds_ft_2": 4.5,
        "season_id": 10,
        "home_team_id": 1,
        "away_team_id": 2,
    }

    features, reason, source_meta = trainer._build_match_features_with_meta(
        match=match,
        required_columns=required_columns,
        pool_probs={"P1": 0.4, "PX": 0.3, "P2": 0.3},
    )

    conn.close()
    assert reason is None
    assert features is not None
    assert features["xg_diff"] == pytest.approx(0.3)
    assert features["shots_diff"] == pytest.approx(3.0)   # home: 12.0, away: 9.0
    assert features["possession_diff"] == pytest.approx(7.0)  # home: 55.0, away: 48.0
    assert source_meta.get("xg_diff") == "db_lookup"
    assert source_meta.get("shots_diff") == "db_lookup"
    assert source_meta.get("possession_diff") == "db_lookup"


def test_runtime_feature_snapshot_exposes_quality_gating_fields() -> None:
    trainer = AutoTrainer()
    required_columns = [
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "ppg_diff",
        "split_advantage",
        "goals_diff",
        "xg_diff",
        "shots_diff",
        "possession_diff",
        "draw_pct",
        "home_advantage",
        "avg_goals",
        "entropy",
        "gap",
        "volatility",
    ]
    match = {
        "odds_ft_1": 1.92,
        "odds_ft_x": 3.35,
        "odds_ft_2": 4.25,
        "features": {
            "ppg_diff": 0.55,
            "split_advantage": 0.48,
            "goals_diff": 0.35,
            "xg_diff": 0.29,
            "shots_diff": 2.4,
            "possession_diff": 5.0,
            "draw_pct": 0.27,
            "home_advantage": 0.10,
            "avg_goals": 2.6,
        },
    }

    snapshot = trainer.build_runtime_feature_snapshot(match=match, required_columns=required_columns)

    assert snapshot["status"] == "ok"
    assert snapshot["signal_strength"] in {"medium_signal", "strong_signal"}
    assert snapshot["quality_score"] > 0.45
    assert snapshot["non_market_rich_flag"] is True
    assert 0.0 <= snapshot["market_alignment_score"] <= 1.0


def test_runtime_feature_snapshot_flags_market_vs_stats_disagreement() -> None:
    trainer = AutoTrainer()
    required_columns = [
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "ppg_diff",
        "split_advantage",
        "goals_diff",
        "xg_diff",
        "shots_diff",
        "possession_diff",
        "draw_pct",
        "home_advantage",
        "entropy",
        "gap",
        "volatility",
    ]
    match = {
        "odds_ft_1": 1.50,
        "odds_ft_x": 4.30,
        "odds_ft_2": 6.20,
        "features": {
            "ppg_diff": -1.2,
            "split_advantage": -0.8,
            "goals_diff": -0.7,
            "xg_diff": -0.6,
            "shots_diff": -3.2,
            "possession_diff": -7.5,
            "draw_pct": 0.23,
            "home_advantage": 0.02,
        },
    }

    snapshot = trainer.build_runtime_feature_snapshot(match=match, required_columns=required_columns)

    assert snapshot["status"] == "ok"
    assert snapshot["market_favorite_outcome"] == "P1"
    assert snapshot["stats_context_favorite_outcome"] == "P2"
    assert snapshot["market_disagreement_flag"] is True
    assert snapshot["stats_override_signal"] is True
