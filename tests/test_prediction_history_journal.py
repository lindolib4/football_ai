from __future__ import annotations

from pathlib import Path

from database.db import Database


def test_prediction_history_save_resolve_and_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "footai_test.sqlite3"
    db = Database(db_path=db_path)
    try:
        db.initialize("database/schema.sql")

        db.conn.execute(
            """
            INSERT INTO matches (
                match_id, season_id, match_date_iso, status,
                home_team_id, away_team_id, home_team_name, away_team_name,
                home_goals, away_goals, winning_team_id,
                raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                101,
                2025,
                "2026-03-30T18:00:00",
                "completed",
                1,
                2,
                "Home FC",
                "Away FC",
                1,
                1,
                None,
                "{}",
            ),
        )
        db.conn.commit()

        row = {
            "predicted_at": "2026-03-30T10:00:00Z",
            "model_version": "model.pkl",
            "model_fingerprint": "abc123",
            "model_mtime": "2026-03-30T09:00:00",
            "match_id": 101,
            "match_date_iso": "2026-03-30T18:00:00",
            "home_team": "Home FC",
            "away_team": "Away FC",
            "season_id": 2025,
            "league_id": 55,
            "p1": 0.35,
            "px": 0.40,
            "p2": 0.25,
            "final_predicted_outcome": "X",
            "confidence": 0.40,
            "calibrated_confidence": 0.39,
            "feature_context_level": "full_context",
            "signal_strength": "strong_signal",
            "market_disagreement_flag": 1,
            "weak_favorite_flag": 0,
            "draw_risk_flag": 1,
            "no_odds_mode": 0,
            "prediction_source": "model_runtime",
            "prediction_status": "predicted",
            "dedupe_key": "k1",
        }

        inserted_1 = db.save_model_prediction_history_rows([row])
        inserted_2 = db.save_model_prediction_history_rows([row])

        assert inserted_1 == 1
        assert inserted_2 == 0

        resolve_stats = db.resolve_model_prediction_history(limit=100)
        assert resolve_stats["resolved"] == 1

        summary = db.get_model_prediction_history_summary()
        assert summary["total_predictions"] == 1
        assert summary["resolved_predictions"] == 1
        assert summary["unresolved_predictions"] == 0
        assert summary["overall_hit_rate"] == 1.0
        assert summary["hit_rate_by_actual_outcome"]["X"]["count"] == 1
        assert summary["hit_rate_by_actual_outcome"]["X"]["hit_rate"] == 1.0
        assert summary["market_disagreement_stats"]["disagreement"]["count"] == 1
    finally:
        db.close()
