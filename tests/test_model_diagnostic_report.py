from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from database.db import Database


def _insert_resolved_rows(db: Database, count: int) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    league_rows = [
        {
            "season_id": 2025,
            "league_name": "League A",
            "country_name": "X",
            "season_label": "2025",
            "chosen_flag": 1,
            "raw_json": "{}",
        },
        {
            "season_id": 2026,
            "league_name": "League B",
            "country_name": "Y",
            "season_label": "2026",
            "chosen_flag": 1,
            "raw_json": "{}",
        },
    ]
    db.conn.executemany(
        """
        INSERT OR REPLACE INTO leagues (
            season_id, league_name, country_name, season_label, chosen_flag, raw_json, updated_at
        ) VALUES (
            :season_id, :league_name, :country_name, :season_label, :chosen_flag, :raw_json, CURRENT_TIMESTAMP
        )
        """,
        league_rows,
    )

    rows = []
    for idx in range(1, count + 1):
        actual = "X" if idx % 3 == 0 else ("1" if idx % 2 == 0 else "2")
        predicted = actual if idx % 5 != 0 else "1"
        rows.append(
            {
                "predicted_at": now_iso,
                "model_version": "model.pkl",
                "model_fingerprint": "fp-main",
                "model_mtime": now_iso,
                "match_id": idx,
                "match_date_iso": "2026-03-31",
                "home_team": f"H{idx}",
                "away_team": f"A{idx}",
                "season_id": 2025 if idx % 2 == 0 else 2026,
                "league_id": 10 if idx % 2 == 0 else 20,
                "p1": 0.4,
                "px": 0.3,
                "p2": 0.3,
                "final_predicted_outcome": predicted,
                "confidence": 0.55,
                "calibrated_confidence": 0.54,
                "feature_context_level": "full_context" if idx % 4 else "degraded_context",
                "signal_strength": "strong_signal" if idx % 3 else "weak_signal",
                "market_disagreement_flag": 1 if idx % 7 == 0 else 0,
                "weak_favorite_flag": 1 if idx % 9 == 0 else 0,
                "draw_risk_flag": 1 if idx % 8 == 0 else 0,
                "stats_override_signal_flag": 1 if idx % 11 == 0 else 0,
                "no_odds_mode": 1 if idx % 13 == 0 else 0,
                "prediction_source": "model_runtime",
                "prediction_status": "predicted",
                "actual_outcome": actual,
                "is_correct": 1 if predicted == actual else 0,
                "resolved_at": now_iso,
                "dedupe_key": f"k-{idx}",
            }
        )

    db.conn.executemany(
        """
        INSERT INTO model_prediction_history (
            predicted_at, model_version, model_fingerprint, model_mtime,
            match_id, match_date_iso, home_team, away_team, season_id, league_id,
            p1, px, p2, final_predicted_outcome,
            confidence, calibrated_confidence,
            feature_context_level, signal_strength,
            market_disagreement_flag, weak_favorite_flag, draw_risk_flag, stats_override_signal_flag,
            no_odds_mode, prediction_source, prediction_status,
            actual_outcome, is_correct, resolved_at, dedupe_key
        ) VALUES (
            :predicted_at, :model_version, :model_fingerprint, :model_mtime,
            :match_id, :match_date_iso, :home_team, :away_team, :season_id, :league_id,
            :p1, :px, :p2, :final_predicted_outcome,
            :confidence, :calibrated_confidence,
            :feature_context_level, :signal_strength,
            :market_disagreement_flag, :weak_favorite_flag, :draw_risk_flag, :stats_override_signal_flag,
            :no_odds_mode, :prediction_source, :prediction_status,
            :actual_outcome, :is_correct, :resolved_at, :dedupe_key
        )
        """,
        rows,
    )
    db.conn.commit()


def test_model_diagnostic_report_threshold_gate(tmp_path: Path) -> None:
    db_path = tmp_path / "footai_test.sqlite3"
    db = Database(db_path=db_path)
    try:
        db.initialize("database/schema.sql")
        _insert_resolved_rows(db, 120)

        result = db.write_model_diagnostic_report(report_dir=tmp_path / "reports", min_resolved=300)

        assert result["generated"] is False
        assert "минимум 300" in str(result["reason"])
        assert result["json_path"] is None
        assert result["md_path"] is None
    finally:
        db.close()


def test_model_diagnostic_report_files_are_generated(tmp_path: Path) -> None:
    db_path = tmp_path / "footai_test.sqlite3"
    db = Database(db_path=db_path)
    try:
        db.initialize("database/schema.sql")
        _insert_resolved_rows(db, 320)

        result = db.write_model_diagnostic_report(report_dir=tmp_path / "reports", min_resolved=300)

        assert result["generated"] is True
        json_path = Path(str(result["json_path"]))
        md_path = Path(str(result["md_path"]))
        assert json_path.exists()
        assert md_path.exists()

        report = result["report"]
        assert report["metadata"]["report_type"] == "model_diagnostic_report"
        assert int(report["metadata"]["resolved_predictions"]) >= 300
        assert "league_errors" in report

        md_content = md_path.read_text(encoding="utf-8")
        assert "Это отчет по реальным историческим прогнозам модели" in md_content
        assert "Это не backtest и не smoke" in md_content
    finally:
        db.close()
