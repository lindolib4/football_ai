from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger("database")


class Database:
    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or settings.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.last_training_dataset_debug: dict[str, Any] = {}

    def initialize(self, schema_path: str = "database/schema.sql") -> None:
        schema = Path(schema_path).read_text(encoding="utf-8")
        self.conn.executescript(schema)
        self.conn.commit()
        logger.info("schema initialized path=%s", self.db_path)

    def _ensure_model_prediction_history_table(self) -> None:
        """Create prediction history table lazily for existing DBs without migrations."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_prediction_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_at TEXT NOT NULL,
                model_version TEXT,
                model_fingerprint TEXT,
                model_mtime TEXT,
                match_id INTEGER NOT NULL,
                match_date_iso TEXT,
                home_team TEXT,
                away_team TEXT,
                season_id INTEGER,
                league_id INTEGER,
                p1 REAL NOT NULL,
                px REAL NOT NULL,
                p2 REAL NOT NULL,
                final_predicted_outcome TEXT NOT NULL,
                confidence REAL,
                calibrated_confidence REAL,
                feature_context_level TEXT,
                signal_strength TEXT,
                market_disagreement_flag INTEGER DEFAULT 0,
                weak_favorite_flag INTEGER DEFAULT 0,
                draw_risk_flag INTEGER DEFAULT 0,
                stats_override_signal_flag INTEGER DEFAULT 0,
                no_odds_mode INTEGER DEFAULT 0,
                prediction_source TEXT,
                prediction_status TEXT,
                actual_outcome TEXT,
                is_correct INTEGER,
                resolved_at TEXT,
                dedupe_key TEXT UNIQUE
            );

            CREATE INDEX IF NOT EXISTS idx_model_prediction_history_match_id
                ON model_prediction_history(match_id);

            CREATE INDEX IF NOT EXISTS idx_model_prediction_history_predicted_at
                ON model_prediction_history(predicted_at);

            CREATE INDEX IF NOT EXISTS idx_model_prediction_history_unresolved
                ON model_prediction_history(actual_outcome, predicted_at);

            CREATE INDEX IF NOT EXISTS idx_model_prediction_history_league_resolved
                ON model_prediction_history(league_id, actual_outcome);
            """
        )

        columns = {
            str(row["name"]).strip().lower()
            for row in self.conn.execute("PRAGMA table_info(model_prediction_history)").fetchall()
        }
        if "stats_override_signal_flag" not in columns:
            self.conn.execute(
                "ALTER TABLE model_prediction_history ADD COLUMN stats_override_signal_flag INTEGER DEFAULT 0"
            )
        self.conn.commit()

    def save_model_prediction_history_rows(self, rows: list[dict[str, Any]]) -> int:
        """Persist compact prediction rows in one batch with dedupe protection."""
        if not rows:
            return 0

        self._ensure_model_prediction_history_table()

        cleaned_rows: list[dict[str, Any]] = []
        for row in rows:
            match_id = self._to_int_or_none(row.get("match_id"))
            if match_id is None:
                continue

            p1 = self._to_float_or_none(row.get("p1"))
            px = self._to_float_or_none(row.get("px"))
            p2 = self._to_float_or_none(row.get("p2"))
            if p1 is None or px is None or p2 is None:
                continue

            outcome = str(row.get("final_predicted_outcome") or "").strip().upper()
            if outcome not in {"1", "X", "2"}:
                continue

            cleaned_rows.append(
                {
                    "predicted_at": str(row.get("predicted_at") or datetime.now(timezone.utc).isoformat()),
                    "model_version": row.get("model_version"),
                    "model_fingerprint": row.get("model_fingerprint"),
                    "model_mtime": row.get("model_mtime"),
                    "match_id": match_id,
                    "match_date_iso": row.get("match_date_iso"),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "season_id": row.get("season_id"),
                    "league_id": row.get("league_id"),
                    "p1": float(p1),
                    "px": float(px),
                    "p2": float(p2),
                    "final_predicted_outcome": outcome,
                    "confidence": self._to_float_or_none(row.get("confidence")),
                    "calibrated_confidence": self._to_float_or_none(row.get("calibrated_confidence")),
                    "feature_context_level": row.get("feature_context_level"),
                    "signal_strength": row.get("signal_strength"),
                    "market_disagreement_flag": int(bool(row.get("market_disagreement_flag"))),
                    "weak_favorite_flag": int(bool(row.get("weak_favorite_flag"))),
                    "draw_risk_flag": int(bool(row.get("draw_risk_flag"))),
                    "stats_override_signal_flag": int(bool(row.get("stats_override_signal_flag"))),
                    "no_odds_mode": int(bool(row.get("no_odds_mode"))),
                    "prediction_source": row.get("prediction_source"),
                    "prediction_status": row.get("prediction_status"),
                    "dedupe_key": row.get("dedupe_key"),
                }
            )

        if not cleaned_rows:
            return 0

        before_changes = int(self.conn.total_changes)
        self.conn.executemany(
            """
            INSERT OR IGNORE INTO model_prediction_history (
                predicted_at, model_version, model_fingerprint, model_mtime,
                match_id, match_date_iso, home_team, away_team, season_id, league_id,
                p1, px, p2, final_predicted_outcome,
                confidence, calibrated_confidence,
                feature_context_level, signal_strength,
                market_disagreement_flag, weak_favorite_flag, draw_risk_flag,
                stats_override_signal_flag,
                no_odds_mode, prediction_source, prediction_status, dedupe_key
            ) VALUES (
                :predicted_at, :model_version, :model_fingerprint, :model_mtime,
                :match_id, :match_date_iso, :home_team, :away_team, :season_id, :league_id,
                :p1, :px, :p2, :final_predicted_outcome,
                :confidence, :calibrated_confidence,
                :feature_context_level, :signal_strength,
                :market_disagreement_flag, :weak_favorite_flag, :draw_risk_flag,
                :stats_override_signal_flag,
                :no_odds_mode, :prediction_source, :prediction_status, :dedupe_key
            )
            """,
            cleaned_rows,
        )
        self.conn.commit()
        inserted = int(self.conn.total_changes) - before_changes
        logger.info(
            "prediction_journal_saved requested=%s inserted=%s ignored_duplicates=%s",
            len(cleaned_rows),
            inserted,
            max(len(cleaned_rows) - inserted, 0),
        )
        return max(inserted, 0)

    def resolve_model_prediction_history(self, limit: int = 1000) -> dict[str, int]:
        """Resolve unresolved predictions against completed match outcomes."""
        self._ensure_model_prediction_history_table()

        unresolved_rows = self.conn.execute(
            """
            SELECT h.history_id, h.match_id, h.final_predicted_outcome,
                   m.home_team_id, m.away_team_id, m.winning_team_id,
                   m.home_goals, m.away_goals, m.status
            FROM model_prediction_history h
            JOIN matches m ON m.match_id = h.match_id
            WHERE h.actual_outcome IS NULL
              AND m.status = 'completed'
            ORDER BY h.predicted_at ASC
            LIMIT ?
            """,
            (max(1, int(limit)),),
        ).fetchall()

        resolved = 0
        unresolved = 0
        updates: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for row in unresolved_rows:
            match = dict(row)
            actual_outcome: str | None = None
            winning_team_id = match.get("winning_team_id")
            home_team_id = match.get("home_team_id")
            away_team_id = match.get("away_team_id")
            home_goals = match.get("home_goals")
            away_goals = match.get("away_goals")

            if winning_team_id is not None:
                if winning_team_id == home_team_id:
                    actual_outcome = "1"
                elif winning_team_id == away_team_id:
                    actual_outcome = "2"
            if actual_outcome is None and home_goals is not None and away_goals is not None:
                if int(home_goals) == int(away_goals):
                    actual_outcome = "X"

            if actual_outcome is None:
                unresolved += 1
                continue

            predicted = str(match.get("final_predicted_outcome") or "").strip().upper()
            is_correct = int(predicted == actual_outcome)
            updates.append(
                {
                    "history_id": int(match["history_id"]),
                    "actual_outcome": actual_outcome,
                    "is_correct": is_correct,
                    "resolved_at": now_iso,
                }
            )
            resolved += 1

        if updates:
            self.conn.executemany(
                """
                UPDATE model_prediction_history
                SET actual_outcome = :actual_outcome,
                    is_correct = :is_correct,
                    resolved_at = :resolved_at
                WHERE history_id = :history_id
                  AND actual_outcome IS NULL
                """,
                updates,
            )
            self.conn.commit()

        return {
            "checked": len(unresolved_rows),
            "resolved": resolved,
            "still_unresolved": unresolved,
        }

    def get_model_prediction_history_summary(self) -> dict[str, Any]:
        """Return lightweight quality summary for prediction journal analytics."""
        self._ensure_model_prediction_history_table()

        totals = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved_predictions,
                SUM(CASE WHEN actual_outcome IS NULL THEN 1 ELSE 0 END) AS unresolved_predictions,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS correct_predictions
            FROM model_prediction_history
            """
        ).fetchone()

        def _safe_rate(num: int, den: int) -> float | None:
            if den <= 0:
                return None
            return round(float(num) / float(den), 4)

        total_predictions = int((totals["total_predictions"] or 0) if totals is not None else 0)
        resolved_predictions = int((totals["resolved_predictions"] or 0) if totals is not None else 0)
        unresolved_predictions = int((totals["unresolved_predictions"] or 0) if totals is not None else 0)
        correct_predictions = int((totals["correct_predictions"] or 0) if totals is not None else 0)

        by_outcome_rows = self.conn.execute(
            """
            SELECT actual_outcome,
                   COUNT(*) AS n,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY actual_outcome
            """
        ).fetchall()
        by_outcome = {
            "1": {"count": 0, "hit_rate": None},
            "X": {"count": 0, "hit_rate": None},
            "2": {"count": 0, "hit_rate": None},
        }
        for row in by_outcome_rows:
            outcome = str(row["actual_outcome"])
            if outcome not in by_outcome:
                continue
            count = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            by_outcome[outcome] = {
                "count": count,
                "hit_rate": _safe_rate(hits, count),
            }

        by_context_rows = self.conn.execute(
            """
            SELECT
                CASE
                    WHEN no_odds_mode = 1 THEN 'no_odds_mode'
                    WHEN feature_context_level IN ('full_context','partial_context') THEN feature_context_level
                    ELSE COALESCE(feature_context_level, 'unknown')
                END AS context_group,
                COUNT(*) AS n,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY context_group
            """
        ).fetchall()
        by_context: dict[str, dict[str, Any]] = {}
        for row in by_context_rows:
            key = str(row["context_group"])
            n = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            by_context[key] = {"count": n, "hit_rate": _safe_rate(hits, n)}

        by_signal_rows = self.conn.execute(
            """
            SELECT COALESCE(signal_strength, 'unknown') AS signal_strength,
                   COUNT(*) AS n,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY COALESCE(signal_strength, 'unknown')
            """
        ).fetchall()
        by_signal: dict[str, dict[str, Any]] = {}
        for row in by_signal_rows:
            key = str(row["signal_strength"])
            n = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            by_signal[key] = {"count": n, "hit_rate": _safe_rate(hits, n)}

        market_rows = self.conn.execute(
            """
            SELECT
                market_disagreement_flag,
                COUNT(*) AS n,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY market_disagreement_flag
            """
        ).fetchall()
        market_disagreement_stats = {
            "agreement": {"count": 0, "hit_rate": None},
            "disagreement": {"count": 0, "hit_rate": None},
        }
        for row in market_rows:
            flag = int(row["market_disagreement_flag"] or 0)
            n = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            key = "disagreement" if flag == 1 else "agreement"
            market_disagreement_stats[key] = {"count": n, "hit_rate": _safe_rate(hits, n)}

        flag_error_row = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN actual_outcome = 'X' AND is_correct = 0 THEN 1 ELSE 0 END) AS x_errors,
                SUM(CASE WHEN weak_favorite_flag = 1 AND is_correct = 0 THEN 1 ELSE 0 END) AS weak_favorite_errors,
                SUM(CASE WHEN draw_risk_flag = 1 AND is_correct = 0 THEN 1 ELSE 0 END) AS draw_risk_errors
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            """
        ).fetchone()

        return {
            "total_predictions": total_predictions,
            "resolved_predictions": resolved_predictions,
            "unresolved_predictions": unresolved_predictions,
            "overall_hit_rate": _safe_rate(correct_predictions, resolved_predictions),
            "hit_rate_by_actual_outcome": by_outcome,
            "hit_rate_by_context": by_context,
            "hit_rate_by_signal_strength": by_signal,
            "market_disagreement_stats": market_disagreement_stats,
            "error_focus": {
                "x_errors": int((flag_error_row["x_errors"] or 0) if flag_error_row is not None else 0),
                "weak_favorite_errors": int((flag_error_row["weak_favorite_errors"] or 0) if flag_error_row is not None else 0),
                "draw_risk_errors": int((flag_error_row["draw_risk_errors"] or 0) if flag_error_row is not None else 0),
            },
        }

    def _diagnostic_reliability_level(self, resolved_predictions: int) -> str:
        if resolved_predictions >= 1000:
            return "high"
        if resolved_predictions >= 500:
            return "medium"
        return "low"

    def get_model_diagnostic_report(self, min_resolved: int = 300, league_min_count: int = 20) -> dict[str, Any]:
        """Build a structured diagnostic report from resolved prediction journal rows."""
        self._ensure_model_prediction_history_table()

        totals_row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS total_predictions,
                SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) AS resolved_predictions,
                SUM(CASE WHEN actual_outcome IS NULL THEN 1 ELSE 0 END) AS unresolved_predictions,
                SUM(CASE WHEN actual_outcome IS NOT NULL AND is_correct = 1 THEN 1 ELSE 0 END) AS correct_predictions,
                AVG(CASE WHEN actual_outcome IS NOT NULL THEN confidence END) AS avg_confidence,
                AVG(CASE WHEN actual_outcome IS NOT NULL THEN calibrated_confidence END) AS avg_calibrated_confidence,
                MIN(CASE WHEN actual_outcome IS NOT NULL THEN predicted_at END) AS analyzed_period_start,
                MAX(CASE WHEN actual_outcome IS NOT NULL THEN predicted_at END) AS analyzed_period_end
            FROM model_prediction_history
            """
        ).fetchone()

        total_predictions = int((totals_row["total_predictions"] or 0) if totals_row is not None else 0)
        resolved_predictions = int((totals_row["resolved_predictions"] or 0) if totals_row is not None else 0)
        unresolved_predictions = int((totals_row["unresolved_predictions"] or 0) if totals_row is not None else 0)
        correct_predictions = int((totals_row["correct_predictions"] or 0) if totals_row is not None else 0)

        overall_hit_rate = (float(correct_predictions) / float(resolved_predictions)) if resolved_predictions > 0 else None
        reliability_level = self._diagnostic_reliability_level(resolved_predictions)

        model_row = self.conn.execute(
            """
            SELECT
                COALESCE(model_version, 'unknown') AS model_version,
                COALESCE(model_fingerprint, 'unknown') AS model_fingerprint,
                COUNT(*) AS n
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY COALESCE(model_version, 'unknown'), COALESCE(model_fingerprint, 'unknown')
            ORDER BY n DESC
            LIMIT 1
            """
        ).fetchone()

        model_version = str(model_row["model_version"]) if model_row is not None else "unknown"
        model_fingerprint = str(model_row["model_fingerprint"]) if model_row is not None else "unknown"

        predicted_dist_rows = self.conn.execute(
            """
            SELECT final_predicted_outcome AS outcome, COUNT(*) AS n
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY final_predicted_outcome
            """
        ).fetchall()
        actual_dist_rows = self.conn.execute(
            """
            SELECT actual_outcome AS outcome, COUNT(*) AS n
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY actual_outcome
            """
        ).fetchall()

        predicted_distribution = {"1": 0, "X": 0, "2": 0}
        actual_distribution = {"1": 0, "X": 0, "2": 0}
        for row in predicted_dist_rows:
            key = str(row["outcome"] or "")
            if key in predicted_distribution:
                predicted_distribution[key] = int(row["n"] or 0)
        for row in actual_dist_rows:
            key = str(row["outcome"] or "")
            if key in actual_distribution:
                actual_distribution[key] = int(row["n"] or 0)

        by_actual_rows = self.conn.execute(
            """
            SELECT
                actual_outcome,
                COUNT(*) AS n,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY actual_outcome
            """
        ).fetchall()
        hit_rate_by_actual = {
            "1": {"count": 0, "hit_rate": None},
            "X": {"count": 0, "hit_rate": None},
            "2": {"count": 0, "hit_rate": None},
        }
        for row in by_actual_rows:
            key = str(row["actual_outcome"] or "")
            if key not in hit_rate_by_actual:
                continue
            n = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            hit_rate_by_actual[key] = {
                "count": n,
                "hit_rate": (float(hits) / float(n)) if n > 0 else None,
            }

        confusion_rows = self.conn.execute(
            """
            SELECT
                actual_outcome,
                final_predicted_outcome,
                COUNT(*) AS n
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY actual_outcome, final_predicted_outcome
            """
        ).fetchall()
        confusion_matrix = {
            "1": {"1": 0, "X": 0, "2": 0},
            "X": {"1": 0, "X": 0, "2": 0},
            "2": {"1": 0, "X": 0, "2": 0},
        }
        top_confusions: list[dict[str, Any]] = []
        for row in confusion_rows:
            actual = str(row["actual_outcome"] or "")
            predicted = str(row["final_predicted_outcome"] or "")
            n = int(row["n"] or 0)
            if actual in confusion_matrix and predicted in confusion_matrix[actual]:
                confusion_matrix[actual][predicted] = n
                if actual != predicted and n > 0:
                    top_confusions.append({"from_actual": actual, "to_predicted": predicted, "count": n})
        top_confusions.sort(key=lambda item: int(item["count"]), reverse=True)

        context_rows = self.conn.execute(
            """
            SELECT
                CASE
                    WHEN no_odds_mode = 1 THEN 'no_odds_mode'
                    WHEN feature_context_level = 'full_context' THEN 'full_context'
                    WHEN feature_context_level = 'partial_context' THEN 'partial_context'
                    WHEN feature_context_level = 'degraded_context' THEN 'degraded_context'
                    ELSE COALESCE(feature_context_level, 'unknown')
                END AS context_group,
                COUNT(*) AS n,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY context_group
            """
        ).fetchall()
        context_errors: dict[str, dict[str, Any]] = {}
        for row in context_rows:
            key = str(row["context_group"] or "unknown")
            n = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            context_errors[key] = {
                "count": n,
                "hit_rate": (float(hits) / float(n)) if n > 0 else None,
                "errors": max(n - hits, 0),
            }

        signal_rows = self.conn.execute(
            """
            SELECT
                COALESCE(signal_strength, 'unknown') AS signal_strength,
                COUNT(*) AS n,
                SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history
            WHERE actual_outcome IS NOT NULL
            GROUP BY COALESCE(signal_strength, 'unknown')
            """
        ).fetchall()
        signal_errors: dict[str, dict[str, Any]] = {}
        for row in signal_rows:
            key = str(row["signal_strength"] or "unknown")
            n = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            signal_errors[key] = {
                "count": n,
                "hit_rate": (float(hits) / float(n)) if n > 0 else None,
                "errors": max(n - hits, 0),
            }

        def _flag_metric(flag_column: str) -> dict[str, Any]:
            row = self.conn.execute(
                f"""
                SELECT
                    SUM(CASE WHEN {flag_column} = 1 THEN 1 ELSE 0 END) AS flagged_count,
                    SUM(CASE WHEN {flag_column} = 1 AND is_correct = 1 THEN 1 ELSE 0 END) AS flagged_hits,
                    SUM(CASE WHEN {flag_column} = 0 THEN 1 ELSE 0 END) AS unflagged_count,
                    SUM(CASE WHEN {flag_column} = 0 AND is_correct = 1 THEN 1 ELSE 0 END) AS unflagged_hits
                FROM model_prediction_history
                WHERE actual_outcome IS NOT NULL
                """
            ).fetchone()
            flagged_count = int((row["flagged_count"] or 0) if row is not None else 0)
            flagged_hits = int((row["flagged_hits"] or 0) if row is not None else 0)
            unflagged_count = int((row["unflagged_count"] or 0) if row is not None else 0)
            unflagged_hits = int((row["unflagged_hits"] or 0) if row is not None else 0)
            return {
                "flagged_count": flagged_count,
                "flagged_hit_rate": (float(flagged_hits) / float(flagged_count)) if flagged_count > 0 else None,
                "flagged_errors": max(flagged_count - flagged_hits, 0),
                "unflagged_count": unflagged_count,
                "unflagged_hit_rate": (float(unflagged_hits) / float(unflagged_count)) if unflagged_count > 0 else None,
            }

        flag_errors = {
            "weak_favorite": _flag_metric("weak_favorite_flag"),
            "draw_risk": _flag_metric("draw_risk_flag"),
            "market_disagreement": _flag_metric("market_disagreement_flag"),
            "stats_override_signal": _flag_metric("stats_override_signal_flag"),
        }

        league_rows = self.conn.execute(
            """
            SELECT
                COALESCE(l.league_name, 'league_' || COALESCE(CAST(h.league_id AS TEXT), 'unknown')) AS league_name,
                COUNT(*) AS n,
                SUM(CASE WHEN h.is_correct = 1 THEN 1 ELSE 0 END) AS hits
            FROM model_prediction_history h
            LEFT JOIN leagues l ON l.season_id = h.season_id
            WHERE h.actual_outcome IS NOT NULL
            GROUP BY COALESCE(l.league_name, 'league_' || COALESCE(CAST(h.league_id AS TEXT), 'unknown'))
            """
        ).fetchall()
        leagues: list[dict[str, Any]] = []
        for row in league_rows:
            n = int(row["n"] or 0)
            hits = int(row["hits"] or 0)
            hit_rate = (float(hits) / float(n)) if n > 0 else None
            leagues.append(
                {
                    "league": str(row["league_name"] or "unknown"),
                    "count": n,
                    "hit_rate": hit_rate,
                    "errors": max(n - hits, 0),
                }
            )

        leagues_by_count = sorted(leagues, key=lambda item: int(item["count"]), reverse=True)
        top_leagues = leagues_by_count[:10]
        worst_leagues = sorted(
            [row for row in leagues if int(row["count"]) >= max(league_min_count, 1)],
            key=lambda item: float(item["hit_rate"] if item["hit_rate"] is not None else 1.0),
        )[:10]
        low_data_leagues = [row for row in leagues_by_count if int(row["count"]) < max(league_min_count, 1)][:10]

        overall_error_rate = (1.0 - float(overall_hit_rate)) if overall_hit_rate is not None else None
        suspicious_error_concentration: list[dict[str, Any]] = []
        if overall_error_rate is not None:
            for row in leagues:
                if int(row["count"]) < max(league_min_count, 20):
                    continue
                hit_rate_value = row.get("hit_rate")
                if hit_rate_value is None:
                    continue
                error_rate = 1.0 - float(hit_rate_value)
                if error_rate >= (overall_error_rate + 0.15):
                    suspicious_error_concentration.append(
                        {
                            "league": row["league"],
                            "count": row["count"],
                            "hit_rate": hit_rate_value,
                            "error_rate": error_rate,
                        }
                    )
        suspicious_error_concentration.sort(key=lambda item: float(item["error_rate"]), reverse=True)

        strong_sides: list[str] = []
        weak_sides: list[str] = []
        action_segments: list[str] = []
        keep_as_is: list[str] = []
        recommendations: list[str] = []

        if overall_hit_rate is not None and overall_hit_rate >= 0.56:
            strong_sides.append("Overall hit rate is stable on resolved predictions.")
            keep_as_is.append("Current core predictor behavior on major classes.")
        elif overall_hit_rate is not None:
            weak_sides.append("Overall hit rate is below target and requires focused improvements.")
            recommendations.append("retrain")

        x_hit = hit_rate_by_actual.get("X", {}).get("hit_rate")
        if isinstance(x_hit, float) and x_hit < 0.45:
            weak_sides.append("Draw (X) segment underperforms relative to other outcomes.")
            action_segments.append("Improve draw sensitivity and calibration around draw-heavy matches.")
            recommendations.append("calibration")

        degraded_hit = context_errors.get("degraded_context", {}).get("hit_rate")
        full_hit = context_errors.get("full_context", {}).get("hit_rate")
        if isinstance(degraded_hit, float) and isinstance(full_hit, float) and degraded_hit + 0.08 < full_hit:
            weak_sides.append("Degraded context has materially lower quality than full context.")
            action_segments.append("Improve feature delivery quality to reduce degraded-context share.")
            recommendations.append("improve feature delivery")

        weak_flag_rate = flag_errors.get("weak_favorite", {}).get("flagged_hit_rate")
        if isinstance(weak_flag_rate, float) and overall_hit_rate is not None and weak_flag_rate + 0.08 < overall_hit_rate:
            weak_sides.append("Weak-favorite scenarios are a consistent error hotspot.")
            action_segments.append("Tune handling of weak favorites and market disagreement edge cases.")

        if worst_leagues:
            action_segments.append("Inspect league-specific data quality for worst leagues by hit rate.")
            recommendations.append("improve feature delivery")

        if not recommendations:
            recommendations.append("keep as is")
        recommendations = sorted(set(recommendations), key=recommendations.index)

        can_generate = resolved_predictions >= max(1, int(min_resolved))
        report = {
            "metadata": {
                "report_generated_at": datetime.now(timezone.utc).isoformat(),
                "report_type": "model_diagnostic_report",
                "model_version": model_version,
                "model_fingerprint": model_fingerprint,
                "total_predictions": total_predictions,
                "resolved_predictions": resolved_predictions,
                "unresolved_predictions": unresolved_predictions,
                "analyzed_period": {
                    "start": totals_row["analyzed_period_start"] if totals_row is not None else None,
                    "end": totals_row["analyzed_period_end"] if totals_row is not None else None,
                },
                "reliability_level": reliability_level,
                "thresholds": {
                    "min_resolved_required": int(min_resolved),
                    "normal_reliability_from": 500,
                    "high_reliability_from": 1000,
                },
                "can_generate_report": can_generate,
            },
            "why_this_report": {
                "description": "This report analyzes real historical model predictions resolved with actual outcomes.",
                "safety_goal": "Use it to improve the model safely without changing strong segments blindly.",
                "not_backtest": True,
                "not_smoke": True,
                "usage_note": "Do not use this report as direct punitive auto-learning from mistakes.",
            },
            "overall_summary": {
                "overall_hit_rate": overall_hit_rate,
                "total_resolved": resolved_predictions,
                "average_confidence": float(totals_row["avg_confidence"]) if totals_row is not None and totals_row["avg_confidence"] is not None else None,
                "average_calibrated_confidence": float(totals_row["avg_calibrated_confidence"]) if totals_row is not None and totals_row["avg_calibrated_confidence"] is not None else None,
                "predicted_outcome_distribution": predicted_distribution,
                "actual_outcome_distribution": actual_distribution,
            },
            "outcome_errors": {
                "hit_rate_by_actual": hit_rate_by_actual,
                "confusion_matrix": confusion_matrix,
                "top_confusions": top_confusions[:12],
            },
            "context_errors": context_errors,
            "signal_strength_errors": signal_errors,
            "flag_errors": flag_errors,
            "league_errors": {
                "top_leagues_by_prediction_count": top_leagues,
                "worst_leagues_by_hit_rate": worst_leagues,
                "leagues_with_too_little_data": low_data_leagues,
                "leagues_with_suspicious_error_concentration": suspicious_error_concentration[:10],
                "league_min_count": int(league_min_count),
            },
            "main_conclusions": {
                "strong_sides": strong_sides,
                "weak_sides": weak_sides,
                "segments_to_improve": action_segments,
                "what_not_to_touch_yet": keep_as_is,
                "recommendations": recommendations,
            },
        }

        if not can_generate:
            report["availability"] = {
                "available": False,
                "reason": (
                    "Недостаточно завершенных прогнозов для надежного отчета. "
                    f"Нужно минимум {int(min_resolved)}."
                ),
            }
        else:
            report["availability"] = {"available": True, "reason": "ok"}
        return report

    def _render_model_diagnostic_markdown(self, report: dict[str, Any]) -> str:
        metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
        summary = report.get("overall_summary", {}) if isinstance(report.get("overall_summary"), dict) else {}
        outcome_errors = report.get("outcome_errors", {}) if isinstance(report.get("outcome_errors"), dict) else {}
        context_errors = report.get("context_errors", {}) if isinstance(report.get("context_errors"), dict) else {}
        signal_errors = report.get("signal_strength_errors", {}) if isinstance(report.get("signal_strength_errors"), dict) else {}
        flag_errors = report.get("flag_errors", {}) if isinstance(report.get("flag_errors"), dict) else {}
        league_errors = report.get("league_errors", {}) if isinstance(report.get("league_errors"), dict) else {}
        conclusions = report.get("main_conclusions", {}) if isinstance(report.get("main_conclusions"), dict) else {}

        lines: list[str] = [
            "# Model Diagnostic Report",
            "",
            "## Что это за отчет",
            "- Это отчет по реальным историческим прогнозам модели, которые уже связаны с фактическим исходом.",
            "- Нужен для безопасного улучшения модели: выявить слабые и сильные сегменты до изменений.",
            "- Это не backtest и не smoke.",
            "- Чтение: сначала overall, затем outcome/context/flags/leagues, в конце рекомендации.",
            "",
            "## METADATA",
            f"- report_generated_at: {metadata.get('report_generated_at')}",
            f"- report_type: {metadata.get('report_type')}",
            f"- model_version: {metadata.get('model_version')}",
            f"- model_fingerprint: {metadata.get('model_fingerprint')}",
            f"- total_predictions: {metadata.get('total_predictions')}",
            f"- resolved_predictions: {metadata.get('resolved_predictions')}",
            f"- unresolved_predictions: {metadata.get('unresolved_predictions')}",
            f"- analyzed_period: {((metadata.get('analyzed_period') or {}).get('start') if isinstance(metadata.get('analyzed_period'), dict) else None)} -> {((metadata.get('analyzed_period') or {}).get('end') if isinstance(metadata.get('analyzed_period'), dict) else None)}",
            f"- reliability_level: {metadata.get('reliability_level')}",
            "",
            "## Общая сводка",
            f"- overall hit rate: {summary.get('overall_hit_rate')}",
            f"- total resolved: {summary.get('total_resolved')}",
            f"- average confidence: {summary.get('average_confidence')}",
            f"- average calibrated confidence: {summary.get('average_calibrated_confidence')}",
            f"- predicted outcome distribution: {summary.get('predicted_outcome_distribution')}",
            f"- actual outcome distribution: {summary.get('actual_outcome_distribution')}",
            "",
            "## Ошибки по исходам",
            f"- hit rate by actual: {outcome_errors.get('hit_rate_by_actual')}",
            f"- confusion matrix: {outcome_errors.get('confusion_matrix')}",
            "- top confusions:",
        ]
        top_confusions = outcome_errors.get("top_confusions", []) if isinstance(outcome_errors.get("top_confusions"), list) else []
        if top_confusions:
            for row in top_confusions[:12]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"  - {row.get('from_actual')} -> {row.get('to_predicted')}: {row.get('count')}"
                )
        else:
            lines.append("  - no major confusions")

        lines.extend(
            [
                "",
                "## Ошибки по контексту",
                f"- {context_errors}",
                "",
                "## Ошибки по силе сигнала",
                f"- {signal_errors}",
                "",
                "## Ошибки по флагам",
                f"- {flag_errors}",
                "",
                "## Ошибки по лигам",
                f"- top leagues by prediction count: {league_errors.get('top_leagues_by_prediction_count')}",
                f"- worst leagues by hit rate: {league_errors.get('worst_leagues_by_hit_rate')}",
                f"- leagues with too little data: {league_errors.get('leagues_with_too_little_data')}",
                f"- leagues with suspicious error concentration: {league_errors.get('leagues_with_suspicious_error_concentration')}",
                "",
                "## Главные выводы",
                f"- сильные стороны: {conclusions.get('strong_sides')}",
                f"- слабые стороны: {conclusions.get('weak_sides')}",
                f"- сегменты для улучшения: {conclusions.get('segments_to_improve')}",
                f"- что пока не трогать: {conclusions.get('what_not_to_touch_yet')}",
                f"- рекомендации: {conclusions.get('recommendations')}",
                "",
                "## Важно",
                "- Этот отчет предназначен для анализа слабых мест модели перед следующими улучшениями.",
                "- Не используйте его как прямой источник грубого штрафа модели за ошибки.",
            ]
        )
        return "\n".join(lines)

    def write_model_diagnostic_report(
        self,
        report_dir: str | Path = "reports",
        min_resolved: int = 300,
        league_min_count: int = 20,
    ) -> dict[str, Any]:
        report = self.get_model_diagnostic_report(min_resolved=min_resolved, league_min_count=league_min_count)
        availability = report.get("availability", {}) if isinstance(report.get("availability"), dict) else {}
        if not bool(availability.get("available", False)):
            return {
                "generated": False,
                "reason": str(availability.get("reason") or "not_available"),
                "report": report,
                "json_path": None,
                "md_path": None,
            }

        out_dir = Path(report_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "model_diagnostic_report.json"
        md_path = out_dir / "model_diagnostic_report.md"

        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(self._render_model_diagnostic_markdown(report), encoding="utf-8")

        return {
            "generated": True,
            "reason": "ok",
            "report": report,
            "json_path": str(json_path),
            "md_path": str(md_path),
        }

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
        rows: list[dict[str, Any]] = []
        skipped_missing_identity = 0
        identity_relinked = 0
        duplicate_input_match_id = 0
        duplicate_input_identity = 0
        seen_match_ids: set[int] = set()
        seen_natural_keys: set[tuple[Any, ...]] = set()

        for match in matches:
            incoming_match_id = self._to_int_or_none(match.get("match_id"))
            resolved_match_id = incoming_match_id

            # Fallback identity resolution for unstable/missing upstream IDs.
            if resolved_match_id is None:
                resolved_match_id = self._find_existing_match_id_by_identity(match)
                if resolved_match_id is None:
                    skipped_missing_identity += 1
                    continue
                identity_relinked += 1
            else:
                existing_match_id = self._find_existing_match_id_by_identity(match)
                if existing_match_id is not None and existing_match_id != resolved_match_id:
                    resolved_match_id = existing_match_id
                    identity_relinked += 1

            if resolved_match_id in seen_match_ids:
                duplicate_input_match_id += 1
                continue

            natural_key = self._match_natural_key(match)
            if natural_key is not None and natural_key in seen_natural_keys:
                duplicate_input_identity += 1
                continue

            seen_match_ids.add(resolved_match_id)
            if natural_key is not None:
                seen_natural_keys.add(natural_key)

            rows.append(
                {
                    "match_id": resolved_match_id,
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
            )

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
                odds_ft_1=COALESCE(excluded.odds_ft_1, matches.odds_ft_1),
                odds_ft_x=COALESCE(excluded.odds_ft_x, matches.odds_ft_x),
                odds_ft_2=COALESCE(excluded.odds_ft_2, matches.odds_ft_2),
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
        logger.info(
            "upserted matches=%s relinked_by_identity=%s skipped_missing_identity=%s duplicate_input_match_id=%s duplicate_input_identity=%s",
            len(rows),
            identity_relinked,
            skipped_missing_identity,
            duplicate_input_match_id,
            duplicate_input_identity,
        )

    def get_completed_matches(self) -> list[dict[str, Any]]:
        """
        Fetch all completed matches with results (winning_team_id not NULL).
        Used for building training datasets from accumulated match history.

        Returns:
            List of dict, each matching a row from the matches table.
            Empty list if no completed matches found.
        """
        try:
            query = """
                SELECT
                    match_id, season_id, date_unix, match_date_iso, status,
                    home_team_id, away_team_id, home_team_name, away_team_name,
                    home_goals, away_goals, winning_team_id,
                    odds_ft_1, odds_ft_x, odds_ft_2,
                    btts_potential, o15_potential, o25_potential, o35_potential, o45_potential,
                    corners_potential, cards_potential, avg_potential,
                    home_ppg, away_ppg, pre_match_home_ppg, pre_match_away_ppg
                FROM matches
                WHERE status = 'completed' AND winning_team_id IS NOT NULL
                ORDER BY date_unix ASC
            """
            cursor = self.conn.execute(query)
            rows = cursor.fetchall()
            if not rows:
                logger.info("get_completed_matches: no completed matches found")
                return []

            result = [dict(row) for row in rows]
            logger.info("get_completed_matches: fetched %s completed matches", len(result))
            return result
        except Exception as exc:
            logger.exception("get_completed_matches failed: %s", exc)
            return []

    def build_training_dataset_from_db(self) -> list[dict[str, Any]]:
        """
        Build a training dataset from completed matches in the database.
        Converts match results into target labels (0/1/2) and assembles feature vectors.
        
        Assembles ALL features required by ModelTrainer.clean_data() including:
        - Odds and implied probabilities (from match odds)
        - Team stats (PPG, defaults if not available)
        - Derived features (entropy, gap, volatility calculated from odds)
        - League stats (defaults to 0.0)

        Target encoding:
            1 = home team won
            2 = away team won
            0 = draw (home_goals = away_goals)

        Returns:
            List of training rows, each with 'target' and ALL 20 required feature columns.
            Compatible with ModelTrainer.clean_data().
            Empty list if no suitable rows found.
        """
        # Query: select completed matches where either winning_team_id is set OR goals are equal (draw)
        try:
            query = """
                SELECT
                    m.match_id, m.season_id, m.date_unix, m.match_date_iso, m.status,
                    m.home_team_id, m.away_team_id, m.home_team_name, m.away_team_name,
                    m.home_goals, m.away_goals, m.winning_team_id,
                    m.odds_ft_1, m.odds_ft_x, m.odds_ft_2,
                    m.btts_potential, m.o15_potential, m.o25_potential, m.o35_potential, m.o45_potential,
                    m.corners_potential, m.cards_potential, m.avg_potential,
                    m.home_ppg, m.away_ppg, m.pre_match_home_ppg, m.pre_match_away_ppg,
                    m.raw_json,
                    -- Home team season stats (pre-match averages from /league-teams?include=stats)
                    htss.goals_for_avg_overall  AS home_goals_for_avg,
                    htss.goals_for_avg_home     AS home_goals_for_avg_home,
                    htss.goals_against_avg_home AS home_goals_against_avg_home,
                    htss.shots_avg_overall      AS home_shots_avg,
                    htss.shots_avg_home         AS home_shots_avg_home,
                    htss.possession_avg_overall AS home_possession_avg,
                    htss.possession_avg_home    AS home_possession_avg_home,
                    htss.xg_for_avg_overall     AS home_xg_for_avg,
                    htss.season_ppg_overall     AS home_season_ppg,
                    htss.team_id                AS home_team_stats_joined_team_id,
                    -- Away team season stats (pre-match averages from /league-teams?include=stats)
                    atss.goals_for_avg_overall  AS away_goals_for_avg,
                    atss.goals_for_avg_away     AS away_goals_for_avg_away,
                    atss.goals_against_avg_away AS away_goals_against_avg_away,
                    atss.shots_avg_overall      AS away_shots_avg,
                    atss.shots_avg_away         AS away_shots_avg_away,
                    atss.possession_avg_overall AS away_possession_avg,
                    atss.possession_avg_away    AS away_possession_avg_away,
                    atss.xg_for_avg_overall     AS away_xg_for_avg,
                    atss.season_ppg_overall     AS away_season_ppg,
                    atss.team_id                AS away_team_stats_joined_team_id,
                    -- League stats (from /league-season)
                    lss.draw_pct                AS league_draw_pct,
                    lss.home_win_pct            AS league_home_win_pct,
                    lss.away_win_pct            AS league_away_win_pct,
                    lss.home_advantage          AS league_home_advantage,
                    lss.season_avg_goals        AS league_avg_goals,
                    lss.season_id               AS league_stats_joined_season_id
                FROM matches m
                LEFT JOIN team_season_stats htss
                    ON htss.team_id = m.home_team_id AND htss.season_id = m.season_id
                LEFT JOIN team_season_stats atss
                    ON atss.team_id = m.away_team_id AND atss.season_id = m.season_id
                LEFT JOIN league_season_stats lss ON lss.season_id = m.season_id
                WHERE m.status = 'completed'
                  AND (winning_team_id IS NOT NULL OR (home_goals IS NOT NULL AND away_goals IS NOT NULL))
                ORDER BY m.date_unix ASC
            """
            cursor = self.conn.execute(query)
            rows = [dict(row) for row in cursor.fetchall()]
            if not rows:
                logger.info("build_training_dataset_from_db: no completed matches found")
                return []
        except Exception as exc:
            logger.exception("build_training_dataset_from_db: query failed: %s", exc)
            return []

        dataset: list[dict[str, Any]] = []
        seen_match_ids: set[int] = set()
        duplicate_match_ids_removed = 0
        invalid_odds_count = 0
        invalid_odds_samples: list[int] = []

        def _normalize_percent_like(value: float | None) -> float | None:
            if value is None:
                return None
            if value > 1.0:
                return value / 100.0
            return value

        # Pre-compute per-season win/draw rates from completed matches.
        # Used as tertiary fallback when league_season_stats API fields are NULL.
        _season_drawn: dict[Any, int] = {}
        _season_home_w: dict[Any, int] = {}
        _season_away_w: dict[Any, int] = {}
        _season_total: dict[Any, int] = {}
        for _r in rows:
            _sid = _r.get("season_id")
            if not _sid:
                continue
            _wt = _r.get("winning_team_id")
            _ht = _r.get("home_team_id")
            _at = _r.get("away_team_id")
            _hg = _r.get("home_goals")
            _ag = _r.get("away_goals")
            if _wt and _wt == _ht:
                _season_home_w[_sid] = _season_home_w.get(_sid, 0) + 1
                _season_total[_sid] = _season_total.get(_sid, 0) + 1
            elif _wt and _wt == _at:
                _season_away_w[_sid] = _season_away_w.get(_sid, 0) + 1
                _season_total[_sid] = _season_total.get(_sid, 0) + 1
            elif _hg is not None and _ag is not None and int(_hg) == int(_ag):
                _season_drawn[_sid] = _season_drawn.get(_sid, 0) + 1
                _season_total[_sid] = _season_total.get(_sid, 0) + 1

        for match in rows:
            try:
                # Validate required fields
                match_id = match.get("match_id")
                home_team_id = match.get("home_team_id")
                away_team_id = match.get("away_team_id")
                home_goals = match.get("home_goals")
                away_goals = match.get("away_goals")
                winning_team_id = match.get("winning_team_id")
                match_date_iso = match.get("match_date_iso")

                if not all([match_id, home_team_id, away_team_id]):
                    logger.debug("Skipping match %s: missing required team ids", match_id)
                    continue

                # Determine target (winning team ID or goals equality → 1/2/0)
                # winning_team_id = 0 or None means draw; truthy value = real winner
                target: int | None = None
                if winning_team_id and winning_team_id == home_team_id:
                    target = 1
                elif winning_team_id and winning_team_id == away_team_id:
                    target = 2
                elif home_goals is not None and away_goals is not None:
                    # Draw: home_goals == away_goals (also covers winning_team_id=0/None case)
                    if home_goals == away_goals:
                        target = 0
                
                if target is None:
                    logger.debug("Skipping match %s: could not determine target (no winner, no goals)", match_id)
                    continue

                # Core odds (required for any model)
                odds_ft_1 = self._safe_float(match.get("odds_ft_1"))
                odds_ft_x = self._safe_float(match.get("odds_ft_x"))
                odds_ft_2 = self._safe_float(match.get("odds_ft_2"))

                if not all([odds_ft_1, odds_ft_x, odds_ft_2]):
                    invalid_odds_count += 1
                    if len(invalid_odds_samples) < 10:
                        try:
                            invalid_odds_samples.append(int(match_id))
                        except (TypeError, ValueError):
                            pass
                    continue

                # Compute implied probabilities from odds
                implied = self._normalize_odds(odds_ft_1, odds_ft_x, odds_ft_2)

                # Team stats (fallback to 0.0 if missing)
                home_ppg = self._safe_float(match.get("home_ppg"), default=0.0)
                away_ppg = self._safe_float(match.get("away_ppg"), default=0.0)
                pre_match_home_ppg = self._safe_float(match.get("pre_match_home_ppg"), default=0.0)
                pre_match_away_ppg = self._safe_float(match.get("pre_match_away_ppg"), default=0.0)

                # Fall back to team_season_stats.season_ppg when match-level ppg is absent
                if home_ppg == 0.0 and match.get("home_season_ppg") is not None:
                    home_ppg = float(match["home_season_ppg"])
                if away_ppg == 0.0 and match.get("away_season_ppg") is not None:
                    away_ppg = float(match["away_season_ppg"])

                # Computed differences and derived features
                ppg_diff = home_ppg - away_ppg
                split_advantage = pre_match_home_ppg - pre_match_away_ppg

                home_team_join_ok = match.get("home_team_stats_joined_team_id") is not None
                away_team_join_ok = match.get("away_team_stats_joined_team_id") is not None
                league_join_ok = match.get("league_stats_joined_season_id") is not None

                def _parse_feature_component(raw_value: Any, join_ok: bool) -> tuple[float | None, str]:
                    if raw_value in (-1, -2, -1.0, -2.0):
                        return None, "missing_sentinel"
                    if raw_value is None:
                        return None, "missing_source" if join_ok else "join_failed"
                    try:
                        value = float(raw_value)
                    except (TypeError, ValueError):
                        return None, "invalid_numeric"
                    return value, ("real_zero" if value == 0.0 else "real_nonzero")

                def _source_for_diff(home_src: str, away_src: str) -> str:
                    if home_src.startswith("real_") and away_src.startswith("real_"):
                        return "real"
                    if home_src == "join_failed" and away_src == "join_failed":
                        return "join_failed"
                    if home_src == "join_failed" or away_src == "join_failed":
                        return "partial_join"
                    if home_src == "missing_sentinel" or away_src == "missing_sentinel":
                        return "missing_sentinel"
                    if home_src == "invalid_numeric" or away_src == "invalid_numeric":
                        return "invalid_numeric"
                    return "missing_source"

                # --- Team-season stats (pre-match averages — no post-match leakage) ---
                # goals_diff: prefer contextual split (home-context vs away-context), then overall fallback.
                home_goals_raw = match.get("home_goals_for_avg_home")
                if home_goals_raw is None:
                    home_goals_raw = match.get("home_goals_for_avg")
                away_goals_raw = match.get("away_goals_for_avg_away")
                if away_goals_raw is None:
                    away_goals_raw = match.get("away_goals_for_avg")

                home_goals_avg, home_goals_src = _parse_feature_component(home_goals_raw, home_team_join_ok)
                away_goals_avg, away_goals_src = _parse_feature_component(away_goals_raw, away_team_join_ok)
                if home_goals_avg is not None and away_goals_avg is not None:
                    goals_diff = float(home_goals_avg) - float(away_goals_avg)
                    goals_diff_source = "real_zero" if goals_diff == 0.0 else "real_nonzero"
                else:
                    # Fallback: derive directional pressure from conceded-side context when available.
                    home_conceded, home_conceded_src = _parse_feature_component(
                        match.get("home_goals_against_avg_home"),
                        home_team_join_ok,
                    )
                    away_conceded, away_conceded_src = _parse_feature_component(
                        match.get("away_goals_against_avg_away"),
                        away_team_join_ok,
                    )
                    if home_conceded is not None and away_conceded is not None:
                        goals_diff = float(away_conceded) - float(home_conceded)
                        goals_diff_source = "real_derived_nonzero" if goals_diff != 0.0 else "real_derived_zero"
                    else:
                        # Tertiary fallback: use xG as goal-scoring proxy when direct goals stats unavailable.
                        home_xg_g, _ = _parse_feature_component(match.get("home_xg_for_avg"), home_team_join_ok)
                        away_xg_g, _ = _parse_feature_component(match.get("away_xg_for_avg"), away_team_join_ok)
                        if home_xg_g is not None and away_xg_g is not None:
                            goals_diff = float(home_xg_g) - float(away_xg_g)
                            goals_diff_source = "real_xg_proxy_nonzero" if goals_diff != 0.0 else "real_xg_proxy_zero"
                        else:
                            goals_diff = 0.0
                            goals_diff_source = _source_for_diff(
                                _source_for_diff(home_goals_src, away_goals_src),
                                _source_for_diff(home_conceded_src, away_conceded_src),
                            )

                # xg_diff: xG average per game; NULL when API doesn't expose it
                home_xg_avg, home_xg_src = _parse_feature_component(match.get("home_xg_for_avg"), home_team_join_ok)
                away_xg_avg, away_xg_src = _parse_feature_component(match.get("away_xg_for_avg"), away_team_join_ok)
                if home_xg_avg is not None and away_xg_avg is not None:
                    xg_diff = float(home_xg_avg) - float(away_xg_avg)
                    xg_diff_source = "real_zero" if xg_diff == 0.0 else "real_nonzero"
                else:
                    xg_diff = 0.0
                    xg_diff_source = _source_for_diff(home_xg_src, away_xg_src)

                # shots_diff: prefer context-specific avg (home team's home-avg vs away team's away-avg)
                home_shots_raw = match.get("home_shots_avg_home")
                if home_shots_raw is None:
                    home_shots_raw = match.get("home_shots_avg")
                away_shots_raw = match.get("away_shots_avg_away")
                if away_shots_raw is None:
                    away_shots_raw = match.get("away_shots_avg")
                home_shots_val, home_shots_src = _parse_feature_component(home_shots_raw, home_team_join_ok)
                away_shots_val, away_shots_src = _parse_feature_component(away_shots_raw, away_team_join_ok)
                if home_shots_val is not None and away_shots_val is not None:
                    shots_diff = float(home_shots_val) - float(away_shots_val)
                    shots_diff_source = "real_zero" if shots_diff == 0.0 else "real_nonzero"
                else:
                    shots_diff = 0.0
                    shots_diff_source = _source_for_diff(home_shots_src, away_shots_src)

                # possession_diff: prefer context-specific averages
                home_poss_raw = match.get("home_possession_avg_home")
                if home_poss_raw is None:
                    home_poss_raw = match.get("home_possession_avg")
                away_poss_raw = match.get("away_possession_avg_away")
                if away_poss_raw is None:
                    away_poss_raw = match.get("away_possession_avg")
                home_poss_val, home_poss_src = _parse_feature_component(home_poss_raw, home_team_join_ok)
                away_poss_val, away_poss_src = _parse_feature_component(away_poss_raw, away_team_join_ok)
                if home_poss_val is not None and away_poss_val is not None:
                    possession_diff = float(home_poss_val) - float(away_poss_val)
                    possession_diff_source = "real_zero" if possession_diff == 0.0 else "real_nonzero"
                else:
                    possession_diff = 0.0
                    possession_diff_source = _source_for_diff(home_poss_src, away_poss_src)

                draw_pct_value, draw_pct_source = _parse_feature_component(match.get("league_draw_pct"), league_join_ok)
                if draw_pct_value is not None:
                    draw_pct = float(_normalize_percent_like(draw_pct_value) or 0.0)
                    draw_pct_source = "real_zero" if draw_pct == 0.0 else "real_nonzero"
                else:
                    # Derive from league home/away split if draw_pct missing.
                    home_win_raw = self._to_float_or_none(match.get("league_home_win_pct"))
                    away_win_raw = self._to_float_or_none(match.get("league_away_win_pct"))
                    home_win = _normalize_percent_like(home_win_raw)
                    away_win = _normalize_percent_like(away_win_raw)
                    if home_win is not None and away_win is not None:
                        draw_pct = max(0.0, min(1.0, 1.0 - home_win - away_win))
                        draw_pct_source = "real_derived_nonzero" if draw_pct != 0.0 else "real_derived_zero"
                    else:
                        # Tertiary fallback from season history.
                        sid = match.get("season_id")
                        season_total = _season_total.get(sid, 0) if sid else 0
                        if season_total > 0:
                            draw_pct = _season_drawn.get(sid, 0) / season_total
                            draw_pct_source = "real_derived_history_nonzero" if draw_pct != 0.0 else "real_derived_history_zero"
                        else:
                            draw_pct = 0.0
                            draw_pct_source = "missing_source"

                home_advantage_value, home_advantage_source = _parse_feature_component(match.get("league_home_advantage"), league_join_ok)
                if home_advantage_value is not None:
                    home_advantage = float(_normalize_percent_like(home_advantage_value) or 0.0)
                    home_advantage_source = "real_zero" if home_advantage == 0.0 else "real_nonzero"
                else:
                    # Derive directional edge from league home/away win rates.
                    home_win_raw = self._to_float_or_none(match.get("league_home_win_pct"))
                    away_win_raw = self._to_float_or_none(match.get("league_away_win_pct"))
                    home_win = _normalize_percent_like(home_win_raw)
                    away_win = _normalize_percent_like(away_win_raw)
                    if home_win is not None and away_win is not None:
                        home_advantage = home_win - away_win
                        home_advantage_source = "real_derived_nonzero" if home_advantage != 0.0 else "real_derived_zero"
                    else:
                        # Tertiary fallback from season history.
                        sid = match.get("season_id")
                        season_total = _season_total.get(sid, 0) if sid else 0
                        if season_total > 0:
                            home_advantage = _season_home_w.get(sid, 0) / season_total - _season_away_w.get(sid, 0) / season_total
                            home_advantage_source = "real_derived_history_nonzero" if home_advantage != 0.0 else "real_derived_history_zero"
                        else:
                            home_advantage = 0.0
                            home_advantage_source = "missing_source"

                avg_goals_value, avg_goals_source = _parse_feature_component(match.get("league_avg_goals"), league_join_ok)
                if avg_goals_value is not None:
                    avg_goals = float(avg_goals_value)
                    avg_goals_source = "real_zero" if avg_goals == 0.0 else "real_nonzero"
                else:
                    # Derive from match potential first, then team scoring means.
                    avg_potential = self._to_float_or_none(match.get("avg_potential"))
                    if avg_potential is not None and avg_potential > 0.0:
                        avg_goals = float(avg_potential)
                        avg_goals_source = "real_derived_nonzero"
                    elif home_goals_avg is not None and away_goals_avg is not None:
                        avg_goals = float((home_goals_avg + away_goals_avg) / 2.0)
                        avg_goals_source = "real_derived_nonzero" if avg_goals != 0.0 else "real_derived_zero"
                    else:
                        avg_goals = 0.0

                # Entropy, gap, volatility (from odds-derived probabilities)
                probs = [implied["implied_prob_1"], implied["implied_prob_x"], implied["implied_prob_2"]]
                entropy = self._calc_entropy(probs)
                probs_sorted = sorted(probs, reverse=True)
                gap = float(probs_sorted[0] - probs_sorted[1]) if len(probs_sorted) >= 2 else 0.0
                volatility = self._calc_volatility(probs)

                # Build training row with ALL required feature columns (20 total)
                row: dict[str, Any] = {
                    "target": target,
                    "match_date": match_date_iso or str(match.get("date_unix", "")),
                    "__match_id": match_id,
                    # 1) MATCH LEVEL (odds-based)
                    "odds_ft_1": odds_ft_1,
                    "odds_ft_x": odds_ft_x,
                    "odds_ft_2": odds_ft_2,
                    "implied_prob_1": implied["implied_prob_1"],
                    "implied_prob_x": implied["implied_prob_x"],
                    "implied_prob_2": implied["implied_prob_2"],
                    # 2) TEAM DIFF
                    "ppg_diff": ppg_diff,
                    "goals_diff": goals_diff,
                    "xg_diff": xg_diff,
                    "shots_diff": shots_diff,
                    "possession_diff": possession_diff,
                    # 3) HOME/AWAY
                    "home_home_ppg": home_ppg,
                    "away_away_ppg": away_ppg,
                    # 4) LEAGUE
                    "draw_pct": draw_pct,
                    "home_advantage": home_advantage,
                    "avg_goals": avg_goals,
                    # 5) RISK/UNCERTAINTY
                    "entropy": entropy,
                    "gap": gap,
                    "volatility": volatility,
                    # Additional computed
                    "split_advantage": split_advantage,
                    # Feature source audit metadata
                    "__source_goals_diff": goals_diff_source,
                    "__source_xg_diff": xg_diff_source,
                    "__source_shots_diff": shots_diff_source,
                    "__source_possession_diff": possession_diff_source,
                    "__source_draw_pct": draw_pct_source,
                    "__source_home_advantage": home_advantage_source,
                    "__source_avg_goals": avg_goals_source,
                }

                dataset.append(row)
            except Exception as exc:
                logger.debug("Error building training row for match %s: %s", match_id if 'match_id' in dir() else "?", exc)
                continue

        rows_after_feature_build = len(dataset)

        if invalid_odds_count > 0:
            logger.warning(
                "build_training_dataset: skipped %s completed rows due to invalid odds triplet (sample match_ids=%s)",
                invalid_odds_count,
                invalid_odds_samples,
            )

        deduped_dataset: list[dict[str, Any]] = []
        for row in dataset:
            match_id_raw = row.get("__match_id")
            if match_id_raw is None:
                deduped_dataset.append(row)
                continue
            try:
                match_id_int = int(match_id_raw)
            except (TypeError, ValueError):
                deduped_dataset.append(row)
                continue
            if match_id_int in seen_match_ids:
                duplicate_match_ids_removed += 1
                continue
            seen_match_ids.add(match_id_int)
            deduped_dataset.append(row)

        dataset = deduped_dataset
        rows_after_dedup = len(dataset)

        # ── Feature coverage diagnostic ──────────────────────────────────────
        # Aggregates __source_* metadata added per-row above and logs clearly
        # so callers can see whether priors are alive or still fallback-zero.
        _TRACKED_FEATURES = (
            "goals_diff",
            "xg_diff",
            "shots_diff",
            "possession_diff",
            "draw_pct",
            "home_advantage",
            "avg_goals",
        )
        total_built = len(dataset)
        null_sid_count = sum(1 for r in rows if r.get("season_id") is None)
        with_sid_count = len(rows) - null_sid_count

        if null_sid_count > 0:
            logger.warning(
                "build_training_dataset: %s/%s raw DB rows have NULL season_id → team/league stat joins fire for NONE of them. "
                "Call backfill_season_id_from_raw_json() then re-load historical data to fix.",
                null_sid_count,
                len(rows),
            )

        any_weak = False
        for feature in _TRACKED_FEATURES:
            real_count = sum(1 for r in dataset if str(r.get(f"__source_{feature}", "")).startswith("real_"))
            fallback_count = total_built - real_count
            rate = round(real_count / total_built, 4) if total_built > 0 else 0.0
            if rate >= 0.5:
                status = "healthy"
            elif rate > 0.0:
                status = "partial"
                any_weak = True
            else:
                status = "zero_heavy"
                any_weak = True
            logger.info(
                "feature_coverage  %-24s  real=%s/%s (%.1f%%)  status=%s",
                feature,
                real_count,
                total_built,
                rate * 100,
                status,
            )

        weak_feature_coverage: dict[str, Any] = {}
        weak_tracked = ("goals_diff", "draw_pct", "home_advantage", "avg_goals")
        for feature in weak_tracked:
            source_key = f"__source_{feature}"
            source_counts: dict[str, int] = {}
            real_count = 0
            fallback_count = 0
            zero_count = 0

            for row in dataset:
                source = str(row.get(source_key, "unknown") or "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
                value = self._to_float_or_none(row.get(feature))
                if value is not None and value == 0.0:
                    zero_count += 1

                if source.startswith("real_"):
                    real_count += 1
                elif source in {
                    "join_failed",
                    "partial_join",
                    "missing_source",
                    "missing_sentinel",
                    "invalid_numeric",
                    "fallback_default",
                }:
                    fallback_count += 1

            fill_rate = (real_count / total_built) if total_built > 0 else 0.0
            zero_rate = (zero_count / total_built) if total_built > 0 else 0.0
            fallback_rate = (fallback_count / total_built) if total_built > 0 else 0.0

            if fill_rate >= 0.8 and fallback_rate <= 0.2:
                source_status = "healthy"
            elif fill_rate > 0.0:
                source_status = "degraded"
            else:
                source_status = "weak"

            weak_feature_coverage[feature] = {
                "fill_rate": round(fill_rate, 4),
                "zero_rate": round(zero_rate, 4),
                "fallback_rate": round(fallback_rate, 4),
                "source_status": source_status,
                "source_breakdown": source_counts,
            }

            logger.info(
                "weak_feature_diag  %-16s fill=%.1f%% zero=%.1f%% fallback=%.1f%% status=%s breakdown=%s",
                feature,
                fill_rate * 100,
                zero_rate * 100,
                fallback_rate * 100,
                source_status,
                source_counts,
            )

        weak_model_features_present = any(
            str(weak_feature_coverage.get(name, {}).get("source_status", "")) != "healthy"
            for name in weak_tracked
        )

        # P2 quality diagnostics on model-input implied probabilities.
        p2_ge_020 = sum(1 for row in dataset if float(row.get("implied_prob_2", 0.0)) >= 0.20)
        p2_ge_025 = sum(1 for row in dataset if float(row.get("implied_prob_2", 0.0)) >= 0.25)
        p2_ge_030 = sum(1 for row in dataset if float(row.get("implied_prob_2", 0.0)) >= 0.30)
        p2_top1 = sum(
            1
            for row in dataset
            if float(row.get("implied_prob_2", 0.0)) >= max(
                float(row.get("implied_prob_1", 0.0)),
                float(row.get("implied_prob_x", 0.0)),
            )
        )

        if any_weak:
            if null_sid_count > 0:
                logger.warning(
                    "build_training_dataset: some features are zero_heavy/partial. "
                    "Primary cause: season_id NULL in %s/%s match rows → priors joins fail. "
                    "Fix: run backfill_season_id_from_raw_json() then re-run historical load and re-train.",
                    null_sid_count,
                    len(rows),
                )
            else:
                team_stats_count = self.conn.execute("SELECT COUNT(*) FROM team_season_stats").fetchone()[0]
                league_stats_count = self.conn.execute("SELECT COUNT(*) FROM league_season_stats").fetchone()[0]
                logger.warning(
                    "build_training_dataset: some features are zero_heavy/partial. "
                    "season_id is OK (0/%s NULL). Root cause: team_season_stats=%s rows, league_season_stats=%s rows. "
                    "Fix: run historical load (load_league_teams + load_league_season) for the %s season_ids found in matches.",
                    len(rows),
                    team_stats_count,
                    league_stats_count,
                    with_sid_count,
                )
        else:
            logger.info(
                "build_training_dataset: all tracked features healthy. "
                "rows_with_sid=%s/%s",
                with_sid_count,
                len(rows),
            )
        # ─────────────────────────────────────────────────────────────────────

        logger.info(
            "build_training_dataset_from_db: built %s training rows from %s completed/drawable matches "
            "(with_sid=%s null_sid=%s)",
            len(dataset),
            len(rows),
            with_sid_count,
            null_sid_count,
        )

        self.last_training_dataset_debug = {
            "raw_completed_rows": len(rows),
            "unique_completed_matches": len({r.get("match_id") for r in rows if r.get("match_id") is not None}),
            "rows_after_feature_build": rows_after_feature_build,
            "rows_after_dedup": rows_after_dedup,
            "duplicate_rows_removed": duplicate_match_ids_removed,
            "skipped_due_to_invalid_odds": invalid_odds_count,
            "invalid_odds_sample_match_ids": invalid_odds_samples,
            "null_season_rows": null_sid_count,
            "with_season_rows": with_sid_count,
            "weak_feature_coverage": weak_feature_coverage,
            "weak_model_features_present": weak_model_features_present,
            "p2_probability_quality": {
                "avg_p2": round(
                    sum(float(row.get("implied_prob_2", 0.0)) for row in dataset) / len(dataset),
                    4,
                ) if dataset else 0.0,
                "matches_p2_ge_0_20": p2_ge_020,
                "matches_p2_ge_0_25": p2_ge_025,
                "matches_p2_ge_0_30": p2_ge_030,
                "matches_p2_top1": p2_top1,
            },
        }
        logger.info(
            "train_dataset_debug raw_completed_rows=%s unique_completed_matches=%s rows_after_feature_build=%s rows_after_dedup=%s duplicate_rows_removed=%s",
            self.last_training_dataset_debug["raw_completed_rows"],
            self.last_training_dataset_debug["unique_completed_matches"],
            self.last_training_dataset_debug["rows_after_feature_build"],
            self.last_training_dataset_debug["rows_after_dedup"],
            self.last_training_dataset_debug["duplicate_rows_removed"],
        )
        return dataset

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float, return default if invalid."""
        if value is None:
            return default
        try:
            f = float(value)
            if f <= 0 or f > 1000:  # Sanity check for odds
                return default
            return f
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float_or_none(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_odds(odds_1: float, odds_x: float, odds_2: float) -> dict[str, float]:
        """
        Normalize odds to implied probabilities.
        Handles edge cases (zero/negative odds).
        """
        eps = 1e-12
        try:
            inv_1 = 1.0 / (odds_1 + eps)
            inv_x = 1.0 / (odds_x + eps)
            inv_2 = 1.0 / (odds_2 + eps)
        except ZeroDivisionError:
            return {"implied_prob_1": 1.0 / 3, "implied_prob_x": 1.0 / 3, "implied_prob_2": 1.0 / 3}

        total = inv_1 + inv_x + inv_2
        if total <= 0:
            return {"implied_prob_1": 1.0 / 3, "implied_prob_x": 1.0 / 3, "implied_prob_2": 1.0 / 3}

        return {
            "implied_prob_1": float(inv_1 / total),
            "implied_prob_x": float(inv_x / total),
            "implied_prob_2": float(inv_2 / total),
        }

    @staticmethod
    def _calc_entropy(probs: list[float]) -> float:
        """Calculate Shannon entropy (0 = certain, 1 = max uncertainty for 3 classes)."""
        import math
        eps = 1e-12
        entropy = -sum(p * math.log(max(p, eps), 2) for p in probs if p > 0.0)
        max_entropy = math.log(3, 2)  # log2(3) for 3 classes
        if max_entropy <= 0:
            return 0.0
        return float(entropy / max_entropy)

    @staticmethod
    def _calc_volatility(probs: list[float]) -> float:
        """Calculate standard deviation of probabilities (measure of prediction uncertainty)."""
        if not probs:
            return 0.0
        mean = sum(probs) / len(probs)
        variance = sum((p - mean) ** 2 for p in probs) / len(probs)
        return float(variance ** 0.5)

    def audit_api_sqlite_pipeline(self, limit: int = 5000) -> dict[str, Any]:
        """Audit raw API payload coverage as persisted in SQLite and return field presence summary."""
        query = """
            SELECT raw_json, odds_ft_1, odds_ft_x, odds_ft_2, home_ppg, away_ppg
            FROM matches
            WHERE status = 'completed'
            ORDER BY date_unix DESC
            LIMIT ?
        """
        rows = self.conn.execute(query, (int(limit),)).fetchall()
        total = len(rows)

        def _has_any(payload: dict[str, Any], keys: list[str]) -> bool:
            for key in keys:
                value = payload.get(key)
                if value not in (None, "", [], {}):
                    return True
            return False

        coverage = {
            "goals": 0,
            "xg": 0,
            "shots": 0,
            "possession": 0,
            "odds": 0,
        }
        parse_errors = 0

        for row_raw in rows:
            row = dict(row_raw)
            try:
                payload = json.loads(row.get("raw_json") or "{}")
                if not isinstance(payload, dict):
                    payload = {}
            except Exception:
                payload = {}
                parse_errors += 1

            if _has_any(payload, ["homeGoalCount", "awayGoalCount", "home_goals", "away_goals"]):
                coverage["goals"] += 1
            if _has_any(payload, ["xg_home", "xg_away", "home_xg", "away_xg", "xg"]):
                coverage["xg"] += 1
            if _has_any(payload, ["shots_home", "shots_away", "home_shots", "away_shots", "shots"]):
                coverage["shots"] += 1
            if _has_any(payload, ["possession_home", "possession_away", "home_possession", "away_possession", "possession"]):
                coverage["possession"] += 1

            if row.get("odds_ft_1") is not None and row.get("odds_ft_x") is not None and row.get("odds_ft_2") is not None:
                coverage["odds"] += 1

        def _ratio(count: int) -> float:
            return round((count / total), 4) if total > 0 else 0.0

        return {
            "total_completed_rows_checked": total,
            "raw_json_parse_errors": parse_errors,
            "api_field_coverage": {
                "goals": {"count": coverage["goals"], "ratio": _ratio(coverage["goals"])},
                "xg": {"count": coverage["xg"], "ratio": _ratio(coverage["xg"])},
                "shots": {"count": coverage["shots"], "ratio": _ratio(coverage["shots"])},
                "possession": {"count": coverage["possession"], "ratio": _ratio(coverage["possession"])},
                "odds": {"count": coverage["odds"], "ratio": _ratio(coverage["odds"])},
            },
            "notes": [
                "Coverage is based on raw_json keys persisted in SQLite.",
                "Low ratio means field is often absent in upstream API payload or ingestion mapping.",
            ],
        }

    def audit_season_id_coverage(self) -> dict[str, Any]:
        """
        Audit season_id coverage across matches and stats tables.
        CRITICAL: This determines if team/league priors can join to matches.
        
        Returns comprehensive coverage metrics for diagnostics.
        """
        # Total matches and completed
        total_matches = int(self.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        matches_with_season_id = int(self.conn.execute("SELECT COUNT(*) FROM matches WHERE season_id IS NOT NULL").fetchone()[0])
        completed_matches = int(self.conn.execute("SELECT COUNT(*) FROM matches WHERE status = 'completed'").fetchone()[0])
        completed_with_season_id = int(self.conn.execute("SELECT COUNT(*) FROM matches WHERE status = 'completed' AND season_id IS NOT NULL").fetchone()[0])
        
        # Distinct seasons in matches vs stats tables
        distinct_seasons_in_matches = int(self.conn.execute("SELECT COUNT(DISTINCT season_id) FROM matches WHERE season_id IS NOT NULL").fetchone()[0])
        distinct_seasons_in_stats = int(self.conn.execute("SELECT COUNT(DISTINCT season_id) FROM team_season_stats WHERE season_id IS NOT NULL").fetchone()[0])
        distinct_seasons_in_league_stats = int(self.conn.execute("SELECT COUNT(DISTINCT season_id) FROM league_season_stats WHERE season_id IS NOT NULL").fetchone()[0])
        
        # Stats rows
        team_stats_rows = int(self.conn.execute("SELECT COUNT(*) FROM team_season_stats").fetchone()[0])
        league_stats_rows = int(self.conn.execute("SELECT COUNT(*) FROM league_season_stats").fetchone()[0])
        
        # Can the joins actually work?
        # This simulates what build_training_dataset_from_db does:
        # JOIN team_season_stats ON team_id=home_team_id AND season_id=m.season_id
        # If m.season_id is NULL, join never fires.
        joinable_matches = int(self.conn.execute("""
            SELECT COUNT(DISTINCT m.match_id)
            FROM matches m
            WHERE EXISTS (
                SELECT 1 FROM team_season_stats ts
                WHERE ts.team_id = m.home_team_id AND ts.season_id = m.season_id AND ts.season_id IS NOT NULL
            )
        """).fetchone()[0])
        
        # Teams with stats per season
        teams_with_stats = int(self.conn.execute("SELECT COUNT(DISTINCT team_id) FROM team_season_stats").fetchone()[0])
        
        return {
            "season_id_audit": {
                "matches_total": total_matches,
                "matches_with_season_id": matches_with_season_id,
                "completion_rate": f"{round(100.0 * matches_with_season_id / total_matches, 1)}%" if total_matches > 0 else "0%",
                "completed_matches_total": completed_matches,
                "completed_with_season_id": completed_with_season_id,
                "completed_season_id_rate": f"{round(100.0 * completed_with_season_id / completed_matches, 1)}%" if completed_matches > 0 else "0%",
            },
            "seasons_available": {
                "distinct_in_matches": distinct_seasons_in_matches,
                "distinct_in_team_stats": distinct_seasons_in_stats,
                "distinct_in_league_stats": distinct_seasons_in_league_stats,
            },
            "stats_rows": {
                "team_season_stats": team_stats_rows,
                "league_season_stats": league_stats_rows,
                "teams_with_stats": teams_with_stats,
            },
            "join_viability": {
                "matches_joinable_to_team_stats": joinable_matches,
                "joinable_rate": f"{round(100.0 * joinable_matches / completed_matches, 1)}%" if completed_matches > 0 else "0%",
                "interpretation": "If joinable_rate < 50%, most features will fallback to 0.0 during training." if joinable_matches < completed_matches // 2 else "Good join coverage; features should be populated.",
            },
            "critical_issues": [
                "CRITICAL: season_id is NULL in matches" if completed_with_season_id == 0 else None,
                "WARNING: No team stats loaded" if team_stats_rows == 0 else None,
                "WARNING: No league stats loaded" if league_stats_rows == 0 else None,
                "ERROR: Matches have season_id but no stats exist" if completed_with_season_id > 0 and team_stats_rows == 0 else None,
            ]
        }

    def backfill_season_id_from_raw_json(self) -> dict[str, Any]:
        """
        Backfill season_id for matches that were ingested before the normalizer fix.

        Reads raw_json stored in the matches table for rows where season_id IS NULL,
        extracts competition_id / season_id numeric aliases from the raw API payload,
        and updates the matches table in-place.

        This is the fastest way to restore join-viability for existing data without
        a full re-ingest from the API.

        Returns a diagnostic summary suitable for logging / UI display.
        """
        _SEASON_ALIASES = (
            "season_id",
            "competition_id",
            "seasonId",
            "competitionId",
            "comp_id",
            "seasonID",
            "competitionID",
        )

        null_rows = self.conn.execute(
            "SELECT match_id, raw_json FROM matches WHERE season_id IS NULL"
        ).fetchall()

        total_null = len(null_rows)
        if total_null == 0:
            logger.info("backfill_season_id: no NULL season_id rows found, nothing to do")
            return {
                "total_null_before": 0,
                "fixed": 0,
                "still_missing": 0,
                "fix_rate": "n/a",
            }

        updates: list[tuple[int, int]] = []
        still_missing = 0

        for row in null_rows:
            match_id = row["match_id"]
            try:
                payload = json.loads(row["raw_json"] or "{}")
            except Exception:
                payload = {}

            resolved_season_id: int | None = None
            for alias in _SEASON_ALIASES:
                raw_value = payload.get(alias)
                if raw_value is None:
                    continue
                try:
                    parsed = int(raw_value)
                    if parsed > 0:
                        resolved_season_id = parsed
                        break
                except (TypeError, ValueError):
                    continue

            if resolved_season_id is not None:
                updates.append((resolved_season_id, match_id))
            else:
                still_missing += 1

        if updates:
            self.conn.executemany(
                "UPDATE matches SET season_id = ? WHERE match_id = ? AND season_id IS NULL",
                updates,
            )
            self.conn.commit()

        fixed = len(updates)
        fix_rate = f"{round(100.0 * fixed / total_null, 1)}%" if total_null > 0 else "n/a"

        logger.info(
            "backfill_season_id: total_null=%s fixed=%s still_missing=%s fix_rate=%s",
            total_null, fixed, still_missing, fix_rate,
        )
        if still_missing > 0:
            logger.warning(
                "backfill_season_id: %s matches could not be backfilled (raw_json has no numeric season/competition id). "
                "These rows will still fall back to 0.0 for team/league stats features until re-ingested.",
                still_missing,
            )

        return {
            "total_null_before": total_null,
            "fixed": fixed,
            "still_missing": still_missing,
            "fix_rate": fix_rate,
        }

    def get_dataset_coverage_diagnostic(self) -> dict[str, Any]:
        """
        Run a dry-run of build_training_dataset_from_db and return feature coverage
        without training, so callers can answer 'are priors alive?' before committing.
        """
        dataset = self.build_training_dataset_from_db()
        _TRACKED = (
            "goals_diff",
            "xg_diff",
            "shots_diff",
            "possession_diff",
            "draw_pct",
            "home_advantage",
            "avg_goals",
        )
        total = len(dataset)
        features_coverage: dict[str, Any] = {}
        for feature in _TRACKED:
            real_count = sum(1 for r in dataset if str(r.get(f"__source_{feature}", "")).startswith("real_"))
            fallback_count = total - real_count
            rate = round(real_count / total, 4) if total > 0 else 0.0
            if rate >= 0.5:
                status = "healthy"
            elif rate > 0.0:
                status = "partial"
            else:
                status = "zero_heavy"
            features_coverage[feature] = {
                "real": real_count,
                "fallback": fallback_count,
                "rate": rate,
                "status": status,
            }

        null_sid_matches = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM matches WHERE status = 'completed' AND season_id IS NULL"
            ).fetchone()[0]
        )
        with_sid_matches = int(
            self.conn.execute(
                "SELECT COUNT(*) FROM matches WHERE status = 'completed' AND season_id IS NOT NULL"
            ).fetchone()[0]
        )

        return {
            "dataset_rows": total,
            "completed_matches_with_season_id": with_sid_matches,
            "completed_matches_missing_season_id": null_sid_matches,
            "features": features_coverage,
            "priors_alive": all(
                v["status"] == "healthy" for v in features_coverage.values()
            ),
            "model_is_odds_only_baseline": all(
                v["status"] == "zero_heavy"
                for v in features_coverage.values()
                if v is not None
            ),
        }

    def audit_match_duplicates(self, sample_limit: int = 50) -> dict[str, Any]:
        """
        Audit potential match duplication patterns and upsert-key risks.

        Returns grouped counts and representative samples for:
        - duplicate match_id (should be impossible with PK, acts as integrity guard)
        - duplicate home/away/date_day identity
        - duplicate season/home/away/date_day identity
        - multiple statuses for same natural match identity
        """
        sample_limit = max(1, int(sample_limit))

        total_matches = int(self.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        missing_match_id = int(self.conn.execute("SELECT COUNT(*) FROM matches WHERE match_id IS NULL").fetchone()[0])

        duplicate_match_id_groups = int(
            self.conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT match_id
                    FROM matches
                    WHERE match_id IS NOT NULL
                    GROUP BY match_id
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )

        duplicate_team_date_groups = int(
            self.conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT
                        LOWER(TRIM(COALESCE(home_team_name, ''))) AS home_name_norm,
                        LOWER(TRIM(COALESCE(away_team_name, ''))) AS away_name_norm,
                        DATE(COALESCE(match_date_iso, datetime(date_unix, 'unixepoch'))) AS match_day,
                        COUNT(*) AS c
                    FROM matches
                    GROUP BY home_name_norm, away_name_norm, match_day
                    HAVING c > 1
                )
                """
            ).fetchone()[0]
        )

        duplicate_season_team_date_groups = int(
            self.conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT
                        season_id,
                        home_team_id,
                        away_team_id,
                        DATE(COALESCE(match_date_iso, datetime(date_unix, 'unixepoch'))) AS match_day,
                        COUNT(*) AS c
                    FROM matches
                    WHERE season_id IS NOT NULL
                      AND home_team_id IS NOT NULL
                      AND away_team_id IS NOT NULL
                      AND (match_date_iso IS NOT NULL OR date_unix IS NOT NULL)
                    GROUP BY season_id, home_team_id, away_team_id, match_day
                    HAVING c > 1
                )
                """
            ).fetchone()[0]
        )

        mixed_status_groups = int(
            self.conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT
                        season_id,
                        home_team_id,
                        away_team_id,
                        DATE(COALESCE(match_date_iso, datetime(date_unix, 'unixepoch'))) AS match_day,
                        COUNT(DISTINCT COALESCE(status, '')) AS status_count
                    FROM matches
                    WHERE season_id IS NOT NULL
                      AND home_team_id IS NOT NULL
                      AND away_team_id IS NOT NULL
                      AND (match_date_iso IS NOT NULL OR date_unix IS NOT NULL)
                    GROUP BY season_id, home_team_id, away_team_id, match_day
                    HAVING status_count > 1
                )
                """
            ).fetchone()[0]
        )

        duplicate_team_date_samples = [
            dict(row)
            for row in self.conn.execute(
                """
                SELECT
                    LOWER(TRIM(COALESCE(home_team_name, ''))) AS home_name_norm,
                    LOWER(TRIM(COALESCE(away_team_name, ''))) AS away_name_norm,
                    DATE(COALESCE(match_date_iso, datetime(date_unix, 'unixepoch'))) AS match_day,
                    COUNT(*) AS duplicate_count,
                    GROUP_CONCAT(match_id) AS match_ids,
                    GROUP_CONCAT(COALESCE(status, '')) AS statuses
                FROM matches
                GROUP BY home_name_norm, away_name_norm, match_day
                HAVING duplicate_count > 1
                ORDER BY duplicate_count DESC
                LIMIT ?
                """,
                (sample_limit,),
            ).fetchall()
        ]

        duplicate_season_team_date_samples = [
            dict(row)
            for row in self.conn.execute(
                """
                SELECT
                    season_id,
                    home_team_id,
                    away_team_id,
                    DATE(COALESCE(match_date_iso, datetime(date_unix, 'unixepoch'))) AS match_day,
                    COUNT(*) AS duplicate_count,
                    GROUP_CONCAT(match_id) AS match_ids,
                    GROUP_CONCAT(COALESCE(status, '')) AS statuses
                FROM matches
                WHERE season_id IS NOT NULL
                  AND home_team_id IS NOT NULL
                  AND away_team_id IS NOT NULL
                  AND (match_date_iso IS NOT NULL OR date_unix IS NOT NULL)
                GROUP BY season_id, home_team_id, away_team_id, match_day
                HAVING duplicate_count > 1
                ORDER BY duplicate_count DESC
                LIMIT ?
                """,
                (sample_limit,),
            ).fetchall()
        ]

        mixed_status_samples = [
            dict(row)
            for row in self.conn.execute(
                """
                SELECT
                    season_id,
                    home_team_id,
                    away_team_id,
                    DATE(COALESCE(match_date_iso, datetime(date_unix, 'unixepoch'))) AS match_day,
                    COUNT(*) AS row_count,
                    COUNT(DISTINCT COALESCE(status, '')) AS status_count,
                    GROUP_CONCAT(DISTINCT COALESCE(status, '')) AS statuses,
                    GROUP_CONCAT(match_id) AS match_ids
                FROM matches
                WHERE season_id IS NOT NULL
                  AND home_team_id IS NOT NULL
                  AND away_team_id IS NOT NULL
                  AND (match_date_iso IS NOT NULL OR date_unix IS NOT NULL)
                GROUP BY season_id, home_team_id, away_team_id, match_day
                HAVING status_count > 1
                ORDER BY row_count DESC
                LIMIT ?
                """,
                (sample_limit,),
            ).fetchall()
        ]

        diagnostics = {
            "total_matches": total_matches,
            "missing_match_id_rows": missing_match_id,
            "duplicate_match_id_groups": duplicate_match_id_groups,
            "duplicate_home_away_date_groups": duplicate_team_date_groups,
            "duplicate_season_home_away_date_groups": duplicate_season_team_date_groups,
            "mixed_status_identity_groups": mixed_status_groups,
            "samples": {
                "duplicate_home_away_date": duplicate_team_date_samples,
                "duplicate_season_home_away_date": duplicate_season_team_date_samples,
                "mixed_status_identity": mixed_status_samples,
            },
            "upsert_identity_key": {
                "current_pk": "match_id",
                "secondary_natural_key_used_for_relink": [
                    "season_id",
                    "home_team_id",
                    "away_team_id",
                    "match_day",
                ],
            },
        }

        logger.info(
            "audit_match_duplicates total=%s dup_match_id=%s dup_home_away_day=%s dup_season_home_away_day=%s mixed_status=%s",
            total_matches,
            duplicate_match_id_groups,
            duplicate_team_date_groups,
            duplicate_season_team_date_groups,
            mixed_status_groups,
        )
        return diagnostics

    @staticmethod
    def _to_int_or_none(value: Any) -> int | None:
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return parsed

    @staticmethod
    def _day_from_match_payload(match: dict[str, Any]) -> str | None:
        iso_raw = match.get("match_date_iso")
        if isinstance(iso_raw, str) and iso_raw.strip():
            iso = iso_raw.strip()
            try:
                dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
                return dt.date().isoformat()
            except ValueError:
                # Fallback for loose 'YYYY-MM-DD ...' style strings.
                return iso[:10] if len(iso) >= 10 else None

        date_unix = match.get("date_unix")
        try:
            if date_unix is None:
                return None
            ts = int(float(date_unix))
            if ts <= 0:
                return None
            return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    def _match_natural_key(self, match: dict[str, Any]) -> tuple[Any, ...] | None:
        season_id = self._to_int_or_none(match.get("season_id"))
        home_team_id = self._to_int_or_none(match.get("home_team_id"))
        away_team_id = self._to_int_or_none(match.get("away_team_id"))
        match_day = self._day_from_match_payload(match)
        if season_id is None or home_team_id is None or away_team_id is None or not match_day:
            return None
        return (season_id, home_team_id, away_team_id, match_day)

    def _find_existing_match_id_by_identity(self, match: dict[str, Any]) -> int | None:
        natural_key = self._match_natural_key(match)
        if natural_key is None:
            return None
        season_id, home_team_id, away_team_id, match_day = natural_key
        row = self.conn.execute(
            """
            SELECT match_id
            FROM matches
            WHERE season_id = ?
              AND home_team_id = ?
              AND away_team_id = ?
              AND DATE(COALESCE(match_date_iso, datetime(date_unix, 'unixepoch'))) = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (season_id, home_team_id, away_team_id, match_day),
        ).fetchone()
        if row is None:
            return None
        return self._to_int_or_none(row["match_id"])

    def close(self) -> None:
        self.conn.close()
