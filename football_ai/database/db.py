from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from football_ai.config import settings


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

    def upsert_match(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO matches (match_id, date, league, home_team, away_team, home_score, away_score, status)
            VALUES (:match_id, :date, :league, :home_team, :away_team, :home_score, :away_score, :status)
            ON CONFLICT(match_id) DO UPDATE SET
              date=excluded.date,
              league=excluded.league,
              home_team=excluded.home_team,
              away_team=excluded.away_team,
              home_score=excluded.home_score,
              away_score=excluded.away_score,
              status=excluded.status
            """,
            row,
        )
        self.conn.commit()

    def save_features(self, match_id: str, features: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO features (match_id, features_json)
            VALUES (?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
            features_json=excluded.features_json
            """,
            (match_id, json.dumps(features, ensure_ascii=False)),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
