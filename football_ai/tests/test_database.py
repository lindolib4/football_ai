from __future__ import annotations

from pathlib import Path

from football_ai.database.db import Database


def test_database_init_and_upsert(tmp_path: Path) -> None:
    db = Database(db_path=tmp_path / "test.sqlite3")
    db.initialize()
    db.upsert_countries([{"id": 1, "name": "England", "raw": {"id": 1}}])
    row = db.conn.execute("SELECT id, name FROM countries WHERE id = 1").fetchone()
    assert row is not None
    assert row["name"] == "England"
    db.close()
