from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class JsonFileCache:
    def __init__(self, cache_dir: str = "football_ai/data/raw", ttl_sec: int = 24 * 3600) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_sec = ttl_sec

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def set(self, key: str, payload: dict[str, Any]) -> None:
        data = {"saved_at": int(time.time()), "payload": payload}
        self._path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(time.time()) - int(data.get("saved_at", 0)) > self.ttl_sec:
            return None
        return data.get("payload")
