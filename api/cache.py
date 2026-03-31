from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from config import settings


class JsonFileCache:
    def __init__(self, cache_dir: Path | str | None = None, ttl_sec: int | None = None) -> None:
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_sec = ttl_sec if ttl_sec is not None else settings.cache_ttl_sec

    def make_key(self, endpoint: str, params: dict[str, Any] | None = None) -> str:
        payload = json.dumps({"endpoint": endpoint, "params": params or {}}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def set(self, key: str, payload: Any) -> None:
        data = {"saved_at": int(time.time()), "payload": payload}
        self._path(key).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def get(self, key: str, allow_stale: bool = False, ttl_sec: int | None = None) -> tuple[Any | None, bool]:
        path = self._path(key)
        if not path.exists():
            return None, False

        data = json.loads(path.read_text(encoding="utf-8"))
        age_sec = int(time.time()) - int(data.get("saved_at", 0))
        effective_ttl = self.ttl_sec if ttl_sec is None else int(ttl_sec)
        is_fresh = age_sec <= effective_ttl
        if not is_fresh and not allow_stale:
            return None, False
        return data.get("payload"), is_fresh
