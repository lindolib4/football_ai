from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    footystats_api_key: str = os.getenv("FOOTYSTATS_API_KEY", "")
    footystats_base_url: str = os.getenv("FOOTYSTATS_BASE_URL", "https://api.footystats.org")
    request_timeout_sec: int = int(os.getenv("REQUEST_TIMEOUT_SEC", "10"))
    request_retries: int = int(os.getenv("REQUEST_RETRIES", "3"))
    cache_ttl_sec: int = int(os.getenv("CACHE_TTL_SEC", str(24 * 3600)))
    db_path: Path = Path(os.getenv("DB_PATH", "football_ai/database/footai.sqlite3"))
    model_path: Path = Path(os.getenv("MODEL_PATH", "football_ai/data/models/model.pkl"))


settings = Settings()
