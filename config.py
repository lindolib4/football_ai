from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
SCHEDULER_ENV_PATH = PROJECT_ROOT / "scheduler" / ".env"


def _load_project_env() -> None:
    # Prefer root .env, keep scheduler/.env as backward-compatible fallback.
    for env_path in (ENV_PATH, SCHEDULER_ENV_PATH):
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _env_path(name: str, default: str) -> Path:
    raw_value = _env_str(name, default)
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_toto_api_base_url() -> str:
    """Resolve TotoBrief base URL from supported keys with a safe default.

    Priority:
    1) TOTO_API_BASE_URL (current key)
    2) TOTOBRIEF_API_BASE_URL / TOTOBRIEF_BASE_URL / TOTOBRIEF_URL (aliases)
    3) https://totobrief.com (public host fallback)
    """
    candidates = (
        "TOTO_API_BASE_URL",
        "TOTOBRIEF_API_BASE_URL",
        "TOTOBRIEF_BASE_URL",
        "TOTOBRIEF_URL",
    )
    for key in candidates:
        value = _env_str(key)
        if value:
            return value.rstrip("/")
    return "https://totobrief.com"


_load_project_env()


@dataclass(frozen=True)
class Settings:
    footystats_api_key: str = _env_str("FOOTYSTATS_API_KEY")
    footystats_base_url: str = _env_str("FOOTYSTATS_BASE_URL", "https://api.football-data-api.com")
    toto_api_base_url: str = _resolve_toto_api_base_url()
    app_timezone: str = _env_str("APP_TIMEZONE", "Europe/Berlin")
    request_timeout_sec: int = _env_int("REQUEST_TIMEOUT_SEC", 10)
    request_retries: int = _env_int("REQUEST_RETRIES", 3)
    cache_ttl_hours: int = _env_int("CACHE_TTL_HOURS", 24)
    cache_ttl_sec: int = cache_ttl_hours * 3600
    live_cache_ttl_hours: int = _env_int("LIVE_CACHE_TTL_HOURS", 1)
    live_cache_ttl_sec: int = live_cache_ttl_hours * 3600
    log_level: str = _env_str("LOG_LEVEL", "INFO")
    db_path: Path = _env_path("DB_PATH", "database/footai.sqlite3")
    model_path: Path = _env_path("MODEL_PATH", "data/models/model.pkl")
    raw_data_dir: Path = _env_path("RAW_DATA_DIR", "data/raw")
    cache_dir: Path = _env_path("CACHE_DIR", "data/cache")


settings = Settings()

# Keep backward compatibility with code paths still reading os.getenv("TOTO_API_BASE_URL").
if settings.toto_api_base_url:
    os.environ.setdefault("TOTO_API_BASE_URL", settings.toto_api_base_url)
