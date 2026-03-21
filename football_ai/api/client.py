from __future__ import annotations

import logging
import time
from typing import Any

import requests

from football_ai.api.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class HttpClient:
    def __init__(
        self,
        timeout_sec: int = 10,
        retries: int = 3,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.rate_limiter = rate_limiter or RateLimiter()

    def get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self.rate_limiter.wait_for_slot()
            try:
                response = requests.get(url, params=params, timeout=self.timeout_sec)
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                sleep_sec = 2 ** (attempt - 1)
                logger.warning("GET failed (%s/%s): %s", attempt, self.retries, exc)
                time.sleep(sleep_sec)
        logger.error("GET failed after retries: %s", last_error)
        return None
