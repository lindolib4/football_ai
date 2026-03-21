from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from football_ai.api.rate_limiter import RateLimiter

logger = logging.getLogger("api")


@dataclass
class HttpResponse:
    payload: dict[str, Any] | None
    status_code: int | None
    from_cache: bool = False


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
        self.session = requests.Session()

    def get(self, url: str, params: dict[str, Any] | None = None) -> HttpResponse:
        params = params or {}
        last_exc: Exception | None = None
        start = time.perf_counter()

        for attempt in range(1, self.retries + 1):
            self.rate_limiter.wait_for_slot()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_sec)
                status = response.status_code

                if 500 <= status < 600:
                    raise requests.HTTPError(f"Server error {status}", response=response)

                if 400 <= status < 500:
                    elapsed_ms = (time.perf_counter() - start) * 1000
                    logger.warning(
                        "GET %s params=%s status=%s duration_ms=%.1f attempt=%s cache=miss",
                        url,
                        params,
                        status,
                        elapsed_ms,
                        attempt,
                    )
                    return HttpResponse(payload=None, status_code=status)

                payload = response.json()
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "GET %s params=%s status=%s duration_ms=%.1f attempt=%s cache=miss",
                    url,
                    params,
                    status,
                    elapsed_ms,
                    attempt,
                )
                return HttpResponse(payload=payload, status_code=status)
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError, ValueError) as exc:
                last_exc = exc
                if attempt == self.retries:
                    break
                time.sleep(2 ** (attempt - 1))

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "GET %s params=%s failed duration_ms=%.1f retries=%s error=%s",
            url,
            params,
            elapsed_ms,
            self.retries,
            last_exc,
        )
        return HttpResponse(payload=None, status_code=None)
