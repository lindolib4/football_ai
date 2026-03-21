from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    max_requests_per_hour: int = 1800
    min_interval_sec: float = 0.2
    _timestamps: deque[float] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _next_allowed: float = field(default=0.0, init=False)

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cutoff = now - 3600
                while self._timestamps and self._timestamps[0] < cutoff:
                    self._timestamps.popleft()

                delay_min_interval = max(0.0, self._next_allowed - now)
                delay_hourly = 0.0
                if len(self._timestamps) >= self.max_requests_per_hour:
                    delay_hourly = max(0.0, self._timestamps[0] + 3600 - now)

                delay = max(delay_min_interval, delay_hourly)
                if delay <= 0:
                    stamp = time.monotonic()
                    self._timestamps.append(stamp)
                    self._next_allowed = stamp + self.min_interval_sec
                    return

            time.sleep(delay)

    @property
    def requests_last_hour(self) -> int:
        with self._lock:
            now = time.monotonic()
            cutoff = now - 3600
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            return len(self._timestamps)
