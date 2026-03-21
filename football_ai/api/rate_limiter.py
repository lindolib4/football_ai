from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass


@dataclass
class RateLimiter:
    max_requests_per_sec: float = 2.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._queue: queue.Queue[float] = queue.Queue()

    def wait_for_slot(self) -> None:
        with self._lock:
            now = time.monotonic()
            min_interval = 1.0 / self.max_requests_per_sec
            wait_for = max(0.0, self._next_allowed - now)
            self._next_allowed = max(now, self._next_allowed) + min_interval
            self._queue.put(self._next_allowed)
        if wait_for > 0:
            time.sleep(wait_for)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
