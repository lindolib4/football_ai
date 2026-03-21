from __future__ import annotations

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AutoTrainer:
    def __init__(self) -> None:
        self.last_train_time: datetime | None = None

    def should_run(self) -> bool:
        if self.last_train_time is None:
            return True
        return datetime.utcnow() - self.last_train_time >= timedelta(hours=24)

    def mark_done(self) -> None:
        self.last_train_time = datetime.utcnow()
        logger.info("Auto training completed at %s", self.last_train_time.isoformat())
