from __future__ import annotations

import logging
from pathlib import Path

from football_ai.config import settings


def configure_logging(log_dir: str = "football_ai/logs") -> None:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    for logger_name in ["api", "database", "training", "prediction", "toto", "ui", "scheduler"]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(path / f"{logger_name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.propagate = True
