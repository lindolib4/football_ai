from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FeatureSchema:
    """Persistence and strict validation for model feature names."""

    def save(self, feature_names: list[str], path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as fp:
            json.dump([str(name) for name in feature_names], fp, ensure_ascii=False, indent=2)
        logger.info("Feature schema saved to %s with %d features", destination, len(feature_names))

    def load(self, path: str) -> list[str]:
        schema_path = Path(path)
        if not schema_path.exists():
            msg = f"Feature schema file not found: {schema_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        with schema_path.open("r", encoding="utf-8") as fp:
            payload: Any = json.load(fp)

        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            msg = f"Feature schema must be a JSON array of strings: {schema_path}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Feature schema loaded from %s with %d features", schema_path, len(payload))
        return payload

    def validate(self, features: dict, expected: list[str]) -> None:
        incoming = set(features.keys())
        expected_set = set(expected)

        missing = sorted(expected_set - incoming)
        extra = sorted(incoming - expected_set)

        if missing or extra:
            msg = f"Feature schema mismatch. Missing: {missing}. Extra: {extra}."
            logger.error(msg)
            raise ValueError(msg)
