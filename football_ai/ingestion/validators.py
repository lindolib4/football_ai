from __future__ import annotations

from typing import Any


TRAIN_EXCLUDED_STATUSES = {"incomplete", "suspended", "cancelled", "canceled"}


def is_missing_value(value: Any) -> bool:
    return value in (None, "", -1, -2, "-1", "-2")


def normalize_missing(value: Any) -> Any:
    return None if is_missing_value(value) else value


def is_trainable_match(status: str | None, winning_team_id: int | None) -> bool:
    if not status:
        return False
    if status.lower() in TRAIN_EXCLUDED_STATUSES:
        return False
    return winning_team_id is not None
