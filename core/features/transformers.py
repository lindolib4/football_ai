from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Iterable


def handle_missing(features: Mapping[str, Any] | None, default: float = 0.0) -> dict[str, float]:
    """Replace missing/invalid feature values with a default float."""
    if not features:
        return {}

    normalized: dict[str, float] = {}
    for key, value in features.items():
        if value in (None, "", "nan", "NaN"):
            normalized[key] = float(default)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float(default)
        if number != number:  # NaN guard
            number = float(default)
        normalized[key] = number
    return normalized


def normalize_features(features: Mapping[str, Any] | None) -> dict[str, float]:
    """Min-max scale numeric features to [0, 1] with safe fallbacks."""
    values = handle_missing(features)
    if not values:
        return {}

    min_value = min(values.values())
    max_value = max(values.values())
    spread = max_value - min_value
    if spread == 0:
        return {key: 0.0 for key in values}

    return {key: (value - min_value) / spread for key, value in values.items()}


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def calc_form_index(winrate: float, goals_scored: float, goals_conceded: float) -> float:
    return round((0.6 * winrate) + (0.3 * goals_scored) - (0.1 * goals_conceded), 4)
