from __future__ import annotations

from typing import Iterable


def safe_mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def calc_form_index(winrate: float, goals_scored: float, goals_conceded: float) -> float:
    return round((0.6 * winrate) + (0.3 * goals_scored) - (0.1 * goals_conceded), 4)
