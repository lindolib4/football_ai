from __future__ import annotations

import itertools

from toto.coverage import diversify_lines


def generate_coupons(decisions: list[str], limit: int = 16) -> list[list[str]]:
    pools: list[list[str]] = []
    for d in decisions:
        if len(d) == 2:
            pools.append([d[0], d[1]])
        else:
            pools.append([d])
    combinations = [list(c) for c in itertools.product(*pools)]
    return diversify_lines(combinations, limit)
