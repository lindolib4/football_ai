from __future__ import annotations


def pick_doubles_by_entropy(entropies: list[tuple[str, float]], top_n: int = 5) -> set[str]:
    ordered = sorted(entropies, key=lambda x: x[1], reverse=True)
    return {match_id for match_id, _ in ordered[:top_n]}
