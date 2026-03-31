from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toto.optimizer import TotoOptimizer


STAKE = "30"


def build_matches() -> list[dict[str, Any]]:
    rows = [
        ("A", "B", 0.52, 0.25, 0.23),
        ("C", "D", 0.46, 0.31, 0.23),
        ("E", "F", 0.43, 0.27, 0.30),
        ("G", "H", 0.41, 0.34, 0.25),
        ("I", "J", 0.49, 0.24, 0.27),
        ("K", "L", 0.39, 0.29, 0.32),
        ("M", "N", 0.44, 0.30, 0.26),
        ("O", "P", 0.37, 0.33, 0.30),
        ("Q", "R", 0.47, 0.21, 0.32),
        ("S", "T", 0.42, 0.28, 0.30),
        ("U", "V", 0.45, 0.26, 0.29),
        ("W", "X", 0.40, 0.31, 0.29),
        ("Y", "Z", 0.38, 0.30, 0.32),
        ("AA", "BB", 0.50, 0.22, 0.28),
        ("CC", "DD", 0.36, 0.35, 0.29),
    ]
    matches: list[dict[str, Any]] = []
    for idx, (home, away, p1, px, p2) in enumerate(rows, start=1):
        matches.append(
            {
                "match_id": f"m{idx}",
                "home": home,
                "away": away,
                "name": f"{home} vs {away}",
                "probs": {"P1": p1, "PX": px, "P2": p2},
            }
        )
    return matches


def build_history_context() -> dict[str, Any]:
    return {
        "history_draws_loaded_count": 24,
        "history_events_loaded_count": 96,
        "stats": {
            "events_count": 96,
            "upset_rate": 0.42,
            "draw_results_rate": 0.34,
            "pool_vs_bookmaker_gap": 0.11,
        },
    }


def to_coupon_lines(coupons: list[list[str]]) -> list[str]:
    return [";".join([STAKE] + row) for row in coupons]


def cell_diff_count(a: list[list[str]], b: list[list[str]]) -> int:
    total = 0
    for i in range(min(len(a), len(b))):
        row_a = a[i]
        row_b = b[i]
        for j in range(min(len(row_a), len(row_b))):
            if row_a[j] != row_b[j]:
                total += 1
    return total


def run_case(mode: str, strength: float | None, history_on: bool) -> dict[str, Any]:
    matches = build_matches()
    opt = TotoOptimizer()
    history = build_history_context() if history_on else None

    if strength is None:
        coupons = opt.optimize(matches=matches, mode=mode, global_history_context=history)
        insurance_enabled = False
        strength_value = 0.0
    else:
        coupons = opt.optimize_insurance(
            matches=matches,
            mode=mode,
            insurance_strength=strength,
            global_history_context=history,
        )
        insurance_enabled = strength > 0.0
        strength_value = float(strength)

    summary = opt.last_run_summary if isinstance(opt.last_run_summary, dict) else {}
    diagnostics = summary.get("insurance_diagnostics", {}) if isinstance(summary.get("insurance_diagnostics"), dict) else {}

    return {
        "mode": mode,
        "insurance_enabled": insurance_enabled,
        "insurance_strength": strength_value,
        "history_on": history_on,
        "coupon_count": len(coupons),
        "insured_coupons_count": int(summary.get("insured_coupons_count", 0) or 0),
        "changed_cells": int(summary.get("insurance_cells_changed_count", 0) or 0),
        "matches_changed_by_insurance": len(list(summary.get("changed_matches_by_insurance", []))),
        "target_match_indexes": list(summary.get("target_match_indexes", [])),
        "target_match_labels": list(summary.get("target_match_labels", [])),
        "history_influenced_matches": int(summary.get("history_influenced_matches_count", 0) or 0),
        "history_events_loaded_count": int(summary.get("history_events_loaded_count", 0) or 0),
        "diagnostics": diagnostics,
        "coupon_lines": to_coupon_lines(coupons),
        "coupons": coupons,
    }


def main() -> None:
    scenarios: list[dict[str, Any]] = []
    for mode in ("16", "32"):
        for history_on in (False, True):
            scenarios.append(run_case(mode=mode, strength=None, history_on=history_on))
            scenarios.append(run_case(mode=mode, strength=0.7, history_on=history_on))
            scenarios.append(run_case(mode=mode, strength=0.9, history_on=history_on))

    index = {(s["mode"], s["history_on"], s["insurance_strength"]): s for s in scenarios}

    comparisons: list[dict[str, Any]] = []
    for mode in ("16", "32"):
        for history_on in (False, True):
            off = index[(mode, history_on, 0.0)]
            s07 = index[(mode, history_on, 0.7)]
            s09 = index[(mode, history_on, 0.9)]

            set_07 = {tuple(row) for row in s07["coupons"]}
            set_09 = {tuple(row) for row in s09["coupons"]}
            diff_lines = len(set_07.symmetric_difference(set_09))

            comparisons.append(
                {
                    "mode": mode,
                    "history_on": history_on,
                    "off_vs_07_changed_cells_delta": int(s07["changed_cells"] - off["changed_cells"]),
                    "off_vs_09_changed_cells_delta": int(s09["changed_cells"] - off["changed_cells"]),
                    "strength_07_vs_09_changed_cells_delta": int(s09["changed_cells"] - s07["changed_cells"]),
                    "strength_07_vs_09_coupon_set_diff_lines": diff_lines,
                    "strength_07_vs_09_coupon_cell_diff": cell_diff_count(s07["coupons"], s09["coupons"]),
                    "strength_07_target_matches": list(s07["target_match_indexes"]),
                    "strength_09_target_matches": list(s09["target_match_indexes"]),
                    "history_effect_on_09_target_matches": list(s09["target_match_indexes"]),
                }
            )

    output = {
        "scenarios": [
            {
                **{k: v for k, v in scenario.items() if k not in {"coupons", "coupon_lines"}},
                "coupon_line_sample": scenario["coupon_lines"][:3],
            }
            for scenario in scenarios
        ],
        "comparisons": comparisons,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
