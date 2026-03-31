from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.toto_api import TotoAPI
from scheduler.auto_train import AutoTrainer
from toto.optimizer import TotoOptimizer


ALLOWED = ("1", "X", "2")
SCENARIOS: list[tuple[str, float]] = [
    ("16", 0.0),
    ("16", 0.7),
    ("16", 0.9),
    ("32", 0.0),
    ("32", 0.7),
    ("32", 0.9),
]


@dataclass
class ScenarioResult:
    draw_id: int
    history_on: bool
    mode: str
    insurance_strength: float
    coupons: list[list[str]]
    summary: dict[str, Any]
    matches: list[dict[str, Any]]
    diversity: dict[str, Any]
    consistency: dict[str, Any]
    insurance_quality: dict[str, Any]
    toto_summary: dict[str, Any]
    toto_behavior: dict[str, Any]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_runtime_noise_suppression(suppress_warning_spam: bool = True) -> None:
    if suppress_warning_spam:
        # Common sklearn warning during repeated predict calls in audit loops.
        warnings.filterwarnings(
            "ignore",
            message=r".*does not have valid feature names.*",
            category=UserWarning,
        )
        # Keep stdout clean: reduce very verbose third-party logger noise.
        logging.getLogger("lightgbm").setLevel(logging.ERROR)
        logging.getLogger("sklearn").setLevel(logging.ERROR)


def _draw_files(limit: int) -> list[Path]:
    draws_dir = REPO_ROOT / "data" / "toto_draws"
    files: list[tuple[int, Path]] = []
    for path in draws_dir.glob("*.json"):
        name = path.stem
        if not name.isdigit():
            continue
        draw_id = int(name)
        # 101 is a legacy fixture-like payload; prefer modern real draws.
        if draw_id <= 1000:
            continue
        files.append((draw_id, path))
    files.sort(key=lambda item: item[0], reverse=True)
    return [p for _, p in files[:limit]]


def _normalize_draw(draw_file: Path, api: TotoAPI) -> tuple[int, dict[str, Any]]:
    raw = _read_json(draw_file)
    draw_id = int(draw_file.stem)
    payload = raw.get("data", raw)
    normalized = api._normalize_draw(draw_id, payload)
    return draw_id, normalized


def _prepare_matches(
    draw: dict[str, Any],
    trainer: AutoTrainer,
    history_payload: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for match in draw.get("matches", []):
        pool_probs = match.get("pool_probs") if isinstance(match.get("pool_probs"), dict) else None
        if pool_probs is None:
            pool_probs = {"P1": 1.0 / 3.0, "PX": 1.0 / 3.0, "P2": 1.0 / 3.0}

        inf = trainer.prepare_toto_match_for_inference(match=match, pool_probs=pool_probs)
        probs = inf.get("probs", pool_probs)
        model_probs = inf.get("model_probs") if isinstance(inf.get("model_probs"), dict) else None
        resolved_history = history_payload if isinstance(history_payload, dict) else None

        payload_pool_quotes = match.get("pool_quotes") if isinstance(match.get("pool_quotes"), dict) else {}
        payload_bk_quotes = match.get("bookmaker_quotes") if isinstance(match.get("bookmaker_quotes"), dict) else {}

        if isinstance(match.get("norm_bookmaker_probs"), dict):
            bk_probs = match.get("norm_bookmaker_probs", {})
        elif isinstance(match.get("bookmaker_probs"), dict):
            bk_probs = match.get("bookmaker_probs", {})
        elif payload_bk_quotes:
            bk_probs = {
                "P1": _safe_float(payload_bk_quotes.get("bk_win_1")) / 100.0,
                "PX": _safe_float(payload_bk_quotes.get("bk_draw")) / 100.0,
                "P2": _safe_float(payload_bk_quotes.get("bk_win_2")) / 100.0,
            }
        else:
            bk_probs = {}

        out.append(
            {
                "name": str(match.get("name") or f"{match.get('home', '')} vs {match.get('away', '')}"),
                "home": str(match.get("home", "")),
                "away": str(match.get("away", "")),
                "probs": probs,
                "model_probs": model_probs,
                "pool_probs": inf.get("pool_probs", pool_probs),
                "pool_quotes": payload_pool_quotes,
                "bookmaker_quotes": payload_bk_quotes,
                "bookmaker_probs": bk_probs,
                "norm_bookmaker_probs": bk_probs,
                "toto_brief_history": resolved_history,
                "prob_source": str(inf.get("source", "pool_context_only")),
                "fallback_reason": inf.get("fallback_reason"),
                "features": inf.get("runtime_features", {}) if isinstance(inf.get("runtime_features"), dict) else {},
                "feature_context_level": str(inf.get("feature_context_level") or "unknown"),
            }
        )

    return out


def _hamming(a: list[str], b: list[str]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _coupon_diversity(coupons: list[list[str]]) -> dict[str, Any]:
    unique_count = len({tuple(c) for c in coupons})
    dists = [_hamming(coupons[i], coupons[j]) for i, j in combinations(range(len(coupons)), 2)]
    if dists:
        avg_dist = float(sum(dists) / len(dists))
        min_dist = int(min(dists))
        med_dist = float(statistics.median(dists))
        dist_1 = sum(1 for d in dists if d == 1)
        dist_2p = sum(1 for d in dists if d >= 2)
    else:
        avg_dist = 0.0
        min_dist = 0
        med_dist = 0.0
        dist_1 = 0
        dist_2p = 0

    return {
        "total_coupon_count": len(coupons),
        "unique_coupon_count": unique_count,
        "average_hamming_distance": avg_dist,
        "min_hamming_distance": min_dist,
        "median_hamming_distance": med_dist,
        "pairs_diff_1": dist_1,
        "pairs_diff_2_plus": dist_2p,
    }


def _consistency_checks(coupons: list[list[str]], summary: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []

    base_count = _safe_int(summary.get("base_coupon_count", summary.get("base_coupons_count", len(coupons))))
    ins_count = _safe_int(summary.get("insurance_coupon_count", summary.get("insured_coupons_count", 0)))
    insured_matches = _safe_int(summary.get("insurance_target_matches_count", 0))
    changed_matches = _safe_int(summary.get("insurance_changed_matches_count", 0))
    changed_lines = _safe_int(summary.get("insurance_changed_coupon_lines_count", summary.get("affected_coupon_lines_count", 0)))
    hist_infl = _safe_int(summary.get("history_influenced_matches_count", 0))
    strategy_adjusted = _safe_int(summary.get("strategy_adjusted_matches_count", 0))

    actual_ins_count = max(len(coupons) - max(base_count, 0), 0)

    entries = summary.get("coupon_entries") if isinstance(summary.get("coupon_entries"), list) else []
    actual_changed_lines = 0
    if entries:
        for row in entries:
            if not isinstance(row, dict):
                continue
            if str(row.get("coupon_type", "")) != "insurance":
                continue
            changed_positions = row.get("changed_positions", [])
            if isinstance(changed_positions, list) and len(changed_positions) > 0:
                actual_changed_lines += 1

    changed_by_ins = summary.get("changed_matches_by_insurance")
    if isinstance(changed_by_ins, list):
        actual_changed_matches = len({int(x) for x in changed_by_ins if isinstance(x, int)})
    else:
        actual_changed_matches = 0

    target_indexes = summary.get("target_match_indexes")
    actual_insured_matches = len(target_indexes) if isinstance(target_indexes, list) else 0

    mld = summary.get("match_level_decision_diagnostics") if isinstance(summary.get("match_level_decision_diagnostics"), list) else []
    actual_hist_infl = sum(1 for row in mld if isinstance(row, dict) and bool(row.get("history_used", False)))
    actual_strategy_adjusted = sum(
        1
        for row in mld
        if isinstance(row, dict)
        and str(row.get("model_decision", ""))
        and str(row.get("strategy_adjusted_decision", ""))
        and str(row.get("model_decision", "")) != str(row.get("strategy_adjusted_decision", ""))
    )

    if ins_count != actual_ins_count:
        mismatches.append(f"insurance_coupon_count summary={ins_count} actual={actual_ins_count}")
    if changed_lines != actual_changed_lines:
        mismatches.append(f"insurance_changed_coupon_lines_count summary={changed_lines} actual={actual_changed_lines}")
    if changed_matches != actual_changed_matches:
        mismatches.append(f"insurance_changed_matches_count summary={changed_matches} actual={actual_changed_matches}")
    if insured_matches != actual_insured_matches:
        mismatches.append(f"insured_matches_count summary={insured_matches} actual={actual_insured_matches}")
    if hist_infl != actual_hist_infl:
        mismatches.append(f"history_influenced_matches_count summary={hist_infl} actual={actual_hist_infl}")
    if strategy_adjusted != actual_strategy_adjusted:
        mismatches.append(f"strategy_adjusted_matches_count summary={strategy_adjusted} actual={actual_strategy_adjusted}")

    return {
        "ok": len(mismatches) == 0,
        "mismatches": mismatches,
        "summary_values": {
            "insurance_coupon_count": ins_count,
            "insurance_changed_coupon_lines_count": changed_lines,
            "insurance_changed_matches_count": changed_matches,
            "insured_matches_count": insured_matches,
            "history_influenced_matches_count": hist_infl,
            "strategy_adjusted_matches_count": strategy_adjusted,
        },
        "actual_values": {
            "insurance_coupon_count": actual_ins_count,
            "insurance_changed_coupon_lines_count": actual_changed_lines,
            "insurance_changed_matches_count": actual_changed_matches,
            "insured_matches_count": actual_insured_matches,
            "history_influenced_matches_count": actual_hist_infl,
            "strategy_adjusted_matches_count": actual_strategy_adjusted,
        },
    }


def _closest_base_coupon(target: list[str], base_coupons: list[list[str]]) -> tuple[int, list[str] | None]:
    best_dist = 10**9
    best_coupon: list[str] | None = None
    for base in base_coupons:
        d = _hamming(target, base)
        if d < best_dist:
            best_dist = d
            best_coupon = base
    return best_dist, best_coupon


def _classify_change(base_out: str, alt_out: str, probs: dict[str, float]) -> str:
    rank = sorted(ALLOWED, key=lambda o: _safe_float(probs.get({"1": "P1", "X": "PX", "2": "P2"}[o])), reverse=True)
    if len(rank) < 3:
        return "cosmetic_shift"
    top, second, third = rank[0], rank[1], rank[2]
    if base_out == top and alt_out == second:
        return "adjacent_alternative"
    if base_out == top and alt_out == third:
        return "stronger_opposite"
    return "cosmetic_shift"


def _insurance_quality(
    coupons: list[list[str]],
    summary: dict[str, Any],
    matches: list[dict[str, Any]],
    base_coupons_for_compare: list[list[str]],
) -> dict[str, Any]:
    base_count = _safe_int(summary.get("base_coupon_count", summary.get("base_coupons_count", len(coupons))))
    insurance_coupons = coupons[base_count:]

    meaningful_lines = 0
    changed_lines = 0
    change_counter: Counter[str] = Counter()

    if insurance_coupons and base_coupons_for_compare:
        for ins_row in insurance_coupons:
            dist, nearest = _closest_base_coupon(ins_row, base_coupons_for_compare)
            if nearest is None:
                continue
            if dist > 0:
                changed_lines += 1
            if dist >= 2:
                meaningful_lines += 1
            for idx, (a, b) in enumerate(zip(ins_row, nearest)):
                if a == b:
                    continue
                if idx >= len(matches):
                    continue
                cls = _classify_change(base_out=b, alt_out=a, probs=matches[idx].get("probs", {}))
                change_counter[cls] += 1

    mld = summary.get("match_layer_diagnostics") if isinstance(summary.get("match_layer_diagnostics"), list) else []
    reason_counter: Counter[str] = Counter()
    for row in mld:
        if not isinstance(row, dict):
            continue
        codes = row.get("insurance_reason_codes", [])
        if isinstance(codes, list):
            for code in codes:
                reason_counter[str(code)] += 1

    target_match_indexes = summary.get("target_match_indexes") if isinstance(summary.get("target_match_indexes"), list) else []
    changed_matches = summary.get("changed_matches_by_insurance") if isinstance(summary.get("changed_matches_by_insurance"), list) else []

    return {
        "insurance_target_matches_count": len(target_match_indexes),
        "insurance_changed_matches_count": len({int(x) for x in changed_matches if isinstance(x, int)}),
        "insurance_changed_coupon_lines_count": _safe_int(summary.get("insurance_changed_coupon_lines_count", summary.get("affected_coupon_lines_count", 0))),
        "insurance_visible_changed_lines": changed_lines,
        "meaningful_changed_lines": meaningful_lines,
        "meaningful_ratio": (meaningful_lines / len(insurance_coupons)) if insurance_coupons else 0.0,
        "change_semantics": dict(change_counter),
        "insurance_reason_codes": dict(reason_counter),
    }


def _distribution_dict(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        out[str(key)] = _safe_int(value)
    return out


def _summary_slice(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "probability_source": summary.get("probability_source"),
        "history_state_label": summary.get("history_state_label"),
        "history_influenced_matches_count": _safe_int(summary.get("history_influenced_matches_count", 0)),
        "strategy_adjusted_matches_count": _safe_int(summary.get("strategy_adjusted_matches_count", 0)),
        "insurance_target_matches_count": _safe_int(summary.get("insurance_target_matches_count", 0)),
        "insurance_changed_matches_count": _safe_int(summary.get("insurance_changed_matches_count", 0)),
        "target_match_indexes": list(summary.get("target_match_indexes", [])) if isinstance(summary.get("target_match_indexes"), list) else [],
        "match_type_distribution": _distribution_dict(summary.get("match_type_distribution", {})),
        "risk_level_distribution": _distribution_dict(summary.get("risk_level_distribution", {})),
        "resolution_type_distribution": _distribution_dict(summary.get("resolution_type_distribution", {})),
        "avg_draw_risk_score": _safe_float(summary.get("avg_draw_risk_score", 0.0)),
        "avg_upset_risk_score": _safe_float(summary.get("avg_upset_risk_score", 0.0)),
        "avg_weak_favorite_score": _safe_float(summary.get("avg_weak_favorite_score", 0.0)),
        "avg_trap_match_score": _safe_float(summary.get("avg_trap_match_score", 0.0)),
        "avg_single_confidence_score": _safe_float(summary.get("avg_single_confidence_score", 0.0)),
        "avg_insurance_priority_score": _safe_float(summary.get("avg_insurance_priority_score", 0.0)),
    }


def _toto_behavior(summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary.get("match_layer_diagnostics") if isinstance(summary.get("match_layer_diagnostics"), list) else []
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {
        "matches": 0,
        "strategy_changed": 0,
        "double_resolution": 0,
        "insurance_priority_resolution": 0,
        "single_resolution": 0,
        "toto_layer_used": 0,
        "double_recommendation": 0,
        "insurance_cover_added": 0,
    })

    totals = {
        "matches": 0,
        "strategy_changed": 0,
        "double_resolution": 0,
        "insurance_priority_resolution": 0,
        "single_resolution": 0,
        "toto_layer_used": 0,
        "double_recommendation": 0,
        "insurance_cover_added": 0,
    }

    for row in rows:
        if not isinstance(row, dict):
            continue
        match_type = str(row.get("match_type", "unknown"))
        resolution_type = str(row.get("resolution_type", "single"))
        bucket = by_type[match_type]

        bucket["matches"] += 1
        totals["matches"] += 1

        if bool(row.get("strategy_changed_flag", False)):
            bucket["strategy_changed"] += 1
            totals["strategy_changed"] += 1
        if bool(row.get("toto_layer_used", False)):
            bucket["toto_layer_used"] += 1
            totals["toto_layer_used"] += 1
        if bool(row.get("double_recommendation_flag", False)):
            bucket["double_recommendation"] += 1
            totals["double_recommendation"] += 1
        if bool(row.get("insurance_changed_flag", False)):
            bucket["insurance_cover_added"] += 1
            totals["insurance_cover_added"] += 1

        if resolution_type == "double":
            bucket["double_resolution"] += 1
            totals["double_resolution"] += 1
        elif resolution_type == "insurance priority":
            bucket["insurance_priority_resolution"] += 1
            totals["insurance_priority_resolution"] += 1
        else:
            bucket["single_resolution"] += 1
            totals["single_resolution"] += 1

    return {
        "totals": totals,
        "by_match_type": {key: dict(value) for key, value in by_type.items()},
    }


def _run_scenario(
    draw_id: int,
    matches: list[dict[str, Any]],
    mode: str,
    strength: float,
    history_on: bool,
    history_payload: dict[str, Any] | None,
) -> tuple[list[list[str]], dict[str, Any]]:
    optimizer = TotoOptimizer()
    if strength <= 0.0:
        coupons = optimizer.optimize(matches=matches, mode=mode, global_history_context=history_payload if history_on else None)
    else:
        coupons = optimizer.optimize_insurance(
            matches=matches,
            mode=mode,
            insurance_strength=strength,
            global_history_context=history_payload if history_on else None,
        )
    summary = optimizer.last_run_summary if isinstance(optimizer.last_run_summary, dict) else {}
    if not isinstance(summary, dict):
        summary = {}
    summary["_draw_id"] = draw_id
    summary["_history_on"] = history_on
    summary["_mode"] = mode
    summary["_insurance_strength"] = strength
    return coupons, summary


def _aggregate(results: list[ScenarioResult]) -> dict[str, Any]:
    grouped: dict[str, list[ScenarioResult]] = defaultdict(list)
    for item in results:
        key = f"mode={item.mode}|insurance={item.insurance_strength:.1f}|history={'on' if item.history_on else 'off'}"
        grouped[key].append(item)

    aggregated: dict[str, Any] = {}
    for key, rows in grouped.items():
        agg: dict[str, Any] = {"runs": len(rows)}
        for metric_name in (
            "total_coupon_count",
            "base_coupon_count",
            "insurance_coupon_count",
            "unique_coupon_count",
            "average_hamming_distance",
            "min_hamming_distance",
            "median_hamming_distance",
            "pairs_diff_1",
            "pairs_diff_2_plus",
            "insurance_target_matches_count",
            "insurance_changed_matches_count",
            "insurance_changed_coupon_lines_count",
            "meaningful_changed_lines",
            "meaningful_ratio",
        ):
            vals: list[float] = []
            for r in rows:
                src = {}
                src.update(r.diversity)
                src.update(r.insurance_quality)
                if metric_name in src:
                    vals.append(float(src[metric_name]))
            if vals:
                agg[f"avg_{metric_name}"] = float(sum(vals) / len(vals))

        for metric_name in (
            "strategy_adjusted_matches_count",
            "history_influenced_matches_count",
            "insurance_target_matches_count",
            "insurance_changed_matches_count",
            "avg_draw_risk_score",
            "avg_upset_risk_score",
            "avg_weak_favorite_score",
            "avg_trap_match_score",
            "avg_single_confidence_score",
            "avg_insurance_priority_score",
        ):
            vals = [float(r.toto_summary.get(metric_name, 0.0)) for r in rows]
            if vals:
                agg[metric_name if metric_name.startswith("avg_") else f"avg_{metric_name}"] = float(sum(vals) / len(vals))

        dist_fields = (
            "match_type_distribution",
            "risk_level_distribution",
            "resolution_type_distribution",
        )
        for dist_field in dist_fields:
            merged_counter: Counter[str] = Counter()
            for r in rows:
                merged_counter.update(r.toto_summary.get(dist_field, {}))
            agg[dist_field] = dict(merged_counter)

        effect_counter: Counter[str] = Counter()
        by_type_effect: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            effect_counter.update(r.toto_behavior.get("totals", {}))
            for match_type, metrics in r.toto_behavior.get("by_match_type", {}).items():
                if isinstance(metrics, dict):
                    by_type_effect[str(match_type)].update(metrics)
        agg["behavior_totals"] = dict(effect_counter)
        agg["behavior_by_match_type"] = {key: dict(value) for key, value in by_type_effect.items()}

        agg["consistency_failures"] = sum(1 for r in rows if not r.consistency.get("ok", False))
        aggregated[key] = agg

    return aggregated


def _history_delta(results: list[ScenarioResult]) -> dict[str, Any]:
    index: dict[tuple[int, str, float, bool], ScenarioResult] = {
        (r.draw_id, r.mode, r.insurance_strength, r.history_on): r for r in results
    }
    out: dict[str, Any] = {}

    for draw_id, mode, strength, history_on in list(index.keys()):
        if history_on:
            continue
        left = index.get((draw_id, mode, strength, False))
        right = index.get((draw_id, mode, strength, True))
        if left is None or right is None:
            continue
        key = f"draw={draw_id}|mode={mode}|insurance={strength:.1f}"

        left_targets = set(left.summary.get("target_match_indexes", [])) if isinstance(left.summary.get("target_match_indexes"), list) else set()
        right_targets = set(right.summary.get("target_match_indexes", [])) if isinstance(right.summary.get("target_match_indexes"), list) else set()

        left_coupons = {tuple(c) for c in left.coupons}
        right_coupons = {tuple(c) for c in right.coupons}

        out[key] = {
            "history_influenced_matches_off": _safe_int(left.summary.get("history_influenced_matches_count", 0)),
            "history_influenced_matches_on": _safe_int(right.summary.get("history_influenced_matches_count", 0)),
            "strategy_adjusted_matches_off": _safe_int(left.summary.get("strategy_adjusted_matches_count", 0)),
            "strategy_adjusted_matches_on": _safe_int(right.summary.get("strategy_adjusted_matches_count", 0)),
            "target_matches_symdiff": len(left_targets.symmetric_difference(right_targets)),
            "final_coupon_set_symdiff": len(left_coupons.symmetric_difference(right_coupons)),
            "probability_source_off": str(left.summary.get("probability_source", "")),
            "probability_source_on": str(right.summary.get("probability_source", "")),
            "history_state_label_off": str(left.summary.get("history_state_label", "")),
            "history_state_label_on": str(right.summary.get("history_state_label", "")),
        }

    return out


def _run_audit(draw_limit: int, strict: bool = False) -> dict[str, Any]:
    api = TotoAPI()
    trainer = AutoTrainer()
    trainer.load()

    history_path = REPO_ROOT / "data" / "toto_draws" / "history_baltbet-main.json"
    history_payload = _read_json(history_path) if history_path.exists() else None

    draw_files = _draw_files(limit=draw_limit)
    if not draw_files:
        raise RuntimeError("No draw files found for real-data audit")

    results: list[ScenarioResult] = []
    failures: list[dict[str, Any]] = []

    for draw_file in draw_files:
        draw_id = -1
        try:
            draw_id, normalized_draw = _normalize_draw(draw_file, api)

            prepared_without_history = _prepare_matches(normalized_draw, trainer=trainer, history_payload=None)
            prepared_with_history = _prepare_matches(normalized_draw, trainer=trainer, history_payload=history_payload)

            if not prepared_without_history and not prepared_with_history:
                failures.append(
                    {
                        "draw_id": draw_id,
                        "error": "no_prepared_matches",
                    }
                )
                continue

            base_map: dict[tuple[bool, str], list[list[str]]] = {}

            for history_on, prepared in ((False, prepared_without_history), (True, prepared_with_history)):
                for mode, strength in SCENARIOS:
                    coupons, summary = _run_scenario(
                        draw_id=draw_id,
                        matches=prepared,
                        mode=mode,
                        strength=strength,
                        history_on=history_on,
                        history_payload=history_payload,
                    )
                    if strength == 0.0:
                        base_map[(history_on, mode)] = coupons

                    diversity = _coupon_diversity(coupons)
                    base_coupon_count = _safe_int(summary.get("base_coupon_count", summary.get("base_coupons_count", len(coupons))))
                    insurance_coupon_count = _safe_int(summary.get("insurance_coupon_count", summary.get("insured_coupons_count", 0)))
                    diversity["base_coupon_count"] = base_coupon_count
                    diversity["insurance_coupon_count"] = insurance_coupon_count

                    consistency = _consistency_checks(coupons, summary)
                    insurance_quality = _insurance_quality(
                        coupons=coupons,
                        summary=summary,
                        matches=prepared,
                        base_coupons_for_compare=base_map.get((history_on, mode), coupons),
                    )
                    toto_summary = _summary_slice(summary)
                    toto_behavior = _toto_behavior(summary)

                    results.append(
                        ScenarioResult(
                            draw_id=draw_id,
                            history_on=history_on,
                            mode=mode,
                            insurance_strength=strength,
                            coupons=coupons,
                            summary=summary,
                            matches=prepared,
                            diversity=diversity,
                            consistency=consistency,
                            insurance_quality=insurance_quality,
                            toto_summary=toto_summary,
                            toto_behavior=toto_behavior,
                        )
                    )
        except Exception as exc:
            failures.append(
                {
                    "draw_id": draw_id if draw_id > 0 else None,
                    "draw_file": str(draw_file),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if strict:
                raise

    raw_rows: list[dict[str, Any]] = []
    for item in results:
        raw_rows.append(
            {
                "draw_id": item.draw_id,
                "history_on": item.history_on,
                "mode": item.mode,
                "insurance_strength": item.insurance_strength,
                "diversity": item.diversity,
                "insurance_quality": item.insurance_quality,
                "consistency": item.consistency,
                "summary_slice": item.toto_summary,
                "toto_behavior": item.toto_behavior,
            }
        )

    return {
        "draw_ids": sorted({r.draw_id for r in results}),
        "draw_count": len(sorted({r.draw_id for r in results})),
        "draw_requested_count": len(draw_files),
        "draw_processed_count": len(sorted({r.draw_id for r in results})),
        "draw_failed_count": len(failures),
        "draw_failures": failures,
        "scenario_runs": len(results),
        "aggregated": _aggregate(results),
        "history_delta": _history_delta(results),
        "rows": raw_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-data TOTO quality audit")
    parser.add_argument("--draw-limit", type=int, default=5, help="How many latest real draws to evaluate")
    parser.add_argument("--out", type=str, default="reports/toto_quality_audit_real.json", help="Output JSON path")
    parser.add_argument(
        "--show-warning-spam",
        action="store_true",
        help="Do not suppress common warning spam in stdout",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on first draw-level exception instead of continuing",
    )
    args = parser.parse_args()

    _configure_runtime_noise_suppression(suppress_warning_spam=not args.show_warning_spam)

    report = _run_audit(draw_limit=max(1, args.draw_limit), strict=bool(args.strict))
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved_report={out_path}")
    print(f"draw_count={report['draw_count']}")
    print(f"draw_requested_count={report['draw_requested_count']}")
    print(f"draw_failed_count={report['draw_failed_count']}")
    print(f"scenario_runs={report['scenario_runs']}")


if __name__ == "__main__":
    main()
