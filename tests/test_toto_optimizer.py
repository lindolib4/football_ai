from __future__ import annotations

from toto.optimizer import TotoOptimizer


def _match(p1: float, px: float, p2: float, decision: str) -> dict:
    return {
        "probs": {"P1": p1, "PX": px, "P2": p2},
        "decision": decision,
    }


def _build_matches() -> list[dict]:
    return [
        _match(0.70, 0.20, 0.10, "1"),
        _match(0.34, 0.33, 0.33, "1X"),
        _match(0.68, 0.21, 0.11, "1"),
        _match(0.33, 0.34, 0.33, "X2"),
        _match(0.64, 0.22, 0.14, "1"),
        _match(0.33, 0.33, 0.34, "12"),
        _match(0.62, 0.23, 0.15, "1"),
        _match(0.35, 0.32, 0.33, "1X"),
        _match(0.35, 0.33, 0.32, "12"),
        _match(0.66, 0.20, 0.14, "1"),
        _match(0.65, 0.21, 0.14, "1"),
        _match(0.63, 0.22, 0.15, "1"),
        _match(0.69, 0.18, 0.13, "1"),
        _match(0.71, 0.19, 0.10, "1"),
        _match(0.72, 0.18, 0.10, "1"),
    ]


def test_optimize_16_rows_without_duplicates() -> None:
    matches = _build_matches()
    coupons = TotoOptimizer().optimize(matches=matches, mode="16")

    assert len(coupons) == 16
    assert len({tuple(row) for row in coupons}) == 16
    assert all(len(row) == 15 for row in coupons)


def test_risk_matches_are_varied_for_mode_16() -> None:
    matches = _build_matches()
    coupons = TotoOptimizer().optimize(matches=matches, mode="16")

    varied_indexes = [1, 3, 5, 7]
    fixed_indexes = [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14]

    for idx in varied_indexes:
        assert len({coupon[idx] for coupon in coupons}) >= 2

    for idx in fixed_indexes:
        assert len({coupon[idx] for coupon in coupons}) == 1


def test_optimize_32_rows_and_coverage_score() -> None:
    matches = _build_matches()
    coupons = TotoOptimizer().optimize(matches=matches, mode="32")

    assert len(coupons) == 32
    assert len({tuple(row) for row in coupons}) == 32

    score = TotoOptimizer().coverage_score(coupons=coupons, matches=matches)
    assert 0.0 <= score <= 1.0
    assert score > 0.60


def _build_insurance_conflict_matches() -> list[dict]:
    # Deterministic set with mixed uncertainty and conflicting lean for insurance stress.
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
    matches: list[dict] = []
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


def _global_history_context() -> dict:
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


def test_insurance_strength_affects_target_selection() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.7)
    summary_07 = dict(opt.last_run_summary)
    t07 = list(summary_07.get("target_match_indexes", []))

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary_09 = dict(opt.last_run_summary)
    t09 = list(summary_09.get("target_match_indexes", []))

    assert len(t09) >= len(t07)
    assert (len(t09) > len(t07)) or (set(t09) != set(t07))


def test_insurance_strength_affects_changed_cells() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.7)
    changed_07 = int(opt.last_run_summary.get("insurance_cells_changed_count", 0) or 0)

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    changed_09 = int(opt.last_run_summary.get("insurance_cells_changed_count", 0) or 0)

    assert changed_09 >= changed_07
    assert changed_09 > 0


def test_history_affects_insured_choices() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    coupons_off = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary_off = dict(opt.last_run_summary)

    coupons_on = opt.optimize_insurance(
        matches=matches,
        mode="32",
        insurance_strength=0.9,
        global_history_context=_global_history_context(),
    )
    summary_on = dict(opt.last_run_summary)

    assert int(summary_on.get("history_events_loaded_count", 0) or 0) >= int(summary_off.get("history_events_loaded_count", 0) or 0)

    # History must impact insured layer at least by target selection or produced coupons.
    assert (
        list(summary_on.get("target_match_indexes", [])) != list(summary_off.get("target_match_indexes", []))
        or coupons_on != coupons_off
    )


def test_insurance_diagnostics_present() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()
    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)

    summary = dict(opt.last_run_summary)
    diagnostics = summary.get("insurance_diagnostics", {})
    assert isinstance(diagnostics, dict)

    required = {
        "insurance_enabled",
        "insurance_strength",
        "insured_coupons_count",
        "affected_coupon_lines_count",
        "changed_matches_by_insurance",
        "target_match_indexes",
        "target_match_labels",
        "target_reason_summary",
        "change_distribution",
        "coverage_summary",
    }
    assert required.issubset(set(diagnostics.keys()))


def test_insurance_disabled_has_zero_or_empty_diagnostics() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()
    _ = opt.optimize(matches=matches, mode="16")

    summary = dict(opt.last_run_summary)
    diagnostics = summary.get("insurance_diagnostics", {})
    assert isinstance(diagnostics, dict)
    assert diagnostics.get("insurance_enabled") is False
    assert int(diagnostics.get("insured_coupons_count", 0) or 0) == 0
    assert int(diagnostics.get("affected_coupon_lines_count", 0) or 0) == 0
    assert list(diagnostics.get("changed_matches_by_insurance", [])) == []


def test_mode_16_vs_32_insurance_distribution() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    coupons_16 = opt.optimize_insurance(matches=matches, mode="16", insurance_strength=0.9)
    summary_16 = dict(opt.last_run_summary)
    insured_16 = int(summary_16.get("insured_coupons_count", 0) or 0)

    coupons_32 = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary_32 = dict(opt.last_run_summary)
    insured_32 = int(summary_32.get("insured_coupons_count", 0) or 0)

    assert len(coupons_16) == 16
    assert len(coupons_32) == 32
    assert insured_32 >= insured_16
    assert len({tuple(c) for c in coupons_32}) >= len({tuple(c) for c in coupons_16})


def test_coupon_lines_format_reflects_insurance_changes() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    coupons = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary = dict(opt.last_run_summary)

    assert len(coupons) == 32
    assert len({tuple(row) for row in coupons}) == 32

    lines = [";".join(["30"] + coupon) for coupon in coupons]
    assert len(lines) == 32
    for line in lines:
        parts = line.split(";")
        assert parts[0] == "30"
        assert len(parts) == 16

    insured_idxs = [int(i) for i in summary.get("insured_coupon_indices", []) if isinstance(i, int)]
    base_count = int(summary.get("base_coupons_count", 0) or 0)
    if insured_idxs:
        assert min(insured_idxs) >= base_count


def test_summary_contract_is_stable_for_base_and_insurance_modes() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    coupons_base = opt.optimize(matches=matches, mode="16")
    summary_base = dict(opt.last_run_summary)
    assert len(coupons_base) == 16

    required_base_keys = {
        "strategy_adjusted_matches_count",
        "strategy_adjusted_changed_matches",
        "pool_used_as_context_count",
        "pool_used_as_direct_signal_count",
        "history_used_as_context_count",
        "history_used_as_event_signal_count",
    }
    assert required_base_keys.issubset(set(summary_base.keys()))

    assert isinstance(summary_base.get("strategy_adjusted_changed_matches"), list)
    assert int(summary_base.get("strategy_adjusted_matches_count", 0) or 0) == len(
        list(summary_base.get("strategy_adjusted_changed_matches", []))
    )

    coupons_ins = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary_ins = dict(opt.last_run_summary)
    assert len(coupons_ins) == 32
    assert required_base_keys.issubset(set(summary_ins.keys()))


def test_layer_contract_and_coupon_typing_are_explicit() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    coupons = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary = dict(opt.last_run_summary)

    assert len(coupons) == 32
    assert summary.get("layer_contract_version") == "toto_layers_v1"

    base_coupon_count = int(summary.get("base_coupon_count", 0) or 0)
    insurance_coupon_count = int(summary.get("insurance_coupon_count", 0) or 0)
    assert base_coupon_count > 0
    assert insurance_coupon_count >= 0
    assert base_coupon_count + insurance_coupon_count == len(coupons)

    assert int(summary.get("strategy_adjusted_matches_count", 0) or 0) >= 0
    assert int(summary.get("insurance_target_matches_count", 0) or 0) >= 0
    assert int(summary.get("insurance_changed_matches_count", 0) or 0) >= 0
    assert int(summary.get("insurance_changed_coupon_lines_count", 0) or 0) >= 0

    coupon_entries = list(summary.get("coupon_entries", []))
    assert len(coupon_entries) == len(coupons)
    assert all(isinstance(item, dict) for item in coupon_entries)

    base_entries = [item for item in coupon_entries if item.get("coupon_type") == "base"]
    insurance_entries = [item for item in coupon_entries if item.get("coupon_type") == "insurance"]
    assert len(base_entries) == base_coupon_count
    assert len(insurance_entries) == insurance_coupon_count

    for item in base_entries:
        assert item.get("source_stage") == "base_strategy_layer"
        assert item.get("insurance_applied_flag") is False
    for item in insurance_entries:
        assert item.get("source_stage") == "insurance_layer"
        assert item.get("insurance_applied_flag") is True

    match_layers = list(summary.get("match_layer_diagnostics", []))
    assert len(match_layers) == len(matches)
    required_match_layer_keys = {
        "base_model_probabilities",
        "base_model_decision",
        "toto_strategy_adjusted_decision",
        "strategy_changed_flag",
        "strategy_reason_codes",
        "insurance_added_outcomes",
        "insurance_changed_flag",
        "insurance_reason_codes",
        "final_effective_coverage",
    }
    for row in match_layers:
        assert required_match_layer_keys.issubset(set(row.keys()))


def test_base_mode_has_no_insurance_additions_in_match_layer() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    coupons = opt.optimize(matches=matches, mode="16")
    summary = dict(opt.last_run_summary)

    assert len(coupons) == 16
    assert int(summary.get("insurance_coupon_count", 0) or 0) == 0
    assert int(summary.get("insurance_target_matches_count", 0) or 0) == 0
    assert int(summary.get("insurance_changed_matches_count", 0) or 0) == 0

    match_layers = list(summary.get("match_layer_diagnostics", []))
    assert len(match_layers) == len(matches)

    for row in match_layers:
        assert isinstance(row, dict)
        assert list(row.get("insurance_added_outcomes", [])) == []
        assert bool(row.get("insurance_changed_flag", False)) is False
        assert list(row.get("insurance_reason_codes", [])) == ["preserved_base_decision", "no_extra_cover"]


def test_explainability_reason_codes_strength_and_diversity_metrics_present() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary = dict(opt.last_run_summary)

    strategy_allowed = {
        "draw_risk",
        "favorite_fail_risk",
        "upset_risk",
        "pool_conflict",
        "history_conflict",
        "public_bias",
        "trap_signal",
        "model_uncertainty",
        "insufficient_market_context",
        "bookmaker_pool_divergence",
        "no_strong_toto_signal",
    }
    insurance_allowed = {
        "adjacent_safety",
        "opposite_safety",
        "draw_cover",
        "favorite_fail_cover",
        "upset_cover",
        "diversification_cover",
        "no_extra_cover",
        "low_justification_for_opposite",
        "preserved_base_decision",
    }

    match_layers = list(summary.get("match_layer_diagnostics", []))
    assert len(match_layers) == len(matches)

    seen_strategy = False
    seen_insurance = False
    for row in match_layers:
        assert isinstance(row.get("strategy_reason_codes"), list)
        assert isinstance(row.get("insurance_reason_codes"), list)
        assert str(row.get("strategy_justification_strength", "")) in {"weak", "medium", "strong"}
        assert str(row.get("insurance_justification_strength", "")) in {"weak", "medium", "strong"}
        assert isinstance(row.get("strategy_human_explanation"), str)
        assert isinstance(row.get("insurance_human_explanation"), str)

        for code in row.get("strategy_reason_codes", []):
            assert code in strategy_allowed
            seen_strategy = True
        for code in row.get("insurance_reason_codes", []):
            assert code in insurance_allowed
            seen_insurance = True

    assert seen_strategy is True
    assert seen_insurance is True

    for key in (
        "unique_coupon_count",
        "average_hamming_distance_between_coupons",
        "median_hamming_distance",
        "min_hamming_distance",
        "coupon_diversity_score",
    ):
        assert key in summary
    assert float(summary.get("average_hamming_distance_between_coupons", 0.0) or 0.0) >= 0.0
    assert float(summary.get("median_hamming_distance", 0.0) or 0.0) >= 0.0
    assert float(summary.get("min_hamming_distance", 0.0) or 0.0) >= 0.0
    assert float(summary.get("coupon_diversity_score", 0.0) or 0.0) >= 0.0


def test_opposite_safety_requires_strong_justification() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary = dict(opt.last_run_summary)
    layers = list(summary.get("match_layer_diagnostics", []))

    for row in layers:
        if not isinstance(row, dict):
            continue
        ins_codes = row.get("insurance_reason_codes", [])
        if isinstance(ins_codes, list) and "opposite_safety" in ins_codes:
            assert str(row.get("insurance_justification_strength", "")) == "strong"


def test_insurance_profile_07_vs_09_is_quality_different() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.7)
    s07 = dict(opt.last_run_summary)

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    s09 = dict(opt.last_run_summary)

    adj07 = int(s07.get("adjacent_alternatives_added_count", 0) or 0)
    opp07 = int(s07.get("strong_opposite_alternatives_added_count", 0) or 0)
    adj09 = int(s09.get("adjacent_alternatives_added_count", 0) or 0)
    opp09 = int(s09.get("strong_opposite_alternatives_added_count", 0) or 0)

    # 0.7 should stay more conservative than 0.9 for strong opposite outcomes.
    assert opp09 >= opp07
    # 0.9 may broaden protection and should not be less expressive overall.
    assert adj09 + opp09 >= adj07 + opp07


def test_weak_signal_prefers_no_extra_cover() -> None:
    weak_matches = [
        {
            "match_id": f"w{i}",
            "home": f"H{i}",
            "away": f"A{i}",
            "probs": {"P1": 0.80, "PX": 0.12, "P2": 0.08},
        }
        for i in range(1, 16)
    ]

    opt = TotoOptimizer()
    _ = opt.optimize_insurance(matches=weak_matches, mode="16", insurance_strength=0.7)
    summary = dict(opt.last_run_summary)
    layers = list(summary.get("match_layer_diagnostics", []))

    assert len(layers) == 15
    preserved_count = 0
    for row in layers:
        if not isinstance(row, dict):
            continue
        ins_codes = row.get("insurance_reason_codes", [])
        if isinstance(ins_codes, list) and "no_extra_cover" in ins_codes:
            preserved_count += 1

    assert preserved_count >= 1


def test_insurance_effect_summary_counts_are_consistent() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary = dict(opt.last_run_summary)
    layers = [row for row in summary.get("match_layer_diagnostics", []) if isinstance(row, dict)]

    derived_adj = sum(1 for row in layers if "adjacent_safety" in list(row.get("insurance_reason_codes", [])))
    derived_opp = sum(1 for row in layers if "opposite_safety" in list(row.get("insurance_reason_codes", [])))
    derived_opp_strong = sum(
        1
        for row in layers
        if "opposite_safety" in list(row.get("insurance_reason_codes", []))
        and str(row.get("insurance_justification_strength", "weak")) == "strong"
    )

    assert int(summary.get("adjacent_alternatives_added_count", 0) or 0) == derived_adj
    assert int(summary.get("strong_opposite_alternatives_added_count", 0) or 0) == derived_opp
    assert int(summary.get("strong_opposite_with_strong_justification_count", 0) or 0) == derived_opp_strong


def test_mode_32_diversity_has_distance_floor() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.9)
    summary = dict(opt.last_run_summary)

    assert int(summary.get("unique_coupon_count", 0) or 0) == 32
    assert float(summary.get("min_hamming_distance", 0.0) or 0.0) >= 2.0
    assert float(summary.get("median_hamming_distance", 0.0) or 0.0) >= 4.0


def test_global_history_context_increases_strategy_adjustments() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize(matches=matches, mode="16")
    summary_without = dict(opt.last_run_summary)

    _ = opt.optimize(matches=matches, mode="16", global_history_context=_global_history_context())
    summary_with = dict(opt.last_run_summary)

    adjusted_without = int(summary_without.get("strategy_adjusted_matches_count", 0) or 0)
    adjusted_with = int(summary_with.get("strategy_adjusted_matches_count", 0) or 0)

    assert adjusted_with >= adjusted_without
    assert adjusted_with > 0


def test_toto_aware_match_typing_and_scores_present() -> None:
    matches = _build_insurance_conflict_matches()
    opt = TotoOptimizer()

    _ = opt.optimize_insurance(matches=matches, mode="32", insurance_strength=0.7)
    summary = dict(opt.last_run_summary)

    assert "match_type_distribution" in summary
    assert "risk_level_distribution" in summary
    assert "resolution_type_distribution" in summary
    assert "avg_draw_risk_score" in summary
    assert "avg_insurance_priority_score" in summary

    layers = [row for row in summary.get("match_layer_diagnostics", []) if isinstance(row, dict)]
    assert len(layers) == len(matches)

    allowed_match_types = {
        "Strong favorite",
        "Weak favorite",
        "Draw-heavy balance",
        "Trap-like match",
        "Form-vs-market conflict",
        "High-uncertainty match",
    }
    allowed_risk_levels = {"low-risk", "medium-risk", "high-risk", "trap-like"}
    allowed_resolution = {"single", "double", "insurance priority"}

    for row in layers:
        assert str(row.get("match_type", "")) in allowed_match_types
        assert str(row.get("risk_level", "")) in allowed_risk_levels
        assert str(row.get("resolution_type", "")) in allowed_resolution

        for score_key in (
            "draw_risk_score",
            "upset_risk_score",
            "weak_favorite_score",
            "trap_match_score",
            "single_confidence_score",
            "insurance_priority_score",
        ):
            value = float(row.get(score_key, 0.0) or 0.0)
            assert 0.0 <= value <= 1.0


def test_toto_aware_double_recommendation_avoids_fragile_single() -> None:
    weak_favorite_matches = []
    for idx in range(1, 16):
        weak_favorite_matches.append(
            {
                "match_id": f"wf{idx}",
                "home": f"HF{idx}",
                "away": f"AF{idx}",
                "feature_context_level": "partial_context",
                "features": {
                    "entropy": 0.86,
                    "gap": 0.06,
                    "volatility": 0.17,
                    "ppg_diff": 0.10,
                    "split_advantage": 0.08,
                },
                "probs": {"P1": 0.46, "PX": 0.31, "P2": 0.23},
                "pool_probs": {"P1": 0.52, "PX": 0.28, "P2": 0.20},
            }
        )

    opt = TotoOptimizer()
    _ = opt.optimize(matches=weak_favorite_matches, mode="16")
    summary = dict(opt.last_run_summary)
    layers = [row for row in summary.get("match_layer_diagnostics", []) if isinstance(row, dict)]

    assert len(layers) == 15
    # For this crafted weak-favorite profile, at least one row should trigger the TOTO guard.
    assert any(bool(row.get("toto_layer_used", False)) for row in layers)
    # And the strategy-adjusted decision should not stay fragile single for all rows.
    assert any(len(str(row.get("toto_strategy_adjusted_decision", ""))) >= 2 for row in layers)
