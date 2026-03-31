from __future__ import annotations

import itertools
import logging
import math
from typing import Any

logger = logging.getLogger("toto")

ALLOWED_OUTCOMES = {"1", "X", "2"}
_PROB_KEY = {"1": "P1", "X": "PX", "2": "P2"}


class TotoOptimizer:
    def __init__(self) -> None:
        self.last_run_summary: dict[str, Any] = {}
        # Global Baltbet history context — set once via set_global_history_context().
        # Used as a fallback for ALL matches when no per-match history is attached
        # (which is always the case in manual MATCHES -> TOTO mode).
        self.global_history_context: dict | None = None
        self._run_global_history: dict | None = None

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def set_global_history_context(self, context: dict | None) -> None:
        """Persist a Baltbet draw history context for all subsequent optimize calls.

        Pass the payload returned by ``TotoAPI.get_draw_history()`` to enable
        history-aware decision making in manual MATCHES -> TOTO mode.
        Pass ``None`` to clear.
        """
        self.global_history_context = context if isinstance(context, dict) else None

    def optimize(self, matches: list[dict], mode: str, global_history_context: dict | None = None) -> list[list[str]]:
        if mode not in {"16", "32"}:
            raise ValueError("mode must be '16' or '32'")
        if not matches:
            return []

        # Use call-level context first, then persistent context.
        self._run_global_history = (
            global_history_context if isinstance(global_history_context, dict)
            else self.global_history_context
        )
        ordered_matches = list(matches)
        target_lines = 16 if mode == "16" else 32
        risk_count = 4 if mode == "16" else 5

        sorted_indexes = self._sort_indexes_by_risk(matches=ordered_matches)
        risk_indexes = sorted_indexes[: min(risk_count, len(matches))]
        risk_index_set = set(risk_indexes)

        base_contexts = [self._evaluate_base_decision(match) for match in ordered_matches]
        base_decisions = [context["decision"] for context in base_contexts]

        # Allow a small number of non-risk matches to carry history-driven doubles.
        history_expand_candidates: list[tuple[int, float]] = []
        for idx, ctx in enumerate(base_contexts):
            if idx in risk_index_set:
                continue
            if not bool(ctx.get("history_used", False)):
                continue
            decision = str(ctx.get("decision", ""))
            if len(decision) < 2:
                continue
            history_expand_candidates.append((idx, float(ctx.get("history_adjustment", 0.0))))
        history_expand_candidates.sort(key=lambda item: item[1], reverse=True)
        history_expand_indexes = {idx for idx, _ in history_expand_candidates[:2]}

        pools_by_index: dict[int, list[str]] = {}
        for idx, match in enumerate(ordered_matches):
            if idx in risk_index_set or idx in history_expand_indexes:
                pools_by_index[idx] = self._base_pool(match=match, decision=base_decisions[idx])
            else:
                pools_by_index[idx] = [self._single_from_decision(decision=base_decisions[idx], probs=match["probs"])]

        pools = [pools_by_index[idx] for idx in range(len(ordered_matches))]
        coupons = self._assemble_coupons(
            pools=pools,
            matches=ordered_matches,
            target_lines=target_lines,
            diversity_strength=0.08,
        )
        coupons = self._ensure_unique_coupon_set(
            coupons=coupons,
            target_lines=target_lines,
            pools=pools,
            matches=ordered_matches,
        )

        self.last_run_summary = self._build_run_summary(
            matches=ordered_matches,
            mode=mode,
            insurance_strength=0.0,
            base_decisions=base_decisions,
            base_contexts=base_contexts,
            insured_pools=pools,
            coupons=coupons,
            generation_mode="base",
        )

        logger.info(
            "toto_optimize mode=%s risk_count=%d coupons=%d",
            mode,
            len(risk_indexes),
            len(coupons),
        )
        return coupons

    def optimize_insurance(
        self,
        matches: list[dict],
        mode: str,
        insurance_strength: float = 0.7,
        global_history_context: dict | None = None,
    ) -> list[list[str]]:
        if mode not in {"16", "32"}:
            raise ValueError("mode must be '16' or '32'")
        if not matches:
            return []
        if not (0.0 <= insurance_strength <= 1.0):
            raise ValueError("insurance_strength must be between 0.0 and 1.0")

        # Use call-level context first, then persistent context.
        self._run_global_history = (
            global_history_context if isinstance(global_history_context, dict)
            else self.global_history_context
        )
        ordered_matches = list(matches)
        target_lines = 16 if mode == "16" else 32
        base_risk_count = 4 if mode == "16" else 5
        if insurance_strength >= 0.90:
            extra_risk_count = 4
        elif insurance_strength >= 0.70:
            extra_risk_count = 2
        elif insurance_strength >= 0.40:
            extra_risk_count = 1
        else:
            extra_risk_count = 0
        risk_count = min(len(matches), base_risk_count + extra_risk_count)

        sorted_indexes = self._sort_indexes_by_risk(matches=ordered_matches)
        risk_indexes = sorted_indexes[: min(risk_count, len(matches))]

        base_contexts = [self._evaluate_base_decision(match) for match in ordered_matches]
        base_decisions = [context["decision"] for context in base_contexts]

        base_pools_by_index: dict[int, list[str]] = {}
        insured_pools_by_index: dict[int, list[str]] = {}
        insured_matches_dict: dict[int, dict[str, Any]] = {}
        strategy_risk_scores: dict[int, float] = {}

        match_profiles = [self._decision_profile(match) for match in ordered_matches]
        match_strategies = [
            self._extract_strategy_signals(match=match, probs=match_profiles[idx]["probs"])
            for idx, match in enumerate(ordered_matches)
        ]

        enriched_risk: list[tuple[int, float]] = []
        for idx in range(len(ordered_matches)):
            target_score = self._insurance_target_score(
                profile=match_profiles[idx],
                strategy=match_strategies[idx],
            )
            toto_priority = self._safe_float(base_contexts[idx].get("insurance_priority_score", 0.0))
            if toto_priority > 0.0:
                target_score = min(1.0, 0.75 * target_score + 0.25 * toto_priority)
            strategy_risk_scores[idx] = target_score
            enriched_risk.append((idx, target_score))

        enriched_risk.sort(key=lambda item: item[1], reverse=True)

        if insurance_strength >= 0.90:
            insured_candidate_extra = 7 if mode == "32" else 5
        elif insurance_strength >= 0.70:
            insured_candidate_extra = 3 if mode == "32" else 2
        elif insurance_strength >= 0.55:
            insured_candidate_extra = 1
        else:
            insured_candidate_extra = 0

        insured_candidate_count = min(len(matches), base_risk_count + insured_candidate_extra)
        insured_candidate_indexes = {idx for idx, _ in enriched_risk[:insured_candidate_count]}
        candidate_rank = {idx: rank for rank, (idx, _) in enumerate(enriched_risk[:insured_candidate_count])}
        risk_index_set = set(risk_indexes)

        for idx, match in enumerate(ordered_matches):
            profile = match_profiles[idx]
            strategy = match_strategies[idx]
            probs = profile["probs"]
            base_single = self._single_from_decision(decision=base_decisions[idx], probs=probs)
            base_outcomes = self._options_from_decision(decision=base_decisions[idx], probs=probs)

            if idx in risk_index_set:
                base_pool = self._base_pool(match=match, decision=base_decisions[idx])
            else:
                base_pool = [base_single]
            base_pools_by_index[idx] = base_pool

            if idx not in insured_candidate_indexes:
                insured_pools_by_index[idx] = base_pool
                continue

            effective_strength = insurance_strength
            rank_pos = candidate_rank.get(idx, 0)
            if insurance_strength >= 0.90:
                hot_cut = max(2, insured_candidate_count // 3)
                effective_strength = min(1.0, insurance_strength + (0.10 if rank_pos < hot_cut else 0.05))
            elif insurance_strength >= 0.70:
                # Keep 0.7 moderate: widen top half more than lower half.
                effective_strength = max(0.62, insurance_strength - (0.04 if rank_pos > max(1, insured_candidate_count // 2) else 0.0))

            insured_pool = self._insurance_pool(
                match=match,
                decision=base_decisions[idx],
                insurance_strength=effective_strength,
                strategy=strategy,
            )
            if idx in risk_index_set and len(insured_pool) < len(base_pool):
                insured_pool = base_pool

            insured_pools_by_index[idx] = insured_pool

            alternatives = [outcome for outcome in insured_pool if outcome not in base_outcomes]
            if alternatives:
                risk_score = strategy_risk_scores.get(idx, 0.0)
                reason_scores = self._insurance_reason_scores(
                    profile=profile,
                    strategy=strategy,
                    base_single=base_single,
                )
                insured_matches_dict[idx] = {
                    "base_outcome": base_single,
                    "base_outcomes": list(base_outcomes),
                    "alternatives": alternatives,
                    "pool_size": len(insured_pool),
                    "risk_score": risk_score,
                    "reason_scores": reason_scores,
                    "justification_value": max(reason_scores.values()) if reason_scores else 0.0,
                }

        base_pools = [base_pools_by_index[idx] for idx in range(len(ordered_matches))]
        insured_pools = [insured_pools_by_index[idx] for idx in range(len(ordered_matches))]

        # Split final output: base coupons first, insured coupons at end.
        insured_ratio = 0.10 + (0.55 * insurance_strength)
        insured_target = 0
        if insured_matches_dict:
            insured_target = max(1, int(round(target_lines * insured_ratio)))
            insured_target = min(insured_target, max(0, target_lines - 1))
        base_target = target_lines - insured_target

        base_coupons = self._assemble_coupons(
            pools=base_pools,
            matches=ordered_matches,
            target_lines=max(base_target, 1),
            diversity_strength=0.08,
        )

        insured_coupons, changed_matches_by_line = self._build_insured_coupons_from_base(
            base_coupons=base_coupons,
            insured_pools=insured_pools,
            insured_matches=insured_matches_dict,
            strategy_risk_scores=strategy_risk_scores,
            insurance_strength=insurance_strength,
            target_count=insured_target,
        )

        # Base coupons first, then insured coupons at end.
        coupons = list(base_coupons) + insured_coupons
        if len(coupons) > target_lines:
            coupons = coupons[:target_lines]

        if len(coupons) < target_lines:
            filler = self._assemble_coupons(
                pools=insured_pools,
                matches=ordered_matches,
                target_lines=target_lines,
                diversity_strength=0.22 + 0.7 * insurance_strength,
            )
            existing = {tuple(coupon) for coupon in coupons}
            filler_min_distance = 2 if insurance_strength >= 0.90 else 1
            for candidate in filler:
                key = tuple(candidate)
                if key in existing:
                    continue
                if coupons:
                    nearest_existing = min(self._hamming_distance(candidate, current) for current in coupons)
                    if nearest_existing < filler_min_distance:
                        continue
                coupons.append(candidate)
                existing.add(key)
                if len(coupons) >= target_lines:
                    break

        if len(coupons) < target_lines and coupons:
            coupons = self._ensure_unique_coupon_set(
                coupons=coupons,
                target_lines=target_lines,
                pools=insured_pools,
                matches=ordered_matches,
            )

        coupons = self._ensure_unique_coupon_set(
            coupons=coupons,
            target_lines=target_lines,
            pools=insured_pools,
            matches=ordered_matches,
        )

        self.last_run_summary = self._build_run_summary_with_insurance(
            matches=ordered_matches,
            mode=mode,
            insurance_strength=insurance_strength,
            base_decisions=base_decisions,
            base_contexts=base_contexts,
            insured_pools=insured_pools,
            insured_matches_dict=insured_matches_dict,
            strategy_risk_scores=strategy_risk_scores,
            coupons=coupons,
            base_coupons_count=len(base_coupons),
            insured_coupons_count=len(insured_coupons),
            changed_matches_by_line=changed_matches_by_line,
            generation_mode="insurance",
        )

        logger.info(
            "toto_optimize_insurance mode=%s strength=%.2f risk_matches=%d coupons=%d base=%d insured=%d",
            mode,
            insurance_strength,
            len(risk_indexes),
            len(coupons),
            len(base_coupons),
            len(insured_coupons),
        )

        insurance_diag = self.last_run_summary.get("insurance_diagnostics", {}) if isinstance(self.last_run_summary, dict) else {}
        if isinstance(insurance_diag, dict):
            logger.info(
                "[INSURANCE] enabled=%s strength=%.2f target_matches=%s insured_coupons=%s changed_cells=%s history_influenced_matches=%s outcome_expansion=%s",
                bool(insurance_diag.get("insurance_enabled", True)),
                insurance_strength,
                insurance_diag.get("target_match_indexes", []),
                insurance_diag.get("insured_coupons_count", 0),
                insurance_diag.get("insurance_cells_changed_count", 0),
                self.last_run_summary.get("history_influenced_matches_count", 0),
                insurance_diag.get("change_distribution", {}),
            )
            logger.debug(
                "[INSURANCE] labels=%s reason_summary=%s coverage_summary=%s",
                insurance_diag.get("target_match_labels", []),
                insurance_diag.get("target_reason_summary", {}),
                insurance_diag.get("coverage_summary", {}),
            )

        return coupons

    def coverage_score(self, coupons: list[list[str]], matches: list[dict]) -> float:
        if not coupons or not matches:
            return 0.0
        total = 0.0
        for idx, match in enumerate(matches):
            outcomes = {coupon[idx] for coupon in coupons}
            covered = sum(match["probs"][_PROB_KEY[outcome]] for outcome in outcomes)
            total += covered
        return total / len(matches)

    # ---------------------------------------------------------------------
    # Decision engine
    # ---------------------------------------------------------------------

    def _base_decision(self, match: dict) -> str:
        return self._evaluate_base_decision(match)["decision"]

    def _evaluate_base_decision(self, match: dict) -> dict[str, Any]:
        profile = self._decision_profile(match)
        decision_class = self._classify_decision_profile(profile=profile)
        probs = profile["probs"]
        ranked = profile["ranked"]
        top_outcome = ranked[0][0]
        second_outcome = ranked[1][0]

        if decision_class.startswith("strong_single_"):
            model_decision = decision_class.removeprefix("strong_single_")
        elif decision_class.startswith("balanced_"):
            model_decision = decision_class.removeprefix("balanced_")
        else:
            model_decision = self._decision_from_outcomes([top_outcome, second_outcome])

        strategy = self._extract_strategy_signals(match=match, probs=probs)
        history_adjusted = self._apply_history_strategy(
            decision=model_decision,
            decision_class=decision_class,
            profile=profile,
            strategy=strategy,
        )
        adjusted_decision = str(history_adjusted.get("decision", model_decision))
        history_used = bool(history_adjusted.get("history_used", False))

        # TOTO-aware interpretation layer:
        # classify match toxicity and enforce a safety double when a weak single
        # would be fragile for draw-sheet style fixtures.
        toto_interp = self._toto_match_interpretation(
            match=match,
            profile=profile,
            strategy=strategy,
            decision_class=decision_class,
        )
        toto_layer_used = False
        if bool(toto_interp.get("double_recommendation_flag", False)) and len(adjusted_decision) == 1:
            primary = self._single_from_decision(adjusted_decision, probs)
            secondary = self._pick_toto_secondary_outcome(
                primary=primary,
                profile=profile,
                toto_interp=toto_interp,
            )
            if secondary and secondary != primary:
                adjusted_decision = self._select_double_type(
                    primary=primary,
                    secondary=secondary,
                    probs=probs,
                    draw_underpricing=float(strategy.get("draw_underpricing", 0.0)),
                )
                toto_layer_used = True

        return {
            "model_decision": model_decision,
            "decision": adjusted_decision,
            "decision_class": decision_class,
            "history_available": bool(strategy["available"]),
            "history_used": history_used,
            "pool_signals_used": bool(
                strategy.get("pool_public_used", False)
                or any(
                    sig in strategy.get("strategy_signals_used", [])
                    for sig in (
                        "pool_probs_dict",
                        "pool_quotes_dict",
                        "pool_probs_divergence",
                        "public_vs_model_divergence",
                    )
                )
            ),
            "bk_signals_used": bool(strategy.get("bookmaker_used", False)),
            "strategy_reason": strategy["reason"],
            "signal_source": strategy["signal_source"],
            "strategy_signals_used": list(strategy.get("strategy_signals_used", [])),
            "history_context_used": bool(strategy.get("history_context_used", False)),
            "history_mode": str(strategy.get("history_mode", "no_history")),
            "event_history_used": bool(strategy.get("event_history_used", False)),
            "global_history_context_loaded": bool(strategy.get("global_history_context_loaded", False)),
            "global_history_injected": bool(strategy.get("global_history_injected", False)),
            "pool_public_used": bool(strategy.get("pool_public_used", False)),
            "bookmaker_used": bool(strategy.get("bookmaker_used", False)),
            "history_reason": str(strategy.get("history_reason", strategy.get("reason", ""))),
            "history_score": float(strategy.get("history_score", 0.0)),
            "favorite_trap_score": float(strategy.get("favorite_trap_score", 0.0)),
            "upset_potential": float(strategy.get("upset_potential", 0.0)),
            "draw_tendency_score": float(strategy.get("draw_tendency_score", 0.0)),
            "public_bias_score": float(strategy.get("public_bias", 0.0)),
            "pool_vs_bookmaker_divergence": float(strategy.get("pool_vs_bookmaker_divergence", 0.0)),
            "history_conflict_with_model": float(strategy.get("history_conflict_with_model", 0.0)),
            "historical_favorite_fail_score": float(strategy.get("historical_favorite_fail_score", 0.0)),
            "historical_draw_bias_score": float(strategy.get("historical_draw_bias_score", 0.0)),
            "historical_upset_score": float(strategy.get("historical_upset_score", 0.0)),
            "pool_vs_bk_historical_divergence": float(strategy.get("pool_vs_bk_historical_divergence", 0.0)),
            "underdog_conversion_score": float(strategy.get("underdog_conversion_score", 0.0)),
            "history_result_alignment_with_model": float(strategy.get("history_result_alignment_with_model", 0.0)),
            "event_history_signal_strength": float(strategy.get("event_history_signal_strength", 0.0)),
            "payload_toto_brief_history": strategy.get("payload_toto_brief_history", {}),
            "payload_pool_probs": strategy.get("payload_pool_probs", {}),
            "payload_pool_quotes": strategy.get("payload_pool_quotes", {}),
            "payload_bookmaker_probs": strategy.get("payload_bookmaker_probs", {}),
            "payload_bookmaker_quotes": strategy.get("payload_bookmaker_quotes", {}),
            "history_adjustment": float(history_adjusted.get("history_adjustment", 0.0)),
            "market_adjustment": float(history_adjusted.get("market_adjustment", 0.0)),
            "history_adjustment_reason": str(history_adjusted.get("reason", "")),
            "history_ready": bool(strategy.get("history_ready", False)),
            "history_state": str(strategy.get("history_state", "history_not_requested")),
            "history_empty_reason": strategy.get("history_empty_reason"),
            "history_draws_loaded_count": int(strategy.get("history_draws_loaded_count", 0)),
            "history_events_loaded_count": int(strategy.get("history_events_loaded_count", 0)),
            "match_type": str(toto_interp.get("match_type", "High-uncertainty match")),
            "risk_level": str(toto_interp.get("risk_level", "medium-risk")),
            "resolution_type": str(toto_interp.get("resolution_type", "double")),
            "draw_risk_score": float(toto_interp.get("draw_risk_score", 0.0)),
            "upset_risk_score": float(toto_interp.get("upset_risk_score", 0.0)),
            "weak_favorite_score": float(toto_interp.get("weak_favorite_score", 0.0)),
            "trap_match_score": float(toto_interp.get("trap_match_score", 0.0)),
            "single_confidence_score": float(toto_interp.get("single_confidence_score", 0.0)),
            "double_recommendation_flag": bool(toto_interp.get("double_recommendation_flag", False)),
            "insurance_priority_score": float(toto_interp.get("insurance_priority_score", 0.0)),
            "market_model_disagreement_score": float(toto_interp.get("market_model_disagreement_score", 0.0)),
            "context_completeness_score": float(toto_interp.get("context_completeness_score", 0.0)),
            "toto_layer_used": toto_layer_used,
        }

    def _decision_profile(self, match: dict) -> dict[str, Any]:
        probs = self._normalised_probs(match["probs"])
        ranked = self._rank_outcomes(probs)
        top_outcome, top_prob = ranked[0]
        second_outcome, second_prob = ranked[1]
        third_outcome, third_prob = ranked[2]

        entropy = 0.0
        for value in (top_prob, second_prob, third_prob):
            if value > 0:
                entropy -= value * math.log(value, 2)

        return {
            "probs": probs,
            "ranked": ranked,
            "top_outcome": top_outcome,
            "top_prob": top_prob,
            "second_outcome": second_outcome,
            "second_prob": second_prob,
            "third_outcome": third_outcome,
            "third_prob": third_prob,
            "margin_top_second": top_prob - second_prob,
            "margin_top_third": top_prob - third_prob,
            "entropy": entropy,
            "uncertainty": min(1.0, entropy / math.log(3, 2)),
        }

    def _classify_decision_profile(self, profile: dict[str, Any]) -> str:
        top_prob = float(profile["top_prob"])
        third_prob = float(profile["third_prob"])
        top_outcome = str(profile["top_outcome"])
        second_outcome = str(profile["second_outcome"])
        third_outcome = str(profile["third_outcome"])
        margin_top_second = float(profile["margin_top_second"])
        margin_top_third = float(profile["margin_top_third"])
        entropy = float(profile["entropy"])

        if top_prob >= 0.54 and margin_top_second >= 0.14:
            return f"strong_single_{top_outcome}"
        if top_prob >= 0.47 and margin_top_second >= 0.11 and entropy <= 1.38:
            return f"strong_single_{top_outcome}"
        if margin_top_second <= 0.05 and margin_top_third <= 0.10:
            return "high_uncertainty"
        if margin_top_second <= 0.08:
            return f"balanced_{self._decision_from_outcomes([top_outcome, second_outcome])}"
        if third_prob >= 0.26 and margin_top_third <= 0.14:
            return f"balanced_{self._decision_from_outcomes([top_outcome, third_outcome])}"
        if top_prob <= 0.42 and third_prob >= 0.22 and margin_top_third <= 0.13:
            return "high_uncertainty"
        return f"strong_single_{top_outcome}"

    # ---------------------------------------------------------------------
    # Strategy / history signals
    # ---------------------------------------------------------------------

    def _extract_strategy_signals(self, match: dict, probs: dict[str, float]) -> dict[str, Any]:
        merged: dict[str, float] = {}
        strategy_signals_used: set[str] = set()

        history_stats = self._extract_history_stats(match=match)
        draws_loaded = int(history_stats["draws_count"])
        events_loaded = int(history_stats["events_count"])
        history_state = str(history_stats.get("history_state", "history_not_requested"))
        history_ready = bool(history_stats.get("history_ready", False))
        history_empty_reason = history_stats.get("history_empty_reason")
        per_match_history_payload = isinstance(match.get("toto_brief_history"), dict)
        global_history_context_loaded = bool(getattr(self, "_run_global_history", None) or self.global_history_context)
        global_history_injected = bool(global_history_context_loaded and not per_match_history_payload and (draws_loaded > 0 or events_loaded > 0))
        if draws_loaded > 0:
            strategy_signals_used.add("history_draws_loaded")
        if events_loaded > 0:
            strategy_signals_used.add("history_events_loaded")

        # ------------------------------------------------------------------
        # STEP 1: Read pool_probs dict (TotoBrief pool distribution).
        # The UI bridge sets match["pool_probs"] = {"P1": ..., "PX": ..., "P2": ...}
        # whenever it has TotoBrief pool data. This is the primary public signal
        # in manual MATCHES->TOTO mode.
        # ------------------------------------------------------------------
        pool_probs_dict = match.get("pool_probs")
        if isinstance(pool_probs_dict, dict):
            for dest_key, src_keys in (
                ("pool_p1", ("P1", "p1")),
                ("pool_px", ("PX", "px", "pX")),
                ("pool_p2", ("P2", "p2")),
            ):
                if dest_key in merged:
                    continue
                for src_key in src_keys:
                    raw = pool_probs_dict.get(src_key)
                    if raw is None:
                        continue
                    try:
                        val = float(raw)
                        if val > 0.0:
                            merged[dest_key] = val
                            strategy_signals_used.add("pool_probs_dict")
                            break
                    except (TypeError, ValueError):
                        continue

        # ------------------------------------------------------------------
        # STEP 2: Read model_probs dict separately (for divergence analysis).
        # ------------------------------------------------------------------
        model_probs_dict = match.get("model_probs")
        if isinstance(model_probs_dict, dict):
            for dest_key, src_keys in (
                ("model_p1", ("P1", "p1")),
                ("model_px", ("PX", "px", "pX")),
                ("model_p2", ("P2", "p2")),
            ):
                if dest_key in merged:
                    continue
                for src_key in src_keys:
                    raw = model_probs_dict.get(src_key)
                    if raw is None:
                        continue
                    try:
                        val = float(raw)
                        if val > 0.0:
                            merged[dest_key] = val
                            strategy_signals_used.add("model_probs_dict")
                            break
                    except (TypeError, ValueError):
                        continue

        # ------------------------------------------------------------------
        # STEP 3: Read strategy_signals / history_signals / toto_brief_history dicts.
        # ------------------------------------------------------------------
        for key in ("strategy_signals", "history_signals", "toto_brief_history"):
            payload = match.get(key)
            if isinstance(payload, dict):
                for k, v in payload.items():
                    try:
                        merged[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue
                strategy_signals_used.add(key)

        # ------------------------------------------------------------------
        # STEP 4: Flat top-level quote aliases.
        # ------------------------------------------------------------------
        top_level_mappings: list[tuple[str, tuple[str, ...]]] = [
            ("bk_p1", ("bk_win_1", "bk_p1", "bookmaker_p1", "implied_prob_1")),
            ("bk_px", ("bk_draw", "bk_px", "bookmaker_px", "implied_prob_x")),
            ("bk_p2", ("bk_win_2", "bk_p2", "bookmaker_p2", "implied_prob_2")),
            ("pool_p1", ("pool_win_1", "pool_p1", "public_p1")),
            ("pool_px", ("pool_draw", "pool_px", "public_px")),
            ("pool_p2", ("pool_win_2", "pool_p2", "public_p2")),
        ]
        for canonical, aliases in top_level_mappings:
            if canonical in merged:
                continue
            for alias in aliases:
                raw = match.get(alias)
                if raw is None:
                    continue
                try:
                    merged[canonical] = float(raw)
                    strategy_signals_used.add("top_level_quotes")
                    break
                except (TypeError, ValueError):
                    continue

        # ------------------------------------------------------------------
        # STEP 5: Nested bookmaker_quotes dict.
        # ------------------------------------------------------------------
        bk_quotes = match.get("bookmaker_quotes")
        if isinstance(bk_quotes, dict):
            for alias_key, canonical in (("win_1", "bk_p1"), ("draw", "bk_px"), ("win_2", "bk_p2")):
                if canonical in merged:
                    continue
                raw = bk_quotes.get(alias_key)
                if raw is None:
                    continue
                try:
                    merged[canonical] = float(raw)
                    strategy_signals_used.add("bookmaker_quotes")
                except (TypeError, ValueError):
                    pass

        # Nested pool_quotes dict (bk_win_1/pool_draw/pool_win_2 inside pool_quotes sub-dict).
        pool_quotes = match.get("pool_quotes")
        if isinstance(pool_quotes, dict):
            for alias_key, canonical in (
                ("pool_win_1", "pool_p1"),
                ("pool_draw", "pool_px"),
                ("pool_win_2", "pool_p2"),
                ("win_1", "pool_p1"),
                ("draw", "pool_px"),
                ("win_2", "pool_p2"),
            ):
                if canonical in merged:
                    continue
                raw = pool_quotes.get(alias_key)
                if raw is None:
                    continue
                try:
                    merged[canonical] = float(raw)
                    strategy_signals_used.add("pool_quotes_dict")
                except (TypeError, ValueError):
                    pass

        # ------------------------------------------------------------------
        # STEP 6: Features dict (only public/crowd/history/book/pool signals).
        # ------------------------------------------------------------------
        features = match.get("features")
        if isinstance(features, dict):
            for k, v in features.items():
                name = str(k)
                if not any(tag in name for tag in ("public", "crowd", "history", "book", "bk_", "pool_")):
                    continue
                try:
                    merged[name] = float(v)
                    strategy_signals_used.add("feature_quotes")
                except (TypeError, ValueError):
                    continue

        # ------------------------------------------------------------------
        # STEP 6.5: Event history aggregate pool/bk priors.
        # Always compute precise event signals when global events exist.
        # Also inject aggregate pool/bk priors into merged when no per-match
        # pool data is present, so pool-vs-model divergence analysis fires.
        # ------------------------------------------------------------------
        _global_event_sigs: dict | None = None
        if events_loaded > 0:
            _gctx = getattr(self, "_run_global_history", None) or self.global_history_context
            if isinstance(_gctx, dict):
                _gevents = _gctx.get("events")
                if isinstance(_gevents, list) and _gevents:
                    _global_event_sigs = self._compute_event_history_signals(_gevents)
                    # Only inject aggregate priors when no per-match pool data exists.
                    if "pool_p1" not in merged:
                        _pool_avg = _global_event_sigs.get("pool_avg_probs")
                        _bk_avg = _global_event_sigs.get("bk_avg_probs")
                        if _pool_avg is not None:
                            merged["pool_p1"] = float(_pool_avg["P1"])
                            merged["pool_px"] = float(_pool_avg["PX"])
                            merged["pool_p2"] = float(_pool_avg["P2"])
                            strategy_signals_used.add("event_history_pool_priors")
                        if _bk_avg is not None and "bk_p1" not in merged:
                            merged["bk_p1"] = float(_bk_avg["P1"])
                            merged["bk_px"] = float(_bk_avg["PX"])
                            merged["bk_p2"] = float(_bk_avg["P2"])
                            strategy_signals_used.add("event_history_bk_priors")

        # ------------------------------------------------------------------
        # STEP 7: Build public probs.
        # Priority: pool signals (direct public behavior), then bk signals.
        # pool_probs_dict is already extracted into pool_p1/px/p2 in merged.
        # ------------------------------------------------------------------
        def _first_nonzero(*keys: str) -> float:
            for k in keys:
                val = merged.get(k, 0.0)
                if val > 0.0:
                    return val
            return 0.0

        # Pool probs take priority as the "public" signal (they represent what bettors are doing).
        # Then fall back to bookmaker quotes only if no pool data.
        public_raw = {
            "P1": _first_nonzero("pool_p1", "public_p1", "crowd_p1", "history_p1", "bk_p1"),
            "PX": _first_nonzero("pool_px", "public_px", "crowd_px", "history_px", "bk_px"),
            "P2": _first_nonzero("pool_p2", "public_p2", "crowd_p2", "history_p2", "bk_p2"),
        }
        public = self._normalised_probs(public_raw) if sum(public_raw.values()) > 0.0 else None

        # Also extract separate bk probs for pool_vs_bk divergence analysis.
        bk_raw = {
            "P1": _first_nonzero("bk_p1"),
            "PX": _first_nonzero("bk_px"),
            "P2": _first_nonzero("bk_p2"),
        }
        bk_probs = self._normalised_probs(bk_raw) if sum(bk_raw.values()) > 0.0 else None

        contrarian = max(("1", "X", "2"), key=lambda outcome: probs[_PROB_KEY[outcome]])
        crowd_favorite = contrarian
        reason = "history_unavailable"
        pool_vs_model_bias: float = 0.0
        crowd_favorite_competitive: bool = False

        if public is not None:
            divergences = {
                o: probs[_PROB_KEY[o]] - public[_PROB_KEY[o]]
                for o in ("1", "X", "2")
            }
            # Contrarian = outcome where model is most bullish vs pool/public
            contrarian = max(("1", "X", "2"), key=lambda o: divergences[o])
            # Crowd favorite = what the pool/public is betting most on
            crowd_favorite = max(("1", "X", "2"), key=lambda o: public[_PROB_KEY[o]])
            crowds_share = public[_PROB_KEY[crowd_favorite]]
            model_crowd_share = probs[_PROB_KEY[crowd_favorite]]

            # How much does pool disagree with model?
            pool_vs_model_bias = max(
                abs(public["P1"] - probs["P1"]),
                abs(public["PX"] - probs["PX"]),
                abs(public["P2"] - probs["P2"]),
            )
            # Crowd's pick is competitive if:
            # (a) model probability is within 16% of model's top, OR
            # (b) pool itself places >= 32% weight on that outcome (pool conviction)
            crowd_favorite_competitive = (
                probs[_PROB_KEY[crowd_favorite]] >= max(probs.values()) - 0.16
                or public[_PROB_KEY[crowd_favorite]] >= 0.32
            )
            reason = "public_vs_model_divergence"
            strategy_signals_used.add("public_vs_model_divergence")
            strategy_signals_used.add("pool_probs_divergence")

        # Compute pool vs bookmaker divergence (when both are available).
        pool_vs_bk_divergence: float = 0.0
        if public is not None and bk_probs is not None:
            pool_vs_bk_divergence = sum(
                abs(public[_PROB_KEY[o]] - bk_probs[_PROB_KEY[o]])
                for o in ("1", "X", "2")
            ) / 3.0
            if pool_vs_bk_divergence > 0.03:
                strategy_signals_used.add("pool_vs_bookmaker_divergence")

        history_signals = self._derive_history_signals(
            probs=probs,
            history_stats=history_stats,
            public=public,
        )

        if draws_loaded > 0 and events_loaded <= 0:
            draw_context_strength = min(0.25, 0.04 + (float(draws_loaded) / 700.0))
            public_bias_fallback = 0.0
            if public is not None:
                public_bias_fallback = max(
                    abs(public["P1"] - probs["P1"]),
                    abs(public["PX"] - probs["PX"]),
                    abs(public["P2"] - probs["P2"]),
                )
            favorite_fail_fallback = min(1.0, 0.55 * public_bias_fallback + 0.45 * draw_context_strength)
            draw_bias_fallback = max(0.0, (public["PX"] - probs["PX"]) if public is not None else 0.0) * 0.9
            upset_fallback = min(1.0, 0.60 * pool_vs_model_bias + 0.40 * draw_context_strength)
            underdog_conversion_fallback = min(1.0, 0.45 * upset_fallback + 0.55 * max(0.0, probs["P2"] - 0.22))
            pool_bk_div_fallback = min(1.0, max(pool_vs_bk_divergence, pool_vs_model_bias * 0.8))

            history_signals["historical_favorite_fail_score"] = max(float(history_signals.get("historical_favorite_fail_score", 0.0)), favorite_fail_fallback)
            history_signals["historical_draw_bias_score"] = max(float(history_signals.get("historical_draw_bias_score", 0.0)), draw_bias_fallback)
            history_signals["historical_upset_score"] = max(float(history_signals.get("historical_upset_score", 0.0)), upset_fallback)
            history_signals["pool_vs_bk_historical_divergence"] = max(float(history_signals.get("pool_vs_bk_historical_divergence", 0.0)), pool_bk_div_fallback)
            history_signals["underdog_conversion_score"] = max(float(history_signals.get("underdog_conversion_score", 0.0)), underdog_conversion_fallback)
            history_signals["history_result_alignment_with_model"] = max(float(history_signals.get("history_result_alignment_with_model", 0.0)), min(1.0, public_bias_fallback + draw_bias_fallback * 0.6))
            history_signals["event_history_signal_strength"] = max(float(history_signals.get("event_history_signal_strength", 0.0)), min(1.0, 0.40 * favorite_fail_fallback + 0.35 * upset_fallback + 0.25 * draw_bias_fallback))
            history_signals["history_score"] = max(float(history_signals.get("history_score", 0.0)), min(1.0, 0.45 * favorite_fail_fallback + 0.35 * upset_fallback + 0.20 * draw_bias_fallback))
            history_signals["favorite_trap_score"] = max(float(history_signals.get("favorite_trap_score", 0.0)), favorite_fail_fallback)
            history_signals["upset_potential"] = max(float(history_signals.get("upset_potential", 0.0)), upset_fallback)
            history_signals["draw_tendency_score"] = max(float(history_signals.get("draw_tendency_score", 0.0)), draw_bias_fallback)
            history_signals["history_conflict_with_model"] = max(float(history_signals.get("history_conflict_with_model", 0.0)), min(1.0, public_bias_fallback + upset_fallback * 0.4))
            history_signals["public_bias_signal"] = max(float(history_signals.get("public_bias_signal", 0.0)), public_bias_fallback)
            history_signals["pool_vs_bk_gap_signal"] = max(float(history_signals.get("pool_vs_bk_gap_signal", 0.0)), pool_bk_div_fallback)

            signals_used = list(history_signals.get("signals_used", []))
            for fallback_signal in ("draw_context_prior", "draw_context_favorite_fail", "draw_context_upset", "draw_context_draw_bias"):
                if fallback_signal not in signals_used:
                    signals_used.append(fallback_signal)
            history_signals["signals_used"] = signals_used

        # ------------------------------------------------------------------
        # Override history_signals with precise values from _compute_event_history_signals
        # when global events were available (more accurate than derived stats).
        # ------------------------------------------------------------------
        if _global_event_sigs is not None:
            for _sk in (
                "historical_favorite_fail_score",
                "historical_draw_bias_score",
                "historical_upset_score",
                "pool_vs_bk_historical_divergence",
                "underdog_conversion_score",
            ):
                _sv = float(_global_event_sigs.get(_sk, 0.0))
                if _sv > 0.0:
                    history_signals[_sk] = _sv
            # Recompute composite signal strength with precise per-event values.
            history_signals["event_history_signal_strength"] = min(1.0, max(0.0,
                0.25 * float(history_signals.get("historical_favorite_fail_score", 0.0))
                + 0.25 * float(history_signals.get("historical_upset_score", 0.0))
                + 0.20 * float(history_signals.get("historical_draw_bias_score", 0.0))
                + 0.15 * float(history_signals.get("pool_vs_bk_historical_divergence", 0.0))
                + 0.15 * float(history_signals.get("underdog_conversion_score", 0.0))
            ))
            signals_used_ev = list(history_signals.get("signals_used", []))
            for _esig in ("event_history_precise_signals", "event_history_pool_priors"):
                if _esig not in signals_used_ev:
                    signals_used_ev.append(_esig)
            history_signals["signals_used"] = signals_used_ev

        if history_signals["signals_used"]:
            strategy_signals_used.update(history_signals["signals_used"])

        signal_source = (
            "event_history_pool_priors" if "event_history_pool_priors" in strategy_signals_used
            else "pool_probs_dict" if "pool_probs_dict" in strategy_signals_used
            else "explicit_strategy_dict" if any(k in merged for k in ("strategy_signals", "history_signals"))
            else "bk_or_implied_probs" if public is not None
            else "none"
        )

        favorite_overload = max(
            float(merged.get("favorite_overload", merged.get("crowd_overconfidence", 0.0))),
            float(history_signals["favorite_overload_signal"]),
        )
        upset_tendency = max(
            float(merged.get("upset_tendency", merged.get("underdog_bias", 0.0))),
            float(history_signals["upset_history_signal"]),
        )

        # Combined availability check: any signal source counts.
        has_signals = bool(merged) or (public is not None) or history_ready or (draws_loaded > 0)
        if events_loaded > 0:
            history_mode = "event_history_ready"
        elif draws_loaded > 0 and public is not None:
            history_mode = "draw_history_with_pool_signals"
        elif draws_loaded > 0:
            history_mode = "draw_history_only"
        elif public is not None:
            history_mode = "pool_public_only"
        else:
            history_mode = "no_history"

        if events_loaded > 0:
            strategy_signals_used.add("event_history_ready")

        event_history_signal_strength = float(history_signals.get("event_history_signal_strength", 0.0))

        if reason == "history_unavailable" and draws_loaded > 0:
            reason = "draw_context_prior"

        payload_toto_brief_history = match.get("toto_brief_history")
        payload_pool_probs = match.get("pool_probs") if isinstance(match.get("pool_probs"), dict) else {}
        if not payload_pool_probs and public is not None:
            payload_pool_probs = {
                "P1": float(public["P1"]),
                "PX": float(public["PX"]),
                "P2": float(public["P2"]),
            }
        # Fallback: use historical aggregate pool probs when available
        if not payload_pool_probs and _global_event_sigs is not None:
            _pool_avg_pay = _global_event_sigs.get("pool_avg_probs")
            if _pool_avg_pay is not None:
                payload_pool_probs = {
                    "P1": float(_pool_avg_pay["P1"]),
                    "PX": float(_pool_avg_pay["PX"]),
                    "P2": float(_pool_avg_pay["P2"]),
                }
        payload_pool_quotes = match.get("pool_quotes") if isinstance(match.get("pool_quotes"), dict) else {}
        if not payload_pool_quotes and public is not None:
            payload_pool_quotes = {
                "pool_win_1": float(public["P1"]) * 100.0,
                "pool_draw": float(public["PX"]) * 100.0,
                "pool_win_2": float(public["P2"]) * 100.0,
            }
        if not payload_pool_quotes and _global_event_sigs is not None:
            _pool_avg_pay = _global_event_sigs.get("pool_avg_probs")
            if _pool_avg_pay is not None:
                payload_pool_quotes = {
                    "pool_win_1": float(_pool_avg_pay["P1"]) * 100.0,
                    "pool_draw": float(_pool_avg_pay["PX"]) * 100.0,
                    "pool_win_2": float(_pool_avg_pay["P2"]) * 100.0,
                }
        payload_bookmaker_quotes = match.get("bookmaker_quotes") if isinstance(match.get("bookmaker_quotes"), dict) else {}
        norm_bk = match.get("norm_bookmaker_probs")
        payload_bookmaker_probs = norm_bk if isinstance(norm_bk, dict) else {}
        if not payload_bookmaker_probs and isinstance(match.get("bookmaker_probs"), dict):
            payload_bookmaker_probs = match.get("bookmaker_probs")
        if not payload_bookmaker_probs and bk_probs is not None:
            payload_bookmaker_probs = {
                "P1": float(bk_probs["P1"]),
                "PX": float(bk_probs["PX"]),
                "P2": float(bk_probs["P2"]),
            }
        # Fallback: use historical aggregate bk probs when available
        if not payload_bookmaker_probs and _global_event_sigs is not None:
            _bk_avg_pay = _global_event_sigs.get("bk_avg_probs")
            if _bk_avg_pay is not None:
                payload_bookmaker_probs = {
                    "P1": float(_bk_avg_pay["P1"]),
                    "PX": float(_bk_avg_pay["PX"]),
                    "P2": float(_bk_avg_pay["P2"]),
                }
        if not payload_bookmaker_quotes and bk_probs is not None:
            payload_bookmaker_quotes = {
                "bk_win_1": float(bk_probs["P1"]) * 100.0,
                "bk_draw": float(bk_probs["PX"]) * 100.0,
                "bk_win_2": float(bk_probs["P2"]) * 100.0,
            }
        if not payload_bookmaker_quotes and _global_event_sigs is not None:
            _bk_avg_pay = _global_event_sigs.get("bk_avg_probs")
            if _bk_avg_pay is not None:
                payload_bookmaker_quotes = {
                    "bk_win_1": float(_bk_avg_pay["P1"]) * 100.0,
                    "bk_draw": float(_bk_avg_pay["PX"]) * 100.0,
                    "bk_win_2": float(_bk_avg_pay["P2"]) * 100.0,
                }

        return {
            "available": has_signals,
            "history_ready": history_ready,
            "history_state": history_state,
            "history_empty_reason": history_empty_reason,
            "history_mode": history_mode,
            "history_context_used": bool(draws_loaded > 0 or events_loaded > 0 or public is not None),
            "global_history_context_loaded": global_history_context_loaded,
            "global_history_injected": global_history_injected,
            "public_probs": public,
            "bk_probs": bk_probs,
            "pool_public_used": bool(public is not None),
            "bookmaker_used": bool(bk_probs is not None),
            "event_history_used": bool(events_loaded > 0 and history_ready),
            "pool_vs_model_bias": pool_vs_model_bias,
            "pool_vs_bk_divergence": pool_vs_bk_divergence,
            "crowd_favorite": crowd_favorite,
            "crowd_favorite_competitive": crowd_favorite_competitive,
            "contrarian_outcome": contrarian,
            "favorite_overload": favorite_overload,
            "upset_tendency": upset_tendency,
            "underdog_neglect": float(history_signals["underdog_neglect_signal"]),
            "draw_underpricing": float(history_signals["draw_underpricing_signal"]),
            "pool_vs_bk_gap": float(history_signals["pool_vs_bk_gap_signal"]),
            "pool_vs_bookmaker_divergence": float(history_signals["pool_vs_bk_gap_signal"]),
            "public_bias": float(history_signals["public_bias_signal"]),
            "history_score": float(history_signals["history_score"]),
            "favorite_trap_score": float(history_signals["favorite_trap_score"]),
            "upset_potential": float(history_signals["upset_potential"]),
            "draw_tendency_score": float(history_signals["draw_tendency_score"]),
            "history_conflict_with_model": float(history_signals["history_conflict_with_model"]),
            "historical_favorite_fail_score": float(history_signals.get("historical_favorite_fail_score", 0.0)),
            "historical_draw_bias_score": float(history_signals.get("historical_draw_bias_score", 0.0)),
            "historical_upset_score": float(history_signals.get("historical_upset_score", 0.0)),
            "pool_vs_bk_historical_divergence": float(history_signals.get("pool_vs_bk_historical_divergence", 0.0)),
            "underdog_conversion_score": float(history_signals.get("underdog_conversion_score", 0.0)),
            "history_result_alignment_with_model": float(history_signals.get("history_result_alignment_with_model", 0.0)),
            "event_history_signal_strength": event_history_signal_strength,
            "payload_toto_brief_history": payload_toto_brief_history if isinstance(payload_toto_brief_history, dict) else {},
            "payload_pool_probs": payload_pool_probs,
            "payload_pool_quotes": payload_pool_quotes,
            "payload_bookmaker_probs": payload_bookmaker_probs,
            "payload_bookmaker_quotes": payload_bookmaker_quotes,
            "reason": reason,
            "history_reason": reason,
            "signal_source": signal_source,
            "strategy_signals_used": sorted(strategy_signals_used),
            "history_draws_loaded_count": draws_loaded,
            "history_events_loaded_count": events_loaded,
        }

    def _apply_history_strategy(
        self,
        decision: str,
        decision_class: str,
        profile: dict[str, Any],
        strategy: dict[str, Any],
    ) -> dict[str, Any]:
        if not strategy["available"]:
            return {
                "decision": decision,
                "history_used": False,
                "reason": "history_unavailable",
                "history_adjustment": 0.0,
                "market_adjustment": 0.0,
            }

        probs = profile["probs"]
        top_outcome = str(profile["top_outcome"])
        contrarian = str(strategy["contrarian_outcome"])
        crowd_favorite = str(strategy["crowd_favorite"])
        favorite_overload = float(strategy["favorite_overload"])
        upset_tendency = float(strategy["upset_tendency"])
        underdog_neglect = float(strategy.get("underdog_neglect", 0.0))
        draw_underpricing = float(strategy.get("draw_underpricing", 0.0))
        pool_vs_bk_gap = float(strategy.get("pool_vs_bk_gap", 0.0))
        pool_vs_model_bias = float(strategy.get("pool_vs_model_bias", 0.0))
        pool_vs_bk_div = float(strategy.get("pool_vs_bk_divergence", 0.0))
        history_score = float(strategy.get("history_score", 0.0))
        history_conflict = float(strategy.get("history_conflict_with_model", 0.0))
        public_bias = float(strategy.get("public_bias", 0.0))
        crowd_favorite_competitive = bool(strategy.get("crowd_favorite_competitive", False))
        history_mode = str(strategy.get("history_mode", "no_history"))
        event_history_signal_strength = float(strategy.get("event_history_signal_strength", 0.0))

        # Rich event-history signals — only populated when event_history_ready
        historical_favorite_fail_score = float(strategy.get("historical_favorite_fail_score", 0.0))
        historical_draw_bias_score = float(strategy.get("historical_draw_bias_score", 0.0))
        historical_upset_score = float(strategy.get("historical_upset_score", 0.0))
        pool_vs_bk_historical_divergence = float(strategy.get("pool_vs_bk_historical_divergence", 0.0))
        underdog_conversion_score = float(strategy.get("underdog_conversion_score", 0.0))

        market_adjustment = min(1.0, max(0.0, public_bias + pool_vs_bk_gap + pool_vs_model_bias * 0.5))
        history_adjustment = min(1.0, max(0.0, history_score + history_conflict))

        # When event-history is available, boost adjustments with rich signals.
        if history_mode == "event_history_ready":
            history_adjustment = min(1.0, history_adjustment + event_history_signal_strength * 0.5)
            market_adjustment = min(1.0, market_adjustment + historical_favorite_fail_score * 0.4)

        # Adaptive threshold based on history mode.
        # event_history_ready: most aggressive (real Baltbet data)
        # draw_history_with_pool_signals: moderate
        # draw_history_only or partial: slightly lower than default
        if history_mode == "event_history_ready":
            pool_vs_model_threshold = 0.035
            fav_trap_threshold = 0.04
            upset_threshold = 0.04
            draw_underpricing_threshold = 0.04
            underdog_threshold = 0.04
            market_adj_threshold = 0.09
        elif history_mode in ("draw_history_with_pool_signals", "draw_history_only"):
            pool_vs_model_threshold = 0.040
            fav_trap_threshold = 0.050
            upset_threshold = 0.050
            draw_underpricing_threshold = 0.050
            underdog_threshold = 0.050
            market_adj_threshold = 0.10
        else:
            pool_vs_model_threshold = 0.06
            fav_trap_threshold = 0.06
            upset_threshold = 0.06
            draw_underpricing_threshold = 0.06
            underdog_threshold = 0.06
            market_adj_threshold = 0.12

        # ------------------------------------------------------------------
        # DRAW-CONTEXT-ONLY PATH
        # When only aggregate draw history exists (no event rows, no public probs),
        # still allow conservative history-driven widening to avoid decorative history.
        # ------------------------------------------------------------------
        if history_mode == "draw_history_only" and strategy.get("public_probs") is None:
            second_outcome = str(profile["second_outcome"])
            if (
                decision_class.startswith("strong_single_")
                and history_adjustment >= 0.08
                and probs[_PROB_KEY[second_outcome]] >= float(profile["top_prob"]) - 0.16
            ):
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary=("X" if draw_underpricing >= 0.04 else second_outcome),
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "draw_context_prior_shift",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

        # ------------------------------------------------------------------
        # EVENT HISTORY PATH (highest priority when event_history_ready).
        # When real Baltbet historical event data is available, use richer
        # signal set to determine if model decision should be widened.
        # ------------------------------------------------------------------
        if history_mode == "event_history_ready" and event_history_signal_strength > 0.0:
            # EH-1: Historical favorite trap — overloaded favorite in a pool where
            # history shows favorites frequently fail.
            if (historical_favorite_fail_score >= fav_trap_threshold
                    and decision_class.startswith("strong_single_")
                    and contrarian != top_outcome
                    and probs[_PROB_KEY[contrarian]] >= float(profile["top_prob"]) - 0.14):
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary=contrarian,
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "event_history_favorite_trap",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

            # EH-2: Historical draw bias — draw rate in history exceeds model's PX.
            # Give draw a real chance, especially in uncertain or balanced matches.
            if (historical_draw_bias_score >= fav_trap_threshold
                    and probs["PX"] >= float(profile["top_prob"]) - 0.15
                    and "X" not in decision):
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary="X",
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "event_history_draw_bias",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

            # EH-3: Underdog conversion — history shows underdogs converting vs pool.
            if (underdog_conversion_score >= upset_threshold
                    and historical_upset_score >= upset_threshold):
                underdog = "2" if probs["P2"] >= probs["P1"] - 0.08 else contrarian
                if underdog not in decision and probs[_PROB_KEY[underdog]] >= float(profile["top_prob"]) - 0.18:
                    return {
                        "decision": self._select_double_type(
                            primary=top_outcome,
                            secondary=underdog,
                            probs=probs,
                            draw_underpricing=draw_underpricing,
                        ),
                        "history_used": True,
                        "reason": "event_history_underdog_conversion",
                        "history_adjustment": history_adjustment,
                        "market_adjustment": market_adjustment,
                    }

            # EH-4: Pool vs bookmaker historical divergence — systematic market
            # disagreement implies hidden uncertainty not captured by model.
            if (pool_vs_bk_historical_divergence >= underdog_threshold
                    and crowd_favorite != top_outcome
                    and crowd_favorite_competitive):
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary=crowd_favorite,
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "event_history_pool_bk_divergence",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

        # ------------------------------------------------------------------
        # SIGNAL 1 (highest priority): Pool vs model divergence.
        # When pool distribution clearly differs from model prediction and the
        # pool's preferred outcome is plausible, form a double covering both.
        # This is the primary mechanism for multi-signal decision making.
        # ------------------------------------------------------------------
        combined_pressure = max(
            history_adjustment,
            market_adjustment,
            history_conflict + public_bias * 0.5,
            event_history_signal_strength,
        )
        if history_mode in ("event_history_ready", "draw_history_with_pool_signals") and combined_pressure >= 0.16:
            if contrarian != top_outcome and probs[_PROB_KEY[contrarian]] >= float(profile["top_prob"]) - 0.14:
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary=contrarian,
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "history_pressure_alignment",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

        if pool_vs_model_bias >= pool_vs_model_threshold and crowd_favorite != top_outcome and crowd_favorite_competitive:
            return {
                "decision": self._select_double_type(
                    primary=top_outcome,
                    secondary=crowd_favorite,
                    probs=probs,
                    draw_underpricing=draw_underpricing,
                ),
                "history_used": True,
                "reason": "pool_vs_model_divergence",
                "history_adjustment": history_adjustment,
                "market_adjustment": market_adjustment,
            }

        # ------------------------------------------------------------------
        # SIGNAL 2: High uncertainty - unconditionally use history/pool context.
        # ------------------------------------------------------------------
        if decision_class == "high_uncertainty" and strategy["available"]:
            target_contrarian = crowd_favorite if crowd_favorite != top_outcome else contrarian
            if target_contrarian != top_outcome and probs[_PROB_KEY[target_contrarian]] >= float(profile["top_prob"]) - 0.14:
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary=target_contrarian,
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "high_uncertainty_with_pool_context",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

        # ------------------------------------------------------------------
        # SIGNAL 3: Contrarian competitiveness check for remaining signals.
        # ------------------------------------------------------------------
        competitive_contrarian = probs[_PROB_KEY[contrarian]] >= (float(profile["top_prob"]) - 0.10)
        if not competitive_contrarian and pool_vs_model_bias < 0.05 and market_adjustment < 0.10:
            return {
                "decision": decision,
                "history_used": False,
                "reason": "contrarian_not_competitive",
                "history_adjustment": history_adjustment,
                "market_adjustment": market_adjustment,
            }

        # ------------------------------------------------------------------
        # SIGNAL 4: Favorite trap / upset bias from history.
        # ------------------------------------------------------------------
        if decision_class.startswith("strong_single_"):
            if (favorite_overload >= fav_trap_threshold or upset_tendency >= upset_threshold) and contrarian != top_outcome and competitive_contrarian:
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary=contrarian,
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "favorite_trap_or_upset_bias",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }
            if draw_underpricing >= draw_underpricing_threshold and probs["PX"] >= float(profile["top_prob"]) - 0.12:
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary="X",
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "draw_tendency_history_shift",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

        # ------------------------------------------------------------------
        # SIGNAL 5: Existing double + contrarian not covered + strong signal.
        # ------------------------------------------------------------------
        if len(decision) == 2 and contrarian not in decision and upset_tendency >= 0.07 and competitive_contrarian:
            return {
                "decision": self._select_double_type(
                    primary=contrarian,
                    secondary=top_outcome,
                    probs=probs,
                    draw_underpricing=draw_underpricing,
                ),
                "history_used": True,
                "reason": "double_shift_to_contrarian",
                "history_adjustment": history_adjustment,
                "market_adjustment": market_adjustment,
            }

        # ------------------------------------------------------------------
        # SIGNAL 6: Pool vs bookmaker gap + underdog neglect.
        # ------------------------------------------------------------------
        if (pool_vs_bk_gap >= 0.07 or pool_vs_bk_div >= 0.07) and underdog_neglect >= underdog_threshold:
            underdog = "2" if probs["P2"] >= probs["P1"] - 0.08 else contrarian
            if underdog not in decision and probs[_PROB_KEY[underdog]] >= float(profile["top_prob"]) - 0.16:
                return {
                    "decision": self._select_double_type(
                        primary=top_outcome,
                        secondary=underdog,
                        probs=probs,
                        draw_underpricing=draw_underpricing,
                    ),
                    "history_used": True,
                    "reason": "pool_vs_market_gap",
                    "history_adjustment": history_adjustment,
                    "market_adjustment": market_adjustment,
                }

        # ------------------------------------------------------------------
        # SIGNAL 7: Market adjustment threshold - any strong combined signal.
        # ------------------------------------------------------------------
        if market_adjustment >= market_adj_threshold and contrarian != top_outcome and competitive_contrarian:
            return {
                "decision": self._select_double_type(
                    primary=top_outcome,
                    secondary=contrarian,
                    probs=probs,
                    draw_underpricing=draw_underpricing,
                ),
                "history_used": True,
                "reason": "market_adjustment_threshold",
                "history_adjustment": history_adjustment,
                "market_adjustment": market_adjustment,
            }

        return {
            "decision": decision,
            "history_used": False,
            "reason": "history_available_no_change",
            "history_adjustment": history_adjustment,
            "market_adjustment": market_adjustment,
        }

    def _select_double_type(
        self,
        primary: str,
        secondary: str,
        probs: dict[str, float],
        draw_underpricing: float = 0.0,
    ) -> str:
        """Choose the semantically correct double type.

        - 1X: home hedge (pool/history pushes toward draw while model favors home)
        - X2: draw/guest spread (pool/history pushes toward away/uncertain outcome)
        - 12: home vs away (draw is saturated / pool-neutral on draw)
        """
        if primary == secondary:
            return primary
        # Canonicalize: put "1" first when present.
        if secondary == "1" and primary != "1":
            primary, secondary = secondary, primary

        pair = frozenset({primary, secondary})

        # 1X: home + draw hedge
        if pair == frozenset({"1", "X"}):
            return "1X"

        # X2: draw + away spread
        if pair == frozenset({"X", "2"}):
            return "X2"

        # 12: home vs away — draw is not the hedge target
        if pair == frozenset({"1", "2"}):
            return "12"

        # Fallback (should not happen with valid outcomes).
        return self._decision_from_outcomes([primary, secondary])

    # ---------------------------------------------------------------------
    # Pool builders
    # ---------------------------------------------------------------------

    def _base_pool(self, match: dict, decision: str) -> list[str]:
        profile = self._decision_profile(match)
        probs = profile["probs"]
        options = self._options_from_decision(decision=decision, probs=probs)
        if len(options) == 1:
            second = profile["ranked"][1][0]
            if second != options[0] and (
                float(profile["margin_top_second"]) <= 0.09 or float(profile["uncertainty"]) >= 0.83
            ):
                options.append(second)
        return list(dict.fromkeys(options))[:2]

    def _insurance_pool(
        self,
        match: dict,
        decision: str,
        insurance_strength: float,
        strategy: dict[str, Any] | None = None,
    ) -> list[str]:
        profile = self._decision_profile(match)
        probs = profile["probs"]
        ranked = profile["ranked"]
        single_pick = self._single_from_decision(decision=decision, probs=probs)
        options: list[str] = [single_pick]

        second_outcome, second_prob = ranked[1]
        third_outcome, third_prob = ranked[2]
        top_prob = ranked[0][1]
        top_gap_second = top_prob - second_prob
        top_gap_third = top_prob - third_prob

        if strategy is None:
            strategy = self._extract_strategy_signals(match=match, probs=probs)

        history_mode = str(strategy.get("history_mode", "no_history"))
        event_history_signal_strength = float(strategy.get("event_history_signal_strength", 0.0))
        history_conflict = float(strategy.get("history_conflict_with_model", 0.0))
        aggressive_history = (
            history_mode == "event_history_ready"
            and (event_history_signal_strength >= 0.14 or history_conflict >= 0.12)
        )

        strategy_risk_boost = min(
            0.20,
            0.03 * float(strategy.get("favorite_overload", 0.0))
            + 0.03 * float(strategy.get("upset_tendency", 0.0))
            + 0.03 * float(strategy.get("pool_vs_model_bias", 0.0))
            + 0.02 * float(strategy.get("underdog_neglect", 0.0))
            + 0.02 * float(strategy.get("pool_vs_bk_gap", 0.0))
            + 0.03 * float(strategy.get("historical_favorite_fail_score", 0.0))
            + 0.03 * float(strategy.get("historical_draw_bias_score", 0.0))
            + 0.03 * float(strategy.get("historical_upset_score", 0.0))
            + 0.03 * float(strategy.get("pool_vs_bk_historical_divergence", 0.0))
            + 0.03 * float(strategy.get("underdog_conversion_score", 0.0)),
        )

        reason_scores = self._insurance_reason_scores(
            profile=profile,
            strategy=strategy,
            base_single=single_pick,
        )

        opposite = self._opposite_outcome(single_pick)

        def _opposite_is_strongly_justified() -> bool:
            if not opposite:
                return False
            # Strong opposite (1<->2) must be justified by a combined conflict profile,
            # not by a single weak signal.
            strongest = max(
                float(reason_scores.get("favorite_fail_risk", 0.0)),
                float(reason_scores.get("upset_risk", 0.0)),
                float(reason_scores.get("public_trap_risk", 0.0)),
                float(reason_scores.get("history_conflict", 0.0)),
                float(reason_scores.get("public_conflict", 0.0)),
            )
            combined = (
                0.30 * float(reason_scores.get("favorite_fail_risk", 0.0))
                + 0.30 * float(reason_scores.get("upset_risk", 0.0))
                + 0.20 * float(reason_scores.get("history_conflict", 0.0))
                + 0.20 * float(reason_scores.get("public_trap_risk", 0.0))
            )
            event_history_signal = float(strategy.get("event_history_signal_strength", 0.0))
            hard_conflict = (
                float(strategy.get("history_conflict_with_model", 0.0)) >= 0.16
                or event_history_signal >= 0.18
            )

            if insurance_strength >= 0.90:
                return (strongest >= 0.32 and combined >= 0.22) or hard_conflict
            return (strongest >= 0.40 and combined >= 0.28) or hard_conflict

        opposite_allowed = _opposite_is_strongly_justified()

        # Tier 1 (all strengths >= 0.3): add second outcome if close enough.
        second_margin_limit = 0.16 - (0.10 * insurance_strength)
        second_prob_floor = 0.27 - (0.12 * insurance_strength)
        if second_outcome != single_pick and (
            top_gap_second <= second_margin_limit + strategy_risk_boost
            or second_prob >= second_prob_floor - strategy_risk_boost
        ):
            options.append(second_outcome)

        # Draw-first safety: for strong favorites, prefer X as first alternative
        # before opposite side unless opposite is strongly justified.
        draw_score = float(reason_scores.get("draw_risk", 0.0))

        if single_pick in {"1", "2"} and insurance_strength >= 0.50 and "X" not in options:
            if draw_score >= (0.26 - 0.10 * insurance_strength):
                options.append("X")

        # Tier 2 (>= 0.50): add contrarian from strategy signals.
        if strategy["available"] and insurance_strength >= 0.50:
            contrarian = str(strategy["contrarian_outcome"])
            crowd_fav = str(strategy.get("crowd_favorite", contrarian))
            pool_bias = float(strategy.get("pool_vs_model_bias", 0.0))

            # Prefer crowd_favorite over contrarian when pool clearly goes there.
            primary_alt = crowd_fav if (pool_bias >= 0.05 and crowd_fav != single_pick) else contrarian

            if primary_alt not in options and probs[_PROB_KEY[primary_alt]] >= (
                top_prob - (0.13 + strategy_risk_boost)
            ):
                # Strong opposite (1<->2) requires stronger justification.
                if opposite and primary_alt == opposite:
                    if not opposite_allowed:
                        primary_alt = ""
                if primary_alt:
                    options.append(primary_alt)

        # Tier 3 (>= 0.70): add third ranked outcome.
        if insurance_strength >= 0.70 and third_outcome not in options:
            third_threshold_prob = 0.18 - 0.06 * insurance_strength - strategy_risk_boost
            third_margin_limit = 0.24 - 0.14 * insurance_strength
            if insurance_strength >= 0.90:
                third_threshold_prob -= 0.04
            elif aggressive_history:
                third_threshold_prob -= 0.02
            if (
                third_prob >= third_threshold_prob
                or top_gap_third <= third_margin_limit + strategy_risk_boost
            ):
                if opposite and third_outcome == opposite:
                    if not opposite_allowed:
                        pass
                    else:
                        options.append(third_outcome)
                else:
                    options.append(third_outcome)

        # Tier 4 (>= 0.90): AGGRESSIVE - at very high strength, force all 3 outcomes
        # for matches with high history conflict or pool divergence. The qualitative
        # difference: 0.7 gives 2-3 outcomes for risk matches, 0.9 forces 3 for all
        # high-conflict matches regardless of prob margins.
        if insurance_strength >= 0.90:
            pool_bias = float(strategy.get("pool_vs_model_bias", 0.0))
            if history_conflict >= 0.10 or pool_bias >= 0.07 or event_history_signal_strength >= 0.14:
                # Force all 3 outcomes.
                for outcome in ("1", "X", "2"):
                    if opposite and outcome == opposite and not opposite_allowed:
                        continue
                    if outcome not in options:
                        options.append(outcome)

        ordered = sorted(set(options), key=lambda outcome: probs[_PROB_KEY[outcome]], reverse=True)
        # Max outcomes by strength tier:
        if insurance_strength >= 0.90:
            max_options = 3
        elif insurance_strength >= 0.70:
            # Keep 0.7 moderate; allow 3 only on strong history/event conflict.
            max_options = 3 if (aggressive_history or strategy_risk_boost >= 0.12) else 2
        else:
            max_options = 2
        return ordered[:max_options]

    # ---------------------------------------------------------------------
    # Coupon assembly
    # ---------------------------------------------------------------------

    def _assemble_coupons(
        self,
        pools: list[list[str]],
        matches: list[dict],
        target_lines: int,
        diversity_strength: float,
    ) -> list[list[str]]:
        candidates = [list(line) for line in itertools.product(*pools)]
        if not candidates:
            return []

        weighted_candidates: list[tuple[list[str], float]] = []
        for coupon in candidates:
            if len(coupon) != len(matches):
                continue
            log_weight = self._coupon_log_weight(coupon=coupon, matches=matches)
            weighted_candidates.append((coupon, log_weight))

        weighted_candidates.sort(key=lambda item: item[1], reverse=True)

        selected: list[list[str]] = []
        selected_keys: set[tuple[str, ...]] = set()
        per_column_usage: list[dict[str, int]] = [dict() for _ in range(len(matches))]
        variable_columns = sum(1 for pool in pools if len(set(pool)) > 1)
        if target_lines >= 32:
            if variable_columns >= 6:
                min_distance_floor = 3
            elif variable_columns >= 3:
                min_distance_floor = 2
            else:
                min_distance_floor = 1
        else:
            min_distance_floor = 2 if variable_columns >= 5 else 1

        while weighted_candidates and len(selected) < target_lines:
            best_index_strict = -1
            best_score_strict = float("-inf")
            best_index_relaxed = -1
            best_score_relaxed = float("-inf")
            search_window = min(len(weighted_candidates), max(target_lines * 12, 64))
            for idx in range(search_window):
                coupon, log_weight = weighted_candidates[idx]
                score = log_weight + self._diversity_bonus(
                    coupon=coupon,
                    per_column_usage=per_column_usage,
                    strength=diversity_strength,
                )

                min_distance = len(coupon)
                if selected:
                    distances = [self._hamming_distance(coupon, existing) for existing in selected]
                    min_distance = min(distances)
                    avg_distance = sum(distances) / len(distances)
                    score += (min_distance * 0.06) + (avg_distance * 0.015)
                    if min_distance <= 1:
                        score -= 0.22 + (0.03 * len(selected))
                    elif min_distance == 2:
                        score -= 0.07

                if (not selected) or (min_distance >= min_distance_floor):
                    if score > best_score_strict:
                        best_score_strict = score
                        best_index_strict = idx
                elif score > best_score_relaxed:
                    best_score_relaxed = score
                    best_index_relaxed = idx

            pick_index = best_index_strict if best_index_strict >= 0 else best_index_relaxed
            if pick_index < 0:
                break

            coupon, _ = weighted_candidates.pop(pick_index)
            key = tuple(coupon)
            if key in selected_keys:
                continue
            selected_keys.add(key)
            selected.append(coupon)
            for col_idx, outcome in enumerate(coupon):
                usage = per_column_usage[col_idx]
                usage[outcome] = usage.get(outcome, 0) + 1

        return selected

    def _build_insured_coupons_from_base(
        self,
        base_coupons: list[list[str]],
        insured_pools: list[list[str]],
        insured_matches: dict[int, dict[str, Any]],
        strategy_risk_scores: dict[int, float],
        insurance_strength: float,
        target_count: int,
    ) -> tuple[list[list[str]], list[list[int]]]:
        if target_count <= 0 or not base_coupons or not insured_matches:
            return [], []

        ranked_matches = sorted(
            insured_matches.keys(),
            key=lambda idx: strategy_risk_scores.get(idx, 0.0),
            reverse=True,
        )
        if not ranked_matches:
            return [], []

        # Insurance strength tiers - clear qualitative difference:
        # 0.3: change 1 cell per insured coupon
        # 0.5: change 1-2 cells
        # 0.7: force at least 2 meaningful changes
        # 0.9: force at least 3 meaningful changes
        if insurance_strength >= 0.90:
            max_changes_per_coupon = 4
            min_changes_per_coupon = 3
        elif insurance_strength >= 0.70:
            max_changes_per_coupon = 2
            min_changes_per_coupon = 2
        else:
            max_changes_per_coupon = 1
            min_changes_per_coupon = 1

        if insurance_strength >= 0.90:
            influence = 0.95
        elif insurance_strength >= 0.70:
            influence = 0.72
        elif insurance_strength >= 0.55:
            influence = 0.58
        else:
            influence = 0.25 + 0.70 * insurance_strength

        # Semantic diversification: at high strength, prioritize matches that
        # have pool divergence signals (not just highest entropy).
        # We already have ranked_matches sorted by risk_score which incorporates
        # strategy signals, so the order is already signal-aware.

        insured_coupons: list[list[str]] = []
        changed_matches_by_line: list[list[int]] = []
        seen: set[tuple[str, ...]] = {tuple(c) for c in base_coupons}
        seen_change_signatures: dict[tuple[int, ...], int] = {}

        if insurance_strength >= 0.90:
            max_signature_reuse = 1
            min_coupon_distance = 3
        elif insurance_strength >= 0.70:
            max_signature_reuse = 2
            min_coupon_distance = 2
        else:
            max_signature_reuse = 3
            min_coupon_distance = 1

        attempt_cap = max(target_count * 10, 48)
        attempts = 0

        while len(insured_coupons) < target_count and attempts < attempt_cap:
            attempts += 1
            line_idx = len(insured_coupons)
            base_coupon = list(base_coupons[line_idx % len(base_coupons)])
            modified = list(base_coupon)
            changed: list[int] = []

            # Rotate through ranked matches differently per line to create
            # semantic diversity across insured coupons.
            start_rank = line_idx % max(len(ranked_matches), 1)
            rotated = ranked_matches[start_rank:] + ranked_matches[:start_rank]

            # 0.9 profile: ensure broader practical spread across insured targets
            # by forcing one rotating seed change before gated mutations.
            if insurance_strength >= 0.90 and rotated:
                for seed_rank, seed_idx in enumerate(rotated):
                    if seed_idx >= len(modified):
                        continue
                    seed_info = insured_matches[seed_idx]
                    seed_alts = [o for o in seed_info.get("alternatives", []) if o in ALLOWED_OUTCOMES]
                    seed_alts = [o for o in seed_alts if o in insured_pools[seed_idx] and o != modified[seed_idx]]
                    if not seed_alts:
                        continue
                    seed_pick = seed_alts[(line_idx + seed_rank) % len(seed_alts)]
                    if seed_pick == modified[seed_idx]:
                        continue
                    modified[seed_idx] = seed_pick
                    changed.append(seed_idx)
                    break

            for rank, match_idx in enumerate(rotated):
                if len(changed) >= max_changes_per_coupon:
                    break
                if match_idx >= len(modified):
                    continue
                if match_idx in changed:
                    continue

                info = insured_matches[match_idx]
                alternatives = [o for o in info.get("alternatives", []) if o in ALLOWED_OUTCOMES]
                if not alternatives:
                    continue

                # Gate mechanism scaled by influence.
                gate = (line_idx + rank) % 10
                threshold = max(1, int(round(influence * 10)))
                if gate >= threshold:
                    continue

                valid_alternatives = [o for o in alternatives if o in insured_pools[match_idx]]
                if not valid_alternatives:
                    continue

                opposite = self._opposite_outcome(str(info.get("base_outcome", "")))
                reason_scores = info.get("reason_scores", {})
                if not isinstance(reason_scores, dict):
                    reason_scores = {}
                opposite_justification = max(
                    self._safe_float(reason_scores.get("favorite_fail_risk")),
                    self._safe_float(reason_scores.get("upset_risk")),
                    self._safe_float(reason_scores.get("history_conflict")),
                    self._safe_float(reason_scores.get("public_trap_risk")),
                )

                # 0.7 path is conservative (adjacent first); 0.9 can promote opposite
                # only when justification is materially strong.
                if opposite and opposite in valid_alternatives:
                    if insurance_strength >= 0.90 and opposite_justification >= 0.24:
                        valid_alternatives = sorted(
                            valid_alternatives,
                            key=lambda outcome: 0 if outcome == opposite else 1,
                        )
                    else:
                        valid_alternatives = sorted(
                            valid_alternatives,
                            key=lambda outcome: 1 if outcome == opposite else 0,
                        )

                # Cycle through alternatives across different coupon lines for semantic spread.
                pick_idx = (line_idx + rank * 3) % len(valid_alternatives)
                alt_outcome = valid_alternatives[pick_idx]
                if modified[match_idx] == alt_outcome:
                    if len(valid_alternatives) > 1:
                        alt_outcome = valid_alternatives[(pick_idx + 1) % len(valid_alternatives)]
                    else:
                        continue

                if modified[match_idx] == alt_outcome:
                    continue

                modified[match_idx] = alt_outcome
                changed.append(match_idx)

            if not changed:
                continue
            if len(changed) < min_changes_per_coupon:
                continue

            change_signature = tuple(sorted(changed))
            if seen_change_signatures.get(change_signature, 0) >= max_signature_reuse:
                continue

            distance_reference = list(base_coupons) + list(insured_coupons)
            if distance_reference:
                nearest = min(self._hamming_distance(modified, existing) for existing in distance_reference)
                if nearest < min_coupon_distance:
                    continue

            key = tuple(modified)
            if key in seen:
                continue
            seen.add(key)
            insured_coupons.append(modified)
            changed_matches_by_line.append(changed)
            seen_change_signatures[change_signature] = seen_change_signatures.get(change_signature, 0) + 1

        # Fallback: force minimal changes on top-risk matches when we still need more.
        fallback_idx = 0
        while len(insured_coupons) < target_count and ranked_matches:
            base_coupon = list(base_coupons[fallback_idx % len(base_coupons)])
            modified = list(base_coupon)
            changed: list[int] = []

            for rank, match_idx in enumerate(ranked_matches):
                if len(changed) >= max_changes_per_coupon:
                    break
                if match_idx >= len(modified):
                    continue

                info = insured_matches[match_idx]
                alternatives = [o for o in info.get("alternatives", []) if o in ALLOWED_OUTCOMES]
                if not alternatives:
                    continue

                opposite = self._opposite_outcome(str(info.get("base_outcome", "")))
                reason_scores = info.get("reason_scores", {})
                if not isinstance(reason_scores, dict):
                    reason_scores = {}
                opposite_justification = max(
                    self._safe_float(reason_scores.get("favorite_fail_risk")),
                    self._safe_float(reason_scores.get("upset_risk")),
                    self._safe_float(reason_scores.get("history_conflict")),
                    self._safe_float(reason_scores.get("public_trap_risk")),
                )
                if opposite and opposite in alternatives:
                    if insurance_strength >= 0.90 and opposite_justification >= 0.24:
                        alternatives = sorted(alternatives, key=lambda outcome: 0 if outcome == opposite else 1)
                    else:
                        alternatives = sorted(alternatives, key=lambda outcome: 1 if outcome == opposite else 0)

                pick_idx = (fallback_idx + rank * 2) % len(alternatives)
                alt_outcome = alternatives[pick_idx]
                if alt_outcome == modified[match_idx]:
                    if len(alternatives) > 1:
                        alt_outcome = alternatives[(pick_idx + 1) % len(alternatives)]
                    else:
                        continue

                modified[match_idx] = alt_outcome
                changed.append(match_idx)

            change_signature = tuple(sorted(changed))
            key = tuple(modified)
            fallback_distance_floor = max(1, min_coupon_distance - 1)
            is_far_enough = True
            distance_reference = list(base_coupons) + list(insured_coupons)
            if distance_reference:
                nearest = min(self._hamming_distance(modified, existing) for existing in distance_reference)
                is_far_enough = nearest >= fallback_distance_floor

            if (
                len(changed) >= min_changes_per_coupon
                and key not in seen
                and is_far_enough
                and seen_change_signatures.get(change_signature, 0) < max_signature_reuse
            ):
                seen.add(key)
                insured_coupons.append(modified)
                changed_matches_by_line.append(changed)
                seen_change_signatures[change_signature] = seen_change_signatures.get(change_signature, 0) + 1

            fallback_idx += 1
            if fallback_idx > max(target_count * 12, 64):
                break

        return insured_coupons, changed_matches_by_line

    def _opposite_outcome(self, outcome: str) -> str:
        if outcome == "1":
            return "2"
        if outcome == "2":
            return "1"
        return ""

    def _insurance_reason_scores(
        self,
        profile: dict[str, Any],
        strategy: dict[str, Any],
        base_single: str,
    ) -> dict[str, float]:
        probs = profile.get("probs", {"P1": 0.0, "PX": 0.0, "P2": 0.0})
        draw_prob = self._safe_float(probs.get("PX")) if isinstance(probs, dict) else 0.0
        uncertainty = self._safe_float(profile.get("uncertainty"))

        reason_scores = {
            "draw_risk": min(
                1.0,
                max(
                    draw_prob,
                    self._safe_float(strategy.get("draw_tendency_score")),
                    self._safe_float(strategy.get("historical_draw_bias_score")),
                ),
            ),
            "favorite_fail_risk": min(
                1.0,
                max(
                    self._safe_float(strategy.get("favorite_trap_score")),
                    self._safe_float(strategy.get("historical_favorite_fail_score")),
                ),
            ),
            "upset_risk": min(
                1.0,
                max(
                    self._safe_float(strategy.get("upset_potential")),
                    self._safe_float(strategy.get("historical_upset_score")),
                    self._safe_float(strategy.get("underdog_conversion_score")),
                ),
            ),
            "public_trap_risk": min(
                1.0,
                max(
                    self._safe_float(strategy.get("pool_vs_model_bias")),
                    self._safe_float(strategy.get("pool_vs_bk_divergence")),
                ),
            ),
            "history_conflict": min(
                1.0,
                max(
                    self._safe_float(strategy.get("history_conflict_with_model")),
                    self._safe_float(strategy.get("event_history_signal_strength")),
                ),
            ),
            "public_conflict": min(1.0, self._safe_float(strategy.get("public_bias"))),
            "uncertainty_conflict": min(1.0, uncertainty),
        }

        # If base is draw, directional conflicts should dominate draw_risk.
        if base_single == "X":
            reason_scores["draw_risk"] *= 0.6
            reason_scores["upset_risk"] = min(1.0, reason_scores["upset_risk"] + 0.05)

        return reason_scores

    def _insurance_risk_score(self, profile: dict[str, Any], strategy: dict[str, Any]) -> float:
        uncertainty = float(profile.get("uncertainty", 0.0))
        margin_top_second = float(profile.get("margin_top_second", 0.0))
        closeness = max(0.0, 1.0 - min(1.0, margin_top_second / 0.25))
        pool_vs_model = float(strategy.get("pool_vs_model_bias", 0.0))
        event_history_signal_strength = float(strategy.get("event_history_signal_strength", 0.0))
        historical_favorite_fail_score = float(strategy.get("historical_favorite_fail_score", 0.0))
        historical_upset_score = float(strategy.get("historical_upset_score", 0.0))
        strategy_risk = (
            0.25 * float(strategy.get("favorite_overload", 0.0))
            + 0.25 * float(strategy.get("upset_tendency", 0.0))
            + 0.12 * float(strategy.get("underdog_neglect", 0.0))
            + 0.12 * float(strategy.get("pool_vs_bk_gap", 0.0))
            + 0.10 * pool_vs_model
            + 0.10 * event_history_signal_strength
            + 0.06 * historical_favorite_fail_score
        ) if str(strategy.get("history_mode", "")) == "event_history_ready" else (
            0.30 * float(strategy.get("favorite_overload", 0.0))
            + 0.30 * float(strategy.get("upset_tendency", 0.0))
            + 0.15 * float(strategy.get("underdog_neglect", 0.0))
            + 0.15 * float(strategy.get("pool_vs_bk_gap", 0.0))
            + 0.10 * pool_vs_model
        )
        return (0.50 * uncertainty) + (0.25 * closeness) + (0.25 * min(1.0, strategy_risk))

    def _insurance_target_score(self, profile: dict[str, Any], strategy: dict[str, Any]) -> float:
        """Score match for inclusion in insured target set.

        This score intentionally overweights model-vs-market/history conflicts so
        insurance lines prioritize semantically risky matches, not only entropy.
        """
        base_risk = self._insurance_risk_score(profile=profile, strategy=strategy)
        history_mode = str(strategy.get("history_mode", "no_history"))

        conflict_signal = min(
            1.0,
            max(
                float(strategy.get("history_conflict_with_model", 0.0)),
                float(strategy.get("pool_vs_model_bias", 0.0)),
                float(strategy.get("pool_vs_bk_divergence", 0.0)),
                float(strategy.get("pool_vs_bk_gap", 0.0)),
            ),
        )
        upset_signal = min(
            1.0,
            max(
                float(strategy.get("favorite_trap_score", 0.0)),
                float(strategy.get("upset_potential", 0.0)),
                float(strategy.get("draw_tendency_score", 0.0)),
                float(strategy.get("historical_favorite_fail_score", 0.0)),
                float(strategy.get("historical_upset_score", 0.0)),
                float(strategy.get("historical_draw_bias_score", 0.0)),
            ),
        )
        event_signal = float(strategy.get("event_history_signal_strength", 0.0))

        mode_boost = 0.0
        if history_mode == "event_history_ready":
            mode_boost = 0.06 + 0.16 * min(1.0, event_signal)
        elif history_mode in ("draw_history_with_pool_signals", "draw_history_only"):
            mode_boost = 0.035

        event_component = 0.10 * min(1.0, event_signal) if history_mode == "event_history_ready" else 0.0

        return (
            0.50 * base_risk
            + 0.28 * conflict_signal
            + 0.17 * upset_signal
            + event_component
            + mode_boost
        )

    # ---------------------------------------------------------------------
    # Summary / diagnostics
    # ---------------------------------------------------------------------

    def _collect_signal_truth(
        self,
        base_contexts: list[dict[str, Any]],
        match_count: int,
        generation_mode: str,
        history_changed_insurance_matches_count: int = 0,
    ) -> dict[str, Any]:
        global_ctx = getattr(self, "_run_global_history", None)
        if not isinstance(global_ctx, dict):
            global_ctx = self.global_history_context if isinstance(self.global_history_context, dict) else None
        global_history_context_loaded = isinstance(global_ctx, dict)
        global_history_draws_loaded_count = 0
        global_history_events_loaded_count = 0
        if global_history_context_loaded:
            global_history_draws_loaded_count = int(global_ctx.get("history_draws_loaded_count", global_ctx.get("draws_loaded_count", 0)))
            global_history_events_loaded_count = int(global_ctx.get("history_events_loaded_count", 0))

        history_available_count = sum(1 for ctx in base_contexts if bool(ctx.get("history_available")))
        history_changed_decisions_count = sum(1 for ctx in base_contexts if bool(ctx.get("history_used")))
        pool_signals_used_count = sum(1 for ctx in base_contexts if bool(ctx.get("pool_signals_used")))
        bk_signals_used_count = sum(1 for ctx in base_contexts if bool(ctx.get("bk_signals_used")))
        event_history_used_count = sum(1 for ctx in base_contexts if bool(ctx.get("event_history_used")))
        pool_used_as_direct_signal_count = pool_signals_used_count
        pool_used_as_context_count = 0
        history_used_as_context_count = 0
        history_used_as_event_signal_count = 0
        strategy_adjusted_changed_matches: list[str] = []

        for idx, ctx in enumerate(base_contexts, start=1):
            model_decision = str(ctx.get("model_decision", ctx.get("decision", "")))
            adjusted_decision = str(ctx.get("decision", model_decision))
            if model_decision != adjusted_decision:
                strategy_adjusted_changed_matches.append(str(ctx.get("match_id") or f"match_{idx}"))

            pool_public_used = bool(ctx.get("pool_public_used", False))
            pool_used_for_strategy = bool(ctx.get("pool_signals_used", False))
            if pool_public_used and not pool_used_for_strategy:
                pool_used_as_context_count += 1

            history_context_used = bool(ctx.get("history_context_used", False))
            history_used_for_strategy = bool(ctx.get("history_used", False))
            if history_context_used and not history_used_for_strategy:
                history_used_as_context_count += 1

            if bool(ctx.get("event_history_used", False)):
                history_used_as_event_signal_count += 1

        strategy_adjusted_matches_count = len(strategy_adjusted_changed_matches)
        event_history_influenced_count = sum(
            1
            for ctx in base_contexts
            if str(ctx.get("history_mode", "")) == "event_history_ready"
            and (
                bool(ctx.get("history_used"))
                or self._safe_float(ctx.get("event_history_signal_strength")) >= 0.05
                or self._safe_float(ctx.get("history_adjustment")) >= 0.05
            )
        )

        history_draws_loaded_count = 0
        history_events_loaded_count = 0
        global_ctx_active = False
        all_strategy_signals: set[str] = set()
        history_state_counts: dict[str, int] = {}
        history_mode_counts: dict[str, int] = {}
        global_history_injected_count = 0
        history_context_used_count = 0

        _global_draws_max = 0
        _global_events_max = 0
        for ctx in base_contexts:
            state_key = str(ctx.get("history_state", "history_not_requested"))
            history_state_counts[state_key] = history_state_counts.get(state_key, 0) + 1
            mode_key = str(ctx.get("history_mode", "no_history"))
            history_mode_counts[mode_key] = history_mode_counts.get(mode_key, 0) + 1
            if bool(ctx.get("global_history_injected", False)):
                global_history_injected_count += 1
            if bool(ctx.get("history_context_used", False)):
                history_context_used_count += 1
            if "global" in state_key:
                global_ctx_active = True
                _global_draws_max = max(_global_draws_max, int(ctx.get("history_draws_loaded_count", 0)))
                _global_events_max = max(_global_events_max, int(ctx.get("history_events_loaded_count", 0)))
            else:
                history_draws_loaded_count += int(ctx.get("history_draws_loaded_count", 0))
                history_events_loaded_count += int(ctx.get("history_events_loaded_count", 0))
            for signal_name in ctx.get("strategy_signals_used", []):
                all_strategy_signals.add(str(signal_name))

        history_draws_loaded_count += _global_draws_max
        history_events_loaded_count += _global_events_max

        history_strategy_available = bool(
            history_available_count > 0
            or pool_signals_used_count > 0
            or bk_signals_used_count > 0
            or history_draws_loaded_count > 0
            or history_events_loaded_count > 0
        )
        history_strategy_used = bool(
            history_changed_decisions_count > 0
            or history_changed_insurance_matches_count > 0
        )

        if history_events_loaded_count > 0 and global_ctx_active:
            history_state_label = "global_history_baltbet"
            history_availability_state = "events_ready"
        elif history_events_loaded_count > 0:
            history_state_label = "full_history_ready"
            history_availability_state = "events_ready"
        elif history_draws_loaded_count > 0:
            history_state_label = "draws_loaded_no_event_propagation"
            history_availability_state = "draw_context_only"
        elif pool_signals_used_count > 0:
            history_state_label = "pool_probs_used_as_public_signal"
            history_availability_state = "signals_only"
        elif bk_signals_used_count > 0:
            history_state_label = "bookmaker_probs_used_as_market_signal"
            history_availability_state = "signals_only"
        elif history_strategy_available:
            history_state_label = "history_available_not_used"
            history_availability_state = "signals_only"
        else:
            history_state_label = "history_not_in_match_payload"
            history_availability_state = "none"

        if generation_mode == "insurance" and (
            history_events_loaded_count > 0
            or history_draws_loaded_count > 0
            or pool_signals_used_count > 0
            or bk_signals_used_count > 0
        ):
            probability_source = "model_plus_history_and_insurance"
            probability_source_human = "Модель + история/публичные сигналы + страховка"
        elif history_events_loaded_count > 0 and event_history_used_count > 0:
            probability_source = "model_plus_event_history"
            probability_source_human = "Модель + история событий Baltbet"
        elif history_events_loaded_count > 0:
            probability_source = "model_plus_event_history"
            probability_source_human = "Модель + история событий Baltbet (сигналы загружены)"
        elif history_draws_loaded_count > 0:
            probability_source = "model_plus_draw_history_context"
            probability_source_human = "Модель + контекст тиражей Baltbet (без событий)"
        elif pool_signals_used_count > 0:
            probability_source = "model_plus_pool_public"
            probability_source_human = "Модель + public/pool сигналы"
        elif bk_signals_used_count > 0:
            probability_source = "model_plus_bookmaker"
            probability_source_human = "Модель + букмекерские сигналы"
        else:
            probability_source = "model_only"
            probability_source_human = "Только модель"

        source_components = ["model"]
        if pool_signals_used_count > 0:
            source_components.append("pool_public")
        if bk_signals_used_count > 0:
            source_components.append("bookmaker")
        if history_draws_loaded_count > 0:
            source_components.append("draw_history_context")
        if history_events_loaded_count > 0:
            source_components.append("event_history")
        if event_history_used_count > 0:
            source_components.append("event_history_active")
        if generation_mode == "insurance":
            source_components.append("insurance")

        return {
            "model_used_count": match_count,
            "pool_used_count": pool_signals_used_count,
            "bookmaker_used_count": bk_signals_used_count,
            "strategy_adjusted_matches_count": strategy_adjusted_matches_count,
            "strategy_adjusted_changed_matches": strategy_adjusted_changed_matches,
            "pool_used_as_context_count": pool_used_as_context_count,
            "pool_used_as_direct_signal_count": pool_used_as_direct_signal_count,
            "history_used_as_context_count": history_used_as_context_count,
            "history_used_as_event_signal_count": history_used_as_event_signal_count,
            "event_history_used_count": event_history_used_count,
            "event_history_influenced_matches_count": event_history_influenced_count,
            "history_draws_loaded_count": history_draws_loaded_count,
            "history_events_loaded_count": history_events_loaded_count,
            "history_available_count": history_available_count,
            "history_changed_decisions_count": history_changed_decisions_count,
            "history_changed_insurance_matches_count": history_changed_insurance_matches_count,
            "history_strategy_available": history_strategy_available,
            "history_strategy_used": history_strategy_used,
            "history_state_label": history_state_label,
            "history_availability_state": history_availability_state,
            "probability_source": probability_source,
            "probability_source_human": probability_source_human,
            "source_components": source_components,
            "strategy_signals_used": sorted(all_strategy_signals),
            "history_state_counts": history_state_counts,
            "history_mode_counts": history_mode_counts,
            "global_history_injected_count": global_history_injected_count,
            "history_context_used_count": history_context_used_count,
            "global_history_context_loaded": global_history_context_loaded,
            "global_history_draws_loaded_count": global_history_draws_loaded_count,
            "global_history_events_loaded_count": global_history_events_loaded_count,
        }

    def _build_run_summary(
        self,
        matches: list[dict],
        mode: str,
        insurance_strength: float,
        base_decisions: list[str],
        base_contexts: list[dict[str, Any]],
        insured_pools: list[list[str]],
        coupons: list[list[str]],
        generation_mode: str,
    ) -> dict[str, Any]:
        n = len(matches)
        insured_decisions = [self._decision_from_outcomes(pool) for pool in insured_pools]
        model_decisions = [str(ctx.get("model_decision", ctx.get("decision", "1"))) for ctx in base_contexts]

        model_dist = self._model_distribution(matches)
        model_decision_dist = self._decision_distribution(model_decisions)
        base_dist = self._decision_distribution(base_decisions)

        decision_class_distribution: dict[str, int] = {}
        for ctx in base_contexts:
            class_key = str(ctx.get("decision_class", "unknown"))
            decision_class_distribution[class_key] = decision_class_distribution.get(class_key, 0) + 1

        match_type_distribution: dict[str, int] = {}
        risk_level_distribution: dict[str, int] = {}
        resolution_type_distribution: dict[str, int] = {}
        score_sums = {
            "draw_risk_score": 0.0,
            "upset_risk_score": 0.0,
            "weak_favorite_score": 0.0,
            "trap_match_score": 0.0,
            "single_confidence_score": 0.0,
            "insurance_priority_score": 0.0,
        }
        for ctx in base_contexts:
            mtype = str(ctx.get("match_type", "High-uncertainty match"))
            rlevel = str(ctx.get("risk_level", "medium-risk"))
            rtype = str(ctx.get("resolution_type", "double"))
            match_type_distribution[mtype] = match_type_distribution.get(mtype, 0) + 1
            risk_level_distribution[rlevel] = risk_level_distribution.get(rlevel, 0) + 1
            resolution_type_distribution[rtype] = resolution_type_distribution.get(rtype, 0) + 1
            for key in score_sums:
                score_sums[key] += self._safe_float(ctx.get(key, 0.0))

        score_avgs = {
            f"avg_{key}": (value / n) if n > 0 else 0.0
            for key, value in score_sums.items()
        }

        history_used_count = sum(1 for ctx in base_contexts if bool(ctx.get("history_used")))

        history_changed_matches: list[str] = []
        strategy_signal_sources: dict[str, int] = {}
        for match, ctx in zip(matches, base_contexts):
            if ctx.get("history_used"):
                name = str(match.get("home", match.get("match_id", "?"))) + " vs " + str(match.get("away", "?"))
                history_changed_matches.append(name)
            source = str(ctx.get("signal_source", ctx.get("strategy_reason", "unknown")))
            strategy_signal_sources[source] = strategy_signal_sources.get(source, 0) + 1

        strategy_adjusted_dist = dict(base_dist)
        insured_dist = self._decision_distribution(insured_decisions)

        history_influenced_matches_count = sum(
            1 for model_dec, final_dec in zip(model_decisions, base_decisions) if model_dec != final_dec
        )
        history_adjustment_values = [self._safe_float(ctx.get("history_adjustment")) for ctx in base_contexts]
        market_adjustment_values = [self._safe_float(ctx.get("market_adjustment")) for ctx in base_contexts]

        history_adjustment_summary = {
            "mean_history_adjustment": (sum(history_adjustment_values) / len(history_adjustment_values)) if history_adjustment_values else 0.0,
            "max_history_adjustment": max(history_adjustment_values) if history_adjustment_values else 0.0,
            "history_influenced_matches_count": history_influenced_matches_count,
        }
        market_adjustment_summary = {
            "mean_market_adjustment": (sum(market_adjustment_values) / len(market_adjustment_values)) if market_adjustment_values else 0.0,
            "max_market_adjustment": max(market_adjustment_values) if market_adjustment_values else 0.0,
        }

        p2_ge_020 = 0
        p2_competitive = 0
        insurance_added_1 = 0
        insurance_added_x = 0
        insurance_added_2 = 0
        for match, pool, base_dec in zip(matches, insured_pools, base_decisions):
            probs = self._normalised_probs(match["probs"])
            p2 = probs["P2"]
            if p2 >= 0.20:
                p2_ge_020 += 1
            if p2 >= max(probs["P1"], probs["PX"]) - 0.05:
                p2_competitive += 1

            # Insurance additions are meaningful only for the insurance generation path.
            if generation_mode != "insurance":
                continue

            base_single = self._single_from_decision(base_dec, probs)
            if "1" in pool and base_single != "1":
                insurance_added_1 += 1
            if "X" in pool and base_single != "X":
                insurance_added_x += 1
            if "2" in pool and base_single != "2":
                insurance_added_2 += 1

        base_with_2 = sum(1 for d in base_decisions if "2" in d)
        insured_with_2 = sum(1 for pool in insured_pools if "2" in pool)

        final_dist = self._final_coupon_distribution(coupons)
        uniqueness = self._coupon_uniqueness_stats(coupons)
        diversity = self._coupon_diversity_metrics(coupons)
        lines_with_1 = sum(1 for coupon in coupons if "1" in coupon)
        lines_with_x = sum(1 for coupon in coupons if "X" in coupon)
        lines_with_2 = sum(1 for coupon in coupons if "2" in coupon)

        pool_fallback_count = sum(
            1
            for m in matches
            if str(m.get("prob_source", "")).lower() in ("pool_context_only", "pool_only")
        )
        weak_model_features_suspected = (n > 0) and (pool_fallback_count > n // 2)

        signal_truth = self._collect_signal_truth(
            base_contexts=base_contexts,
            match_count=n,
            generation_mode=generation_mode,
            history_changed_insurance_matches_count=0,
        )

        if signal_truth["history_available_count"] == 0 and signal_truth["pool_used_count"] == 0:
            hs_reason = "no_history_signals_in_match_payload"
        elif history_used_count == 0 and signal_truth["pool_used_count"] == 0:
            hs_reason = "history_signals_present_but_no_decision_changed"
        elif signal_truth["pool_used_count"] > 0 and history_used_count > 0:
            hs_reason = "history_and_pool_strategy_applied"
        elif signal_truth["pool_used_count"] > 0:
            hs_reason = "pool_strategy_applied"
        else:
            hs_reason = "history_strategy_applied"

        insurance_enabled = bool(generation_mode == "insurance" and insurance_strength > 0.0)
        default_insurance_diagnostics = {
            "insurance_enabled": insurance_enabled,
            "insurance_strength": float(insurance_strength),
            "insured_coupons_count": 0,
            "affected_coupon_lines_count": 0,
            "changed_matches_by_insurance": [],
            "target_match_indexes": [],
            "target_match_labels": [],
            "target_reason_summary": {
                "model_vs_history_conflict_count": 0,
                "public_vs_model_conflict_count": 0,
                "high_risk_match_count": 0,
            },
            "change_distribution": {"1": 0, "X": 0, "2": 0},
            "coverage_summary": {
                "matches_with_expanded_coverage": 0,
                "matches_with_alternative_outcome": 0,
                "matches_without_changes": 0,
            },
        }

        final_coverages = [list(pool) for pool in insured_pools]
        match_layer_diagnostics = self._build_match_layer_diagnostics(
            matches=matches,
            base_contexts=base_contexts,
            base_decisions=base_decisions,
            final_coverages=final_coverages,
            coupons=coupons,
            base_coupon_count=len(coupons),
            insurance_meta_by_match=None,
        )
        coupon_entries = self._build_coupon_entries(
            coupons=coupons,
            base_coupon_count=len(coupons),
            changed_matches_by_line=None,
        )

        return {
            "layer_contract_version": "toto_layers_v1",
            "mode": mode,
            "generation_mode": generation_mode,
            "insurance_strength": insurance_strength,
            "probability_source": str(signal_truth["probability_source"]),
            "probability_source_human": str(signal_truth["probability_source_human"]),
            "source_components": list(signal_truth["source_components"]),
            "stake_affects_decision": False,
            "mode_affects_decision": False,
            "insurance_is_secondary_layer": True,
            "insurance_enabled": insurance_enabled,
            "coupon_count": len(coupons),
            "generated_coupon_count": int(uniqueness["generated_coupon_count"]),
            "unique_coupon_count": int(uniqueness["unique_coupon_count"]),
            "duplicate_coupon_count": int(uniqueness["duplicate_coupon_count"]),
            "base_coupons_count": int(uniqueness["generated_coupon_count"]),
            "insured_coupons_count": 0,
            "base_coupon_count": int(uniqueness["generated_coupon_count"]),
            "insurance_coupon_count": 0,
            "match_count": n,
            "model_distribution": model_dist,
            "model_decision_distribution": model_decision_dist,
            "base_decision_distribution": base_dist,
            "base_decision_class_distribution": decision_class_distribution,
            "match_type_distribution": match_type_distribution,
            "risk_level_distribution": risk_level_distribution,
            "resolution_type_distribution": resolution_type_distribution,
            **score_avgs,
            "strategy_adjusted_distribution": strategy_adjusted_dist,
            "strategy_adjusted_matches_count": int(signal_truth["strategy_adjusted_matches_count"]),
            "strategy_adjusted_changed_matches": list(signal_truth["strategy_adjusted_changed_matches"]),
            "model_used_count": int(signal_truth["model_used_count"]),
            "pool_used_count": int(signal_truth["pool_used_count"]),
            "bookmaker_used_count": int(signal_truth["bookmaker_used_count"]),
            "pool_used_as_context_count": int(signal_truth["pool_used_as_context_count"]),
            "pool_used_as_direct_signal_count": int(signal_truth["pool_used_as_direct_signal_count"]),
            "history_used_as_context_count": int(signal_truth["history_used_as_context_count"]),
            "history_used_as_event_signal_count": int(signal_truth["history_used_as_event_signal_count"]),
            "event_history_used_count": int(signal_truth["event_history_used_count"]),
            "event_history_influenced_matches_count": int(signal_truth["event_history_influenced_matches_count"]),
            "history_draws_loaded_count": int(signal_truth["history_draws_loaded_count"]),
            "history_events_loaded_count": int(signal_truth["history_events_loaded_count"]),
            "history_availability_state": str(signal_truth["history_availability_state"]),
            "history_state_label": str(signal_truth["history_state_label"]),
            "global_history_context_loaded": bool(signal_truth["global_history_context_loaded"]),
            "global_history_draws_loaded_count": int(signal_truth["global_history_draws_loaded_count"]),
            "global_history_events_loaded_count": int(signal_truth["global_history_events_loaded_count"]),
            "history_influenced_matches_count": history_influenced_matches_count,
            "history_adjustment_summary": history_adjustment_summary,
            "market_adjustment_summary": market_adjustment_summary,
            "insured_decision_distribution": insured_dist,
            "insurance_added_per_outcome": {
                "1": insurance_added_1,
                "X": insurance_added_x,
                "2": insurance_added_2,
            },
            "final_coupon_distribution": final_dist,
            "final_coupon_line_distribution": {
                "lines_with_1": lines_with_1,
                "lines_with_X": lines_with_x,
                "lines_with_2": lines_with_2,
            },
            "average_hamming_distance_between_coupons": float(diversity["average_hamming_distance_between_coupons"]),
            "median_hamming_distance": float(diversity["median_hamming_distance"]),
            "min_hamming_distance": float(diversity["min_hamming_distance"]),
            "coupon_diversity_score": float(diversity["coupon_diversity_score"]),
            "per_match_entropy": dict(diversity["per_match_entropy"]),
            "insurance_diagnostics": default_insurance_diagnostics,
            "coupon_entries": coupon_entries,
            "match_layer_diagnostics": match_layer_diagnostics,
            "history_strategy": {
                "history_strategy_available": bool(signal_truth["history_strategy_available"]),
                "history_strategy_used": bool(signal_truth["history_strategy_used"]),
                "history_state_label": str(signal_truth["history_state_label"]),
                "history_availability_state": str(signal_truth["history_availability_state"]),
                "history_mode_counts": dict(signal_truth["history_mode_counts"]),
                "history_context_used_count": int(signal_truth["history_context_used_count"]),
                "global_history_injected_count": int(signal_truth["global_history_injected_count"]),
                "pool_signals_used_count": int(signal_truth["pool_used_count"]),
                "bk_signals_used_count": int(signal_truth["bookmaker_used_count"]),
                "pool_used_as_context_count": int(signal_truth["pool_used_as_context_count"]),
                "pool_used_as_direct_signal_count": int(signal_truth["pool_used_as_direct_signal_count"]),
                "history_used_as_context_count": int(signal_truth["history_used_as_context_count"]),
                "history_used_as_event_signal_count": int(signal_truth["history_used_as_event_signal_count"]),
                "event_history_used_count": int(signal_truth["event_history_used_count"]),
                "event_history_influenced_matches_count": int(signal_truth["event_history_influenced_matches_count"]),
                "history_draws_loaded_count": int(signal_truth["history_draws_loaded_count"]),
                "history_events_loaded_count": int(signal_truth["history_events_loaded_count"]),
                "matches_with_history_signals": int(signal_truth["history_available_count"]),
                "matches_where_history_changed_decision": history_used_count,
                "history_changed_matches": history_changed_matches,
                "strategy_signal_sources": strategy_signal_sources,
                "all_strategy_signals": list(signal_truth["strategy_signals_used"]),
                "history_state_counts": dict(signal_truth["history_state_counts"]),
                "applied_in_manual_mode_when_signals_present": history_used_count > 0 or int(signal_truth["pool_used_count"]) > 0,
                "reason": hs_reason,
            },
            "match_level_decision_diagnostics": [
                {
                    "match_index": idx + 1,
                    "match": self._match_label(match),
                    "model_decision": model_decisions[idx],
                    "strategy_adjusted_decision": base_decisions[idx],
                    "history_used": bool(ctx.get("history_used", False)),
                    "history_context_used": bool(ctx.get("history_context_used", False)),
                    "history_mode": str(ctx.get("history_mode", "no_history")),
                    "event_history_used": bool(ctx.get("event_history_used", False)),
                    "global_history_context_loaded": bool(ctx.get("global_history_context_loaded", False)),
                    "global_history_injected": bool(ctx.get("global_history_injected", False)),
                    "pool_public_used": bool(ctx.get("pool_public_used", False)),
                    "bookmaker_used": bool(ctx.get("bookmaker_used", False)),
                    "pool_signals_used": bool(ctx.get("pool_signals_used", False)),
                    "bk_signals_used": bool(ctx.get("bk_signals_used", False)),
                    "history_ready": bool(ctx.get("history_ready", False)),
                    "history_state": str(ctx.get("history_state", "history_not_requested")),
                    "history_reason": str(ctx.get("history_reason", "")),
                    "history_adjustment": self._safe_float(ctx.get("history_adjustment")),
                    "market_adjustment": self._safe_float(ctx.get("market_adjustment")),
                    "public_bias_score": self._safe_float(ctx.get("public_bias_score")),
                    "event_history_signal_strength": self._safe_float(ctx.get("event_history_signal_strength")),
                    "historical_favorite_fail_score": self._safe_float(ctx.get("historical_favorite_fail_score")),
                    "historical_draw_bias_score": self._safe_float(ctx.get("historical_draw_bias_score")),
                    "historical_upset_score": self._safe_float(ctx.get("historical_upset_score")),
                    "match_type": str(ctx.get("match_type", "High-uncertainty match")),
                    "risk_level": str(ctx.get("risk_level", "medium-risk")),
                    "resolution_type": str(ctx.get("resolution_type", "double")),
                    "draw_risk_score": self._safe_float(ctx.get("draw_risk_score")),
                    "upset_risk_score": self._safe_float(ctx.get("upset_risk_score")),
                    "weak_favorite_score": self._safe_float(ctx.get("weak_favorite_score")),
                    "trap_match_score": self._safe_float(ctx.get("trap_match_score")),
                    "single_confidence_score": self._safe_float(ctx.get("single_confidence_score")),
                    "double_recommendation_flag": bool(ctx.get("double_recommendation_flag", False)),
                    "insurance_priority_score": self._safe_float(ctx.get("insurance_priority_score")),
                    "toto_layer_used": bool(ctx.get("toto_layer_used", False)),
                    "payload_toto_brief_history": bool(isinstance(ctx.get("payload_toto_brief_history"), dict) and bool(ctx.get("payload_toto_brief_history"))),
                    "payload_pool_probs": ctx.get("payload_pool_probs", {}),
                    "payload_pool_quotes": ctx.get("payload_pool_quotes", {}),
                    "payload_bookmaker_probs": ctx.get("payload_bookmaker_probs", {}),
                    "payload_bookmaker_quotes": ctx.get("payload_bookmaker_quotes", {}),
                    "reason": str(ctx.get("history_adjustment_reason", ctx.get("strategy_reason", ""))),
                }
                for idx, (match, ctx) in enumerate(zip(matches, base_contexts))
            ],
            "insurance_target_matches": [],
            "insurance_target_matches_count": 0,
            "insurance_changed_matches_count": 0,
            "insurance_changed_coupon_lines_count": 0,
            "weak_model_features_suspected": weak_model_features_suspected,
            "pool_fallback_matches": pool_fallback_count,
            "matches_p2_ge_0_20": p2_ge_020,
            "matches_p2_competitive": p2_competitive,
            "p2_base_represented_matches": base_with_2,
            "p2_insured_represented_matches": insured_with_2,
            "insurance_added_2_matches": insurance_added_2,
            "p2_final_coupon_cells": final_dist["2"],
        }

    def _build_run_summary_with_insurance(
        self,
        matches: list[dict],
        mode: str,
        insurance_strength: float,
        base_decisions: list[str],
        base_contexts: list[dict[str, Any]],
        insured_pools: list[list[str]],
        insured_matches_dict: dict[int, dict[str, Any]],
        strategy_risk_scores: dict[int, float],
        coupons: list[list[str]],
        base_coupons_count: int,
        insured_coupons_count: int,
        changed_matches_by_line: list[list[int]],
        generation_mode: str,
    ) -> dict[str, Any]:
        base_summary = self._build_run_summary(
            matches=matches,
            mode=mode,
            insurance_strength=insurance_strength,
            base_decisions=base_decisions,
            base_contexts=base_contexts,
            insured_pools=insured_pools,
            coupons=coupons,
            generation_mode=generation_mode,
        )

        insured_matches_count = len(insured_matches_dict)
        expanded_outcomes_count = sum(len(info.get("alternatives", [])) for info in insured_matches_dict.values())

        per_match_coupon_changes: dict[int, int] = {}
        affected_coupon_lines_set: set[int] = set()
        insurance_cells_changed_count = 0
        change_distribution = {"1": 0, "X": 0, "2": 0}

        for match_idx, insured_info in insured_matches_dict.items():
            base_outcome = insured_info.get("base_outcome")
            alternatives = insured_info.get("alternatives", [])
            coupons_with_alternative = 0

            for coupon_line_idx, coupon in enumerate(coupons):
                if match_idx >= len(coupon):
                    continue
                if coupon[match_idx] in alternatives:
                    coupons_with_alternative += 1
                    if coupon[match_idx] != base_outcome:
                        insurance_cells_changed_count += 1
                        changed_outcome = str(coupon[match_idx])
                        if changed_outcome in change_distribution:
                            change_distribution[changed_outcome] += 1
                    affected_coupon_lines_set.add(coupon_line_idx)

            per_match_coupon_changes[match_idx] = coupons_with_alternative

        changed_matches_set: set[int] = set()
        for changed_line in changed_matches_by_line:
            for idx in changed_line:
                changed_matches_set.add(idx)

        affected_coupon_lines_count = len(affected_coupon_lines_set)
        insurance_rebalance_changes = affected_coupon_lines_count
        insured_coupon_indices = list(range(base_coupons_count, base_coupons_count + insured_coupons_count))
        target_match_indexes = sorted(insured_matches_dict.keys())
        target_match_labels = [
            self._match_label(matches[idx])
            for idx in target_match_indexes
            if 0 <= idx < len(matches)
        ]

        model_vs_history_conflict_count = 0
        public_vs_model_conflict_count = 0
        high_risk_match_count = 0
        for idx in target_match_indexes:
            if idx >= len(base_contexts):
                continue
            ctx = base_contexts[idx]
            if self._safe_float(ctx.get("history_conflict_with_model")) >= 0.08:
                model_vs_history_conflict_count += 1
            if (
                self._safe_float(ctx.get("public_bias_score")) >= 0.06
                or bool(ctx.get("pool_signals_used", False))
            ):
                public_vs_model_conflict_count += 1
            if self._safe_float(strategy_risk_scores.get(idx, 0.0)) >= 0.60:
                high_risk_match_count += 1

        matches_with_expanded_coverage = sum(
            1
            for idx in target_match_indexes
            if len(insured_matches_dict.get(idx, {}).get("alternatives", [])) > 0
        )
        matches_with_alternative_outcome = sum(
            1
            for idx in target_match_indexes
            if int(per_match_coupon_changes.get(idx, 0)) > 0
        )
        matches_without_changes = max(len(target_match_indexes) - matches_with_alternative_outcome, 0)

        history_changed_insurance_matches_count = sum(1 for idx in changed_matches_set if idx in insured_matches_dict)
        signal_truth = self._collect_signal_truth(
            base_contexts=base_contexts,
            match_count=len(matches),
            generation_mode=generation_mode,
            history_changed_insurance_matches_count=history_changed_insurance_matches_count,
        )

        weak_model_features_present = False
        for m in matches:
            if bool(m.get("weak_model_features_present")):
                weak_model_features_present = True
                break

        insurance_diagnostics = {
            "insurance_enabled": True,
            "insurance_strength": float(insurance_strength),
            "insured_matches_count": insured_matches_count,
            "expanded_outcomes_count": expanded_outcomes_count,
            "insurance_rebalance_changes": insurance_rebalance_changes,
            "affected_coupon_lines_count": affected_coupon_lines_count,
            "insurance_cells_changed_count": insurance_cells_changed_count,
            "per_match_coupon_changes": per_match_coupon_changes,
            "base_coupons_count": base_coupons_count,
            "insured_coupons_count": insured_coupons_count,
            "insured_coupon_indices": insured_coupon_indices,
            "changed_matches_by_insurance": sorted(changed_matches_set),
            "target_match_indexes": target_match_indexes,
            "target_match_labels": target_match_labels,
            "target_reason_summary": {
                "model_vs_history_conflict_count": model_vs_history_conflict_count,
                "public_vs_model_conflict_count": public_vs_model_conflict_count,
                "high_risk_match_count": high_risk_match_count,
            },
            "change_distribution": change_distribution,
            "coverage_summary": {
                "matches_with_expanded_coverage": matches_with_expanded_coverage,
                "matches_with_alternative_outcome": matches_with_alternative_outcome,
                "matches_without_changes": matches_without_changes,
            },
            "insurance_effectiveness_ratio": (
                affected_coupon_lines_count / len(coupons)
                if len(coupons) > 0
                else 0.0
            ),
            "adjacent_alternatives_added_count": 0,
            "strong_opposite_alternatives_added_count": 0,
            "strong_opposite_with_strong_justification_count": 0,
        }

        final_coverages = [list(pool) for pool in insured_pools]
        match_layer_diagnostics = self._build_match_layer_diagnostics(
            matches=matches,
            base_contexts=base_contexts,
            base_decisions=base_decisions,
            final_coverages=final_coverages,
            coupons=coupons,
            base_coupon_count=base_coupons_count,
            insurance_meta_by_match=insured_matches_dict,
        )
        coupon_entries = self._build_coupon_entries(
            coupons=coupons,
            base_coupon_count=base_coupons_count,
            changed_matches_by_line=changed_matches_by_line,
        )
        insurance_coupon_count = max(len(coupons) - int(base_coupons_count), 0)

        adjacent_alternatives_added_count = 0
        strong_opposite_alternatives_added_count = 0
        strong_opposite_with_strong_justification_count = 0
        for row in match_layer_diagnostics:
            if not isinstance(row, dict):
                continue
            ins_codes = row.get("insurance_reason_codes", [])
            if not isinstance(ins_codes, list):
                continue
            if "adjacent_safety" in ins_codes:
                adjacent_alternatives_added_count += 1
            if "opposite_safety" in ins_codes:
                strong_opposite_alternatives_added_count += 1
                if str(row.get("insurance_justification_strength", "weak")) == "strong":
                    strong_opposite_with_strong_justification_count += 1

        insurance_diagnostics["adjacent_alternatives_added_count"] = adjacent_alternatives_added_count
        insurance_diagnostics["strong_opposite_alternatives_added_count"] = strong_opposite_alternatives_added_count
        insurance_diagnostics["strong_opposite_with_strong_justification_count"] = (
            strong_opposite_with_strong_justification_count
        )

        base_summary["insurance_diagnostics"] = insurance_diagnostics
        base_summary["base_coupons_count"] = base_coupons_count
        base_summary["insured_coupons_count"] = insured_coupons_count
        base_summary["base_coupon_count"] = base_coupons_count
        base_summary["insurance_coupon_count"] = insurance_coupon_count
        uniqueness = self._coupon_uniqueness_stats(coupons)
        base_summary["generated_coupon_count"] = int(uniqueness["generated_coupon_count"])
        base_summary["unique_coupon_count"] = int(uniqueness["unique_coupon_count"])
        base_summary["duplicate_coupon_count"] = int(uniqueness["duplicate_coupon_count"])
        base_summary["insured_coupon_indices"] = insured_coupon_indices
        base_summary["changed_matches_by_insurance"] = sorted(changed_matches_set)
        base_summary["insurance_target_matches"] = [
            {
                "match_index": idx,
                "risk_score": float(strategy_risk_scores.get(idx, 0.0)),
            }
            for idx in sorted(insured_matches_dict.keys())
        ]
        base_summary["insurance_target_matches_count"] = len(insured_matches_dict)
        base_summary["insurance_changed_matches_count"] = len(changed_matches_set)
        base_summary["insurance_cells_changed_count"] = insurance_cells_changed_count
        base_summary["affected_coupon_lines_count"] = affected_coupon_lines_count
        base_summary["insurance_changed_coupon_lines_count"] = affected_coupon_lines_count
        base_summary["adjacent_alternatives_added_count"] = adjacent_alternatives_added_count
        base_summary["strong_opposite_alternatives_added_count"] = strong_opposite_alternatives_added_count
        base_summary["strong_opposite_with_strong_justification_count"] = (
            strong_opposite_with_strong_justification_count
        )
        base_summary["insurance_enabled"] = True
        base_summary["target_match_indexes"] = target_match_indexes
        base_summary["target_match_labels"] = target_match_labels
        base_summary["target_reason_summary"] = insurance_diagnostics["target_reason_summary"]
        base_summary["change_distribution"] = change_distribution
        base_summary["coverage_summary"] = insurance_diagnostics["coverage_summary"]
        base_summary["coupon_entries"] = coupon_entries
        base_summary["match_layer_diagnostics"] = match_layer_diagnostics
        base_summary["probability_source"] = str(signal_truth["probability_source"])
        base_summary["probability_source_human"] = str(signal_truth["probability_source_human"])
        base_summary["source_components"] = list(signal_truth["source_components"])
        base_summary["model_used_count"] = int(signal_truth["model_used_count"])
        base_summary["pool_used_count"] = int(signal_truth["pool_used_count"])
        base_summary["bookmaker_used_count"] = int(signal_truth["bookmaker_used_count"])
        base_summary["event_history_used_count"] = int(signal_truth["event_history_used_count"])
        base_summary["event_history_influenced_matches_count"] = int(signal_truth["event_history_influenced_matches_count"])
        base_summary["history_draws_loaded_count"] = int(signal_truth["history_draws_loaded_count"])
        base_summary["history_events_loaded_count"] = int(signal_truth["history_events_loaded_count"])
        base_summary["history_availability_state"] = str(signal_truth["history_availability_state"])
        base_summary["history_state_label"] = str(signal_truth["history_state_label"])
        base_summary["history_mode_counts"] = dict(signal_truth["history_mode_counts"])
        base_summary["history_context_used_count"] = int(signal_truth["history_context_used_count"])
        base_summary["global_history_injected_count"] = int(signal_truth["global_history_injected_count"])
        base_summary["global_history_context_loaded"] = bool(signal_truth["global_history_context_loaded"])
        base_summary["global_history_draws_loaded_count"] = int(signal_truth["global_history_draws_loaded_count"])
        base_summary["global_history_events_loaded_count"] = int(signal_truth["global_history_events_loaded_count"])

        # Determine if history was truly available and used.
        base_summary["history_strategy"] = {
            "history_strategy_available": bool(signal_truth["history_strategy_available"]),
            "history_strategy_used": bool(signal_truth["history_strategy_used"]),
            "history_state_label": str(signal_truth["history_state_label"]),
            "history_availability_state": str(signal_truth["history_availability_state"]),
            "history_mode_counts": dict(signal_truth["history_mode_counts"]),
            "history_context_used_count": int(signal_truth["history_context_used_count"]),
            "global_history_injected_count": int(signal_truth["global_history_injected_count"]),
            "history_draws_loaded_count": int(signal_truth["history_draws_loaded_count"]),
            "history_events_loaded_count": int(signal_truth["history_events_loaded_count"]),
            "pool_signals_used_count": int(signal_truth["pool_used_count"]),
            "bk_signals_used_count": int(signal_truth["bookmaker_used_count"]),
            "event_history_used_count": int(signal_truth["event_history_used_count"]),
            "event_history_influenced_matches_count": int(signal_truth["event_history_influenced_matches_count"]),
            "history_changed_decisions_count": int(signal_truth["history_changed_decisions_count"]),
            "history_changed_matches_count": int(signal_truth["history_changed_decisions_count"]),
            "history_changed_insurance_matches_count": history_changed_insurance_matches_count,
            "strategy_signals_used": list(signal_truth["strategy_signals_used"]),
            "history_state_counts": dict(signal_truth["history_state_counts"]),
        }

        base_summary["weak_model_features_present"] = weak_model_features_present
        return base_summary

    # ---------------------------------------------------------------------
    # History helpers
    # ---------------------------------------------------------------------

    def _compute_event_history_signals(self, events: list[dict]) -> dict:
        """Aggregate tactical signals directly from TotoBrief historical event rows.

        Each event row must contain: result, pool_win_1, pool_draw, pool_win_2,
        bk_win_1, bk_draw, bk_win_2 (percentages 0-100 OR fractions 0-1).

        Returns a dict with:
          pool_avg_probs, bk_avg_probs  — normalised dicts {P1, PX, P2}
          historical_favorite_fail_score — fraction where pool-favorite lost
          historical_draw_bias_score     — fraction that were draws
          historical_upset_score         — weighted favorite-fail (pool+bk)
          pool_vs_bk_historical_divergence — mean |pool_i - bk_i| across events
          underdog_conversion_score      — fraction where lowest-bk-prob outcome won
          draw_rate, home_rate, away_rate — simple result fractions
          events_used                    — number of valid events processed
        """
        pool_sums = {"1": 0.0, "X": 0.0, "2": 0.0}
        bk_sums = {"1": 0.0, "X": 0.0, "2": 0.0}
        pool_count = 0
        bk_count = 0
        result_counts: dict[str, int] = {"1": 0, "X": 0, "2": 0}
        pool_fav_fail = 0
        bk_fav_fail = 0
        draw_count = 0
        underdog_conversions = 0
        pool_bk_divergences: list[float] = []

        for ev in events:
            result = str(ev.get("result", "")).strip().upper()
            if result in ("DRAW", "D"):
                result = "X"
            if result not in ("1", "X", "2"):
                continue

            result_counts[result] = result_counts.get(result, 0) + 1

            pool_raw = {
                "1": self._safe_float(ev.get("pool_win_1")),
                "X": self._safe_float(ev.get("pool_draw")),
                "2": self._safe_float(ev.get("pool_win_2")),
            }
            bk_raw = {
                "1": self._safe_float(ev.get("bk_win_1")),
                "X": self._safe_float(ev.get("bk_draw")),
                "2": self._safe_float(ev.get("bk_win_2")),
            }

            pool_total = sum(pool_raw.values())
            bk_total = sum(bk_raw.values())

            pool_norm: dict[str, float] | None = None
            bk_norm: dict[str, float] | None = None

            if pool_total > 0:
                pool_norm = {k: v / pool_total for k, v in pool_raw.items()}
                for k in ("1", "X", "2"):
                    pool_sums[k] += pool_norm[k]
                pool_count += 1
                pool_fav = max(pool_norm, key=pool_norm.get)  # type: ignore[misc]
                if pool_fav != result:
                    pool_fav_fail += 1

            if bk_total > 0:
                bk_norm = {k: v / bk_total for k, v in bk_raw.items()}
                for k in ("1", "X", "2"):
                    bk_sums[k] += bk_norm[k]
                bk_count += 1
                bk_fav = max(bk_norm, key=bk_norm.get)  # type: ignore[misc]
                if bk_fav != result:
                    bk_fav_fail += 1
                # Underdog = outcome with lowest BK probability
                bk_sorted = sorted(bk_norm, key=bk_norm.get)  # type: ignore[misc]
                if result == bk_sorted[0]:
                    underdog_conversions += 1

            if pool_norm is not None and bk_norm is not None:
                div = sum(abs(pool_norm[k] - bk_norm[k]) for k in ("1", "X", "2")) / 3.0
                pool_bk_divergences.append(div)

            if result == "X":
                draw_count += 1

        valid_n = sum(result_counts.values()) or 1

        pool_avg_probs: dict[str, float] | None = None
        if pool_count > 0:
            p1 = pool_sums["1"] / pool_count
            px = pool_sums["X"] / pool_count
            p2 = pool_sums["2"] / pool_count
            total = p1 + px + p2
            if total > 0:
                pool_avg_probs = {"P1": p1 / total, "PX": px / total, "P2": p2 / total}

        bk_avg_probs: dict[str, float] | None = None
        if bk_count > 0:
            p1 = bk_sums["1"] / bk_count
            px = bk_sums["X"] / bk_count
            p2 = bk_sums["2"] / bk_count
            total = p1 + px + p2
            if total > 0:
                bk_avg_probs = {"P1": p1 / total, "PX": px / total, "P2": p2 / total}

        return {
            "pool_avg_probs": pool_avg_probs,
            "bk_avg_probs": bk_avg_probs,
            "historical_favorite_fail_score": pool_fav_fail / valid_n,
            "historical_draw_bias_score": draw_count / valid_n,
            "historical_upset_score": (pool_fav_fail + bk_fav_fail * 0.5) / valid_n,
            "pool_vs_bk_historical_divergence": (
                sum(pool_bk_divergences) / len(pool_bk_divergences)
                if pool_bk_divergences
                else 0.0
            ),
            "underdog_conversion_score": underdog_conversions / valid_n,
            "draw_rate": draw_count / valid_n,
            "home_rate": result_counts["1"] / valid_n,
            "away_rate": result_counts["2"] / valid_n,
            "events_used": valid_n,
        }

    def _extract_history_stats(self, match: dict) -> dict[str, Any]:
        draws_count = 0
        events: list[dict] = []

        draws_payload = match.get("toto_brief_draws")
        if isinstance(draws_payload, list):
            draws_count += len(draws_payload)

        history_payload = match.get("toto_brief_history")
        if isinstance(history_payload, dict):
            drawings = history_payload.get("drawings")
            if isinstance(drawings, list):
                draws_count += len(drawings)
            # draws_loaded_count also directly available
            dl = history_payload.get("history_draws_loaded_count", 0)
            if dl:
                draws_count = max(draws_count, int(dl))
            nested_events = history_payload.get("events")
            if isinstance(nested_events, list):
                for ev in nested_events:
                    if isinstance(ev, dict):
                        events.append(ev)

        # Manual mode bridge: selected draw payload can carry drawing + event history.
        selected_draw = match.get("selected_draw")
        if isinstance(selected_draw, dict):
            draws_count += 1
            selected_events = selected_draw.get("events")
            if isinstance(selected_events, list):
                for ev in selected_events:
                    if isinstance(ev, dict):
                        events.append(ev)
            drawing_info = selected_draw.get("drawing_info")
            if isinstance(drawing_info, dict):
                di_events = drawing_info.get("events")
                if isinstance(di_events, list):
                    for ev in di_events:
                        if isinstance(ev, dict):
                            events.append(ev)

        for key in ("toto_brief_events", "history_events", "event_history"):
            payload = match.get(key)
            if isinstance(payload, list):
                for ev in payload:
                    if isinstance(ev, dict):
                        events.append(ev)

        # Deduplicate events by stable tuple key when available.
        dedup: dict[tuple, dict] = {}
        for ev in events:
            key = (
                ev.get("drawing_id"),
                ev.get("order"),
                ev.get("name"),
                ev.get("result"),
            )
            dedup[key] = ev
        events = list(dedup.values())

        events_count = len(events)

        # Resolve history state.
        history_state = "history_not_requested"
        history_empty_reason: str | None = "history_not_requested"
        history_stats_ready = False
        if isinstance(history_payload, dict):
            raw_state = history_payload.get("history_state")
            history_state = str(raw_state) if raw_state else "history_requested_but_empty"
            history_empty_reason_raw = history_payload.get("history_empty_reason")
            history_empty_reason = str(history_empty_reason_raw) if history_empty_reason_raw is not None else None
            history_stats_ready = bool(history_payload.get("history_stats_ready", False))

        if events_count == 0:
            # ------------------------------------------------------------------
            # FALLBACK: When per-match events weren't propagated (common in draw
            # mode where toto_api attaches toto_brief_history without the full
            # events list per match), try to use pre-computed stats from
            # toto_brief_history.stats dict.
            # ------------------------------------------------------------------
            if isinstance(history_payload, dict):
                stats_dict = history_payload.get("stats")
                if isinstance(stats_dict, dict):
                    precomputed_events = self._safe_float(stats_dict.get("events_count", 0.0))
                    if precomputed_events > 0:
                        # Pool favorite win rate = 1 - upset_rate - 0.5 * draw_results_rate
                        upset_rate = self._safe_float(stats_dict.get("upset_rate", 0.0))
                        draw_results_rate = self._safe_float(stats_dict.get("draw_results_rate", 0.0))
                        pool_fav_win_rate = max(0.0, 1.0 - upset_rate - draw_results_rate * 0.5)
                        pool_vs_bk_gap = self._safe_float(stats_dict.get("pool_vs_bookmaker_gap", 0.0))
                        return {
                            "draws_count": draws_count,
                            "events_count": int(precomputed_events),
                            "pool_favorite_win_rate": pool_fav_win_rate,
                            "upset_rate": upset_rate,
                            "draw_result_rate": draw_results_rate,
                            "avg_pool_vs_bk_gap": pool_vs_bk_gap,
                            "history_state": "stats_from_precomputed",
                            "history_empty_reason": None,
                            "history_stats_ready": True,
                            "history_ready": True,
                            "stats_source": "toto_brief_history_stats_dict",
                        }

            # ------------------------------------------------------------------
            # GLOBAL HISTORY CONTEXT FALLBACK
            # In manual MATCHES -> TOTO mode, per-match history is never set.
            # When the user loads Baltbet history on the TOTO tab via
            # set_global_history_context(), that payload is stored in
            # self._run_global_history and used here as a global prior for
            # ALL matches in the current generation run.
            # ------------------------------------------------------------------
            global_ctx = getattr(self, "_run_global_history", None) or self.global_history_context
            if isinstance(global_ctx, dict):
                g_draws = int(global_ctx.get("history_draws_loaded_count", global_ctx.get("draws_loaded_count", 0)))
                if g_draws > draws_count:
                    draws_count = g_draws
                if draws_count > 0 and history_state == "history_not_requested":
                    history_state = "global_draw_context_only"
                    history_empty_reason = "draws_only_no_events"

                # Option A: raw events list available — inject and compute fresh stats.
                global_events_list = global_ctx.get("events")
                if isinstance(global_events_list, list) and len(global_events_list) > 0:
                    for ev in global_events_list:
                        if isinstance(ev, dict):
                            events.append(ev)
                    events_count = len(events)
                    history_state = "global_history_baltbet"
                    history_empty_reason = None
                    history_stats_ready = True
                    if g_draws > draws_count:
                        draws_count = g_draws

                # Option B: only pre-computed stats available (no raw events).
                if events_count == 0:
                    global_stats = global_ctx.get("stats")
                    if isinstance(global_stats, dict):
                        precomputed_events = self._safe_float(global_stats.get("events_count", 0.0))
                        if precomputed_events > 0:
                            upset_rate_g = self._safe_float(global_stats.get("upset_rate", 0.0))
                            draw_results_rate_g = self._safe_float(global_stats.get("draw_results_rate", 0.0))
                            pool_fav_win_rate_g = max(0.0, 1.0 - upset_rate_g - draw_results_rate_g * 0.5)
                            pool_vs_bk_gap_g = self._safe_float(global_stats.get("pool_vs_bookmaker_gap", 0.0))
                            g_draws = int(global_ctx.get("history_draws_loaded_count", global_ctx.get("draws_loaded_count", 0)))
                            return {
                                "draws_count": max(draws_count, g_draws),
                                "events_count": int(precomputed_events),
                                "pool_favorite_win_rate": pool_fav_win_rate_g,
                                "upset_rate": upset_rate_g,
                                "draw_result_rate": draw_results_rate_g,
                                "avg_pool_vs_bk_gap": pool_vs_bk_gap_g,
                                "history_state": "global_history_baltbet",
                                "history_empty_reason": None,
                                "history_stats_ready": True,
                                "history_ready": True,
                                "stats_source": "global_history_context_stats",
                            }

            # Still no events anywhere — return empty stats.
            if events_count == 0:
                if draws_count > 0 and history_state not in ("history_not_requested",):
                    history_state = "draws_loaded_no_event_propagation"
                return {
                    "draws_count": draws_count,
                    "events_count": 0,
                    "pool_favorite_win_rate": 0.0,
                    "upset_rate": 0.0,
                    "draw_result_rate": 0.0,
                    "avg_pool_vs_bk_gap": 0.0,
                    "history_state": history_state,
                    "history_empty_reason": history_empty_reason,
                    "history_stats_ready": history_stats_ready,
                    "history_ready": False,
                }
            # Option A injected global events — fall through to compute stats below.

        # Compute stats from raw events.
        favorite_wins = 0.0
        upsets = 0.0
        draw_results = 0.0
        gap_values: list[float] = []

        for ev in events:
            result = str(ev.get("result", "")).strip().upper()
            if result in ("X", "DRAW", "D"):
                draw_results += 1.0
                result = "X"
            if result not in ("1", "X", "2"):
                continue

            pool_map = {
                "1": self._safe_float(ev.get("pool_win_1")),
                "X": self._safe_float(ev.get("pool_draw")),
                "2": self._safe_float(ev.get("pool_win_2")),
            }
            bk_map = {
                "1": self._safe_float(ev.get("bk_win_1")),
                "X": self._safe_float(ev.get("bk_draw")),
                "2": self._safe_float(ev.get("bk_win_2")),
            }

            pool_total = pool_map["1"] + pool_map["X"] + pool_map["2"]
            bk_total = bk_map["1"] + bk_map["X"] + bk_map["2"]
            if pool_total <= 0 or bk_total <= 0:
                continue

            pool_probs = {k: pool_map[k] / pool_total for k in ("1", "X", "2")}
            bk_probs = {k: bk_map[k] / bk_total for k in ("1", "X", "2")}
            pool_favorite = max(("1", "X", "2"), key=lambda k: pool_probs[k])
            market_favorite = max(("1", "X", "2"), key=lambda k: bk_probs[k])

            if result == pool_favorite:
                favorite_wins += 1.0
            else:
                upsets += 1.0

            if pool_favorite != market_favorite and result == market_favorite:
                upsets += 0.5

            gap_values.append(
                abs(pool_probs["1"] - bk_probs["1"])
                + abs(pool_probs["X"] - bk_probs["X"])
                + abs(pool_probs["2"] - bk_probs["2"])
            )

        denom = float(max(events_count, 1))
        return {
            "draws_count": draws_count,
            "events_count": events_count,
            "pool_favorite_win_rate": favorite_wins / denom,
            "upset_rate": upsets / denom,
            "draw_result_rate": draw_results / denom,
            "avg_pool_vs_bk_gap": (sum(gap_values) / len(gap_values)) if gap_values else 0.0,
            "history_state": history_state,
            "history_empty_reason": history_empty_reason,
            "history_stats_ready": history_stats_ready,
            "history_ready": bool(events_count > 0),
        }

    def _derive_history_signals(
        self,
        probs: dict[str, float],
        history_stats: dict[str, Any],
        public: dict[str, float] | None,
    ) -> dict[str, Any]:
        events_count = int(history_stats.get("events_count", 0))
        history_ready = bool(history_stats.get("history_ready", False))

        if events_count <= 0 and not history_ready:
            draws_count = int(history_stats.get("draws_count", 0))
            public_bias = 0.0
            draw_delta = 0.0
            if public is not None:
                public_bias = max(
                    abs(public["P1"] - probs["P1"]),
                    abs(public["PX"] - probs["PX"]),
                    abs(public["P2"] - probs["P2"]),
                )
                draw_delta = max(0.0, public["PX"] - probs["PX"])
            draw_context_scale = min(0.25, float(draws_count) / 800.0) if draws_count > 0 else 0.0
            fallback_history_score = min(1.0, 0.70 * public_bias + 0.30 * draw_context_scale)
            return {
                "favorite_overload_signal": 0.0,
                "underdog_neglect_signal": 0.0,
                "pool_vs_bk_gap_signal": draw_context_scale,
                "public_bias_signal": public_bias,
                "upset_history_signal": 0.0,
                "draw_underpricing_signal": draw_delta,
                "history_score": fallback_history_score,
                "favorite_trap_score": fallback_history_score,
                "upset_potential": draw_context_scale,
                "draw_tendency_score": draw_delta,
                "history_conflict_with_model": fallback_history_score,
                "historical_favorite_fail_score": fallback_history_score,
                "historical_draw_bias_score": draw_delta,
                "historical_upset_score": draw_context_scale,
                "pool_vs_bk_historical_divergence": draw_context_scale,
                "underdog_conversion_score": max(0.0, probs["P2"] - 0.22),
                "history_result_alignment_with_model": fallback_history_score,
                "event_history_signal_strength": draw_context_scale,
                "signals_used": ["draw_context_prior", "public_bias_signal"],
            }

        pool_favorite_win_rate = float(history_stats.get("pool_favorite_win_rate", 0.0))
        upset_rate = float(history_stats.get("upset_rate", 0.0))
        draw_result_rate = float(history_stats.get("draw_result_rate", 0.0))
        avg_pool_vs_bk_gap = float(history_stats.get("avg_pool_vs_bk_gap", 0.0))

        # Scale signals by calibrated baselines.
        # Pool favorites win roughly 50-55% in balanced leagues.
        # Upset rate is typically 25-35%.
        favorite_overload = max(0.0, 0.55 - pool_favorite_win_rate)
        underdog_neglect = max(0.0, upset_rate - 0.25)
        pool_vs_bk_gap = min(1.0, avg_pool_vs_bk_gap)

        public_bias = 0.0
        if public is not None:
            public_bias = max(
                abs(public["P1"] - probs["P1"]),
                abs(public["PX"] - probs["PX"]),
                abs(public["P2"] - probs["P2"]),
            )

        upset_history = max(0.0, upset_rate - 0.22)
        draw_underpricing = max(0.0, draw_result_rate - probs["PX"])
        favorite_trap_score = min(1.0, max(0.0, 0.55 * favorite_overload + 0.45 * underdog_neglect))
        upset_potential = min(1.0, max(0.0, 0.55 * upset_history + 0.45 * underdog_neglect))
        draw_tendency_score = min(1.0, max(0.0, draw_underpricing))
        history_score = min(1.0, max(0.0, 0.35 * favorite_trap_score + 0.35 * upset_potential + 0.30 * draw_tendency_score))
        history_conflict_with_model = min(1.0, max(0.0, public_bias + upset_potential))

        # ------------------------------------------------------------------
        # Rich event-history signals (only meaningful when events_count > 0).
        # These represent Baltbet-specific historical patterns.
        # ------------------------------------------------------------------
        # historical_favorite_fail_score: how often pool-favorites actually fail
        #   > 0 means history shows pool overvalues favorites (trap risk)
        historical_favorite_fail_score = favorite_overload

        # historical_draw_bias_score: mismatch between historical draw rate and
        #   model's draw probability — positive means draws are underrepresented
        #   by model relative to history
        historical_draw_bias_score = max(0.0, draw_result_rate - probs["PX"] * 0.85)

        # historical_upset_score: frequency of upset outcomes in historical data
        historical_upset_score = upset_potential

        # pool_vs_bk_historical_divergence: historical average pool vs bookmaker
        #   divergence — high means disagreement was systematically present
        pool_vs_bk_historical_divergence = pool_vs_bk_gap

        # underdog_conversion_score: how often underdogs beat pool expectations
        #   proxy = upset_rate adjusted for upset rate baseline
        underdog_conversion_score = min(1.0, max(0.0, (upset_rate - 0.20) * 2.5))

        # history_result_alignment_with_model: 0 = perfect match, positive = conflict
        #   uses draw rate as primary indicator (model's PX vs historical draw rate)
        history_result_alignment_with_model = min(1.0, max(0.0, abs(draw_result_rate - probs["PX"]) + public_bias * 0.5))

        # Combined event-history signal strength — used to decide if event-history
        # mode should override thresholds in _apply_history_strategy.
        event_history_signal_strength = min(1.0, max(0.0,
            0.25 * historical_favorite_fail_score
            + 0.25 * historical_upset_score
            + 0.20 * historical_draw_bias_score
            + 0.15 * pool_vs_bk_historical_divergence
            + 0.15 * underdog_conversion_score
        )) if events_count > 0 else 0.0

        signals_used = [
            "favorite_overload_signal",
            "underdog_neglect_signal",
            "pool_vs_bk_gap_signal",
            "public_bias_signal",
            "upset_history_signal",
            "draw_underpricing_signal",
            "history_score",
            "favorite_trap_score",
            "upset_potential",
            "draw_tendency_score",
            "history_conflict_with_model",
            "contrarian_signal",
        ]
        if events_count > 0:
            signals_used += [
                "historical_favorite_fail_score",
                "historical_draw_bias_score",
                "historical_upset_score",
                "pool_vs_bk_historical_divergence",
                "underdog_conversion_score",
                "history_result_alignment_with_model",
                "event_history_signal_strength",
            ]

        return {
            "favorite_overload_signal": favorite_overload,
            "underdog_neglect_signal": underdog_neglect,
            "pool_vs_bk_gap_signal": pool_vs_bk_gap,
            "public_bias_signal": public_bias,
            "upset_history_signal": upset_history,
            "draw_underpricing_signal": draw_underpricing,
            "history_score": history_score,
            "favorite_trap_score": favorite_trap_score,
            "upset_potential": upset_potential,
            "draw_tendency_score": draw_tendency_score,
            "history_conflict_with_model": history_conflict_with_model,
            "historical_favorite_fail_score": historical_favorite_fail_score,
            "historical_draw_bias_score": historical_draw_bias_score,
            "historical_upset_score": historical_upset_score,
            "pool_vs_bk_historical_divergence": pool_vs_bk_historical_divergence,
            "underdog_conversion_score": underdog_conversion_score,
            "history_result_alignment_with_model": history_result_alignment_with_model,
            "event_history_signal_strength": event_history_signal_strength,
            "signals_used": signals_used,
        }

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _feature_value(self, match: dict[str, Any], key: str, default: float = 0.0) -> float:
        if not isinstance(match, dict):
            return default
        features = match.get("features")
        if isinstance(features, dict) and key in features:
            val = self._safe_float(features.get(key))
            return val if val != 0.0 or features.get(key) in (0, 0.0, "0", "0.0") else default
        raw = match.get(key)
        val = self._safe_float(raw)
        return val if val != 0.0 or raw in (0, 0.0, "0", "0.0") else default

    def _context_completeness_score(self, match: dict[str, Any]) -> float:
        level = str(match.get("feature_context_level", "")).strip().lower()
        if level == "full_context":
            return 1.0
        if level == "partial_context":
            return 0.72
        if level == "odds_only_context":
            return 0.42
        if level == "degraded_context":
            return 0.22
        return 0.60

    def _pick_toto_secondary_outcome(
        self,
        primary: str,
        profile: dict[str, Any],
        toto_interp: dict[str, Any],
    ) -> str:
        probs = profile.get("probs", {}) if isinstance(profile, dict) else {}
        if not isinstance(probs, dict):
            return ""

        p1 = self._safe_float(probs.get("P1"))
        px = self._safe_float(probs.get("PX"))
        p2 = self._safe_float(probs.get("P2"))

        draw_risk = self._safe_float(toto_interp.get("draw_risk_score"))
        upset_risk = self._safe_float(toto_interp.get("upset_risk_score"))

        if primary in {"1", "2"} and draw_risk >= 0.36 and px >= max(0.18, max(p1, p2) - 0.18):
            return "X"

        ranked = self._rank_outcomes({"P1": p1, "PX": px, "P2": p2})
        for outcome, _ in ranked:
            if outcome != primary:
                if primary in {"1", "2"} and outcome in {"1", "2"} and upset_risk < 0.36:
                    continue
                return outcome
        return ""

    def _toto_match_interpretation(
        self,
        match: dict[str, Any],
        profile: dict[str, Any],
        strategy: dict[str, Any],
        decision_class: str,
    ) -> dict[str, Any]:
        probs = profile.get("probs", {}) if isinstance(profile, dict) else {}
        p1 = self._safe_float(probs.get("P1")) if isinstance(probs, dict) else 0.0
        px = self._safe_float(probs.get("PX")) if isinstance(probs, dict) else 0.0
        p2 = self._safe_float(probs.get("P2")) if isinstance(probs, dict) else 0.0

        top_prob = self._safe_float(profile.get("top_prob"))
        second_prob = self._safe_float(profile.get("second_prob"))
        margin_top_second = self._safe_float(profile.get("margin_top_second"))
        uncertainty = self._safe_float(profile.get("uncertainty"))
        top_outcome = str(profile.get("top_outcome", "1"))

        entropy_feature = self._feature_value(match, "entropy", uncertainty)
        gap_feature = self._feature_value(match, "gap", margin_top_second)
        volatility_feature = self._feature_value(match, "volatility", 0.0)
        ppg_diff = abs(self._feature_value(match, "ppg_diff", 0.0))
        split_advantage = abs(self._feature_value(match, "split_advantage", 0.0))
        goals_diff = abs(self._feature_value(match, "goals_diff", 0.0))
        xg_diff = abs(self._feature_value(match, "xg_diff", 0.0))
        shots_diff = abs(self._feature_value(match, "shots_diff", 0.0))

        context_score = self._context_completeness_score(match)
        raw_market_disagreement = self._clamp01(
            max(
                self._safe_float(strategy.get("pool_vs_model_bias")),
                self._safe_float(strategy.get("public_bias")),
                self._safe_float(strategy.get("history_conflict_with_model")),
                self._safe_float(strategy.get("pool_vs_bk_divergence")),
            )
        )
        market_disagreement = self._clamp01(raw_market_disagreement * (0.65 + 0.35 * context_score))

        feature_balance = self._clamp01(
            1.0
            - min(
                1.0,
                (0.42 * ppg_diff) + (0.32 * split_advantage) + (0.16 * goals_diff) + (0.10 * xg_diff),
            )
        )

        draw_risk_score = self._clamp01(
            0.34 * px
            + 0.18 * self._safe_float(strategy.get("draw_tendency_score"))
            + 0.12 * self._safe_float(strategy.get("historical_draw_bias_score"))
            + 0.14 * feature_balance
            + 0.10 * uncertainty
            + 0.06 * entropy_feature
            + 0.06 * volatility_feature
        )

        upset_risk_score = self._clamp01(
            0.23 * self._safe_float(strategy.get("upset_potential"))
            + 0.18 * self._safe_float(strategy.get("historical_upset_score"))
            + 0.13 * self._safe_float(strategy.get("underdog_conversion_score"))
            + 0.18 * self._safe_float(strategy.get("favorite_trap_score"))
            + 0.18 * market_disagreement
            + 0.10 * (1.0 - context_score)
        )

        weak_favorite_score = 0.0
        if top_outcome in {"1", "2"}:
            weak_favorite_score = self._clamp01(
                0.34 * (1.0 - min(1.0, max(0.0, (top_prob - 0.36) / 0.26)))
                + 0.22 * (1.0 - min(1.0, max(0.0, margin_top_second / 0.24)))
                + 0.20 * market_disagreement
                + 0.12 * draw_risk_score
                + 0.12 * (1.0 - context_score)
            )

        trap_match_score = self._clamp01(
            max(
                0.36 * weak_favorite_score
                + 0.24 * self._safe_float(strategy.get("favorite_trap_score"))
                + 0.18 * self._safe_float(strategy.get("history_conflict_with_model"))
                + 0.12 * market_disagreement
                + 0.10 * (1.0 - context_score),
                0.45 * self._safe_float(strategy.get("event_history_signal_strength"))
                + 0.25 * self._safe_float(strategy.get("historical_favorite_fail_score"))
                + 0.30 * market_disagreement,
            )
        )

        feature_support_score = self._clamp01(
            min(1.0, (0.40 * ppg_diff) + (0.30 * split_advantage) + (0.20 * goals_diff) + (0.10 * shots_diff))
        )

        single_confidence_score = self._clamp01(
            (0.46 * top_prob)
            + (0.16 * margin_top_second)
            + (0.12 * feature_support_score)
            + (0.10 * context_score)
            - (0.08 * draw_risk_score)
            - (0.08 * upset_risk_score)
            - (0.06 * trap_match_score)
            - (0.06 * uncertainty)
            - (0.04 * entropy_feature)
            - (0.04 * volatility_feature)
            + 0.26
        )

        plausible_alternative = bool(
            (second_prob >= top_prob - 0.15)
            or (px >= 0.31 and top_outcome in {"1", "2"})
            or (market_disagreement >= 0.10)
        )

        trap_double_threshold = 0.56
        trap_match_type_threshold = 0.62
        insurance_priority_threshold = 0.66

        double_recommendation_flag = bool(
            (single_confidence_score < 0.56)
            or (draw_risk_score >= 0.36)
            or (weak_favorite_score >= 0.46)
            or (trap_match_score >= trap_double_threshold)
            or (decision_class == "high_uncertainty")
        )

        insurance_priority_score = self._clamp01(
            0.40 * trap_match_score
            + 0.23 * upset_risk_score
            + 0.19 * draw_risk_score
            + 0.10 * market_disagreement
            + 0.08 * (1.0 - context_score)
            + (0.05 if plausible_alternative else 0.0)
        )

        if context_score < 0.35 or uncertainty >= 0.97:
            match_type = "High-uncertainty match"
        elif trap_match_score >= trap_match_type_threshold:
            match_type = "Trap-like match"
        elif draw_risk_score >= 0.44 and abs(p1 - p2) <= 0.14:
            match_type = "Draw-heavy balance"
        elif weak_favorite_score >= 0.46 and top_outcome in {"1", "2"}:
            match_type = "Weak favorite"
        elif market_disagreement >= 0.18 and plausible_alternative:
            match_type = "Form-vs-market conflict"
        else:
            match_type = "Strong favorite"

        if trap_match_score >= trap_match_type_threshold:
            risk_level = "trap-like"
        elif insurance_priority_score >= insurance_priority_threshold:
            risk_level = "high-risk"
        elif double_recommendation_flag or single_confidence_score < 0.62:
            risk_level = "medium-risk"
        else:
            risk_level = "low-risk"

        if insurance_priority_score >= insurance_priority_threshold and plausible_alternative:
            resolution_type = "insurance priority"
        elif double_recommendation_flag:
            resolution_type = "double"
        else:
            resolution_type = "single"

        return {
            "draw_risk_score": draw_risk_score,
            "upset_risk_score": upset_risk_score,
            "weak_favorite_score": weak_favorite_score,
            "trap_match_score": trap_match_score,
            "single_confidence_score": single_confidence_score,
            "double_recommendation_flag": double_recommendation_flag,
            "insurance_priority_score": insurance_priority_score,
            "market_model_disagreement_score": market_disagreement,
            "context_completeness_score": context_score,
            "match_type": match_type,
            "risk_level": risk_level,
            "resolution_type": resolution_type,
        }

    def _strategy_reason_codes(self, ctx: dict[str, Any], decision_class: str) -> list[str]:
        codes: list[str] = []

        if bool(ctx.get("pool_signals_used", False)):
            codes.append("pool_conflict")
        if bool(ctx.get("pool_public_used", False)) or self._safe_float(ctx.get("public_bias_score")) >= 0.06:
            codes.append("public_bias")
        if bool(ctx.get("history_used", False)) or self._safe_float(ctx.get("history_conflict_with_model")) >= 0.08:
            codes.append("history_conflict")
        if (
            self._safe_float(ctx.get("pool_vs_bookmaker_divergence")) >= 0.04
            or self._safe_float(ctx.get("pool_vs_bk_historical_divergence")) >= 0.04
        ):
            codes.append("bookmaker_pool_divergence")
        if self._safe_float(ctx.get("favorite_trap_score")) >= 0.10 or self._safe_float(ctx.get("historical_favorite_fail_score")) >= 0.10:
            codes.append("favorite_fail_risk")
        if self._safe_float(ctx.get("draw_tendency_score")) >= 0.08 or self._safe_float(ctx.get("historical_draw_bias_score")) >= 0.08:
            codes.append("draw_risk")
        if self._safe_float(ctx.get("upset_potential")) >= 0.10 or self._safe_float(ctx.get("historical_upset_score")) >= 0.10:
            codes.append("upset_risk")
        if self._safe_float(ctx.get("event_history_signal_strength")) >= 0.12:
            codes.append("trap_signal")
        if decision_class.startswith("balanced_") or decision_class == "high_uncertainty":
            codes.append("model_uncertainty")

        has_market_context = bool(ctx.get("pool_signals_used", False) or ctx.get("bk_signals_used", False) or ctx.get("history_used", False))
        if not has_market_context:
            codes.append("insufficient_market_context")

        if not codes:
            codes.append("no_strong_toto_signal")

        return sorted(set(codes))

    def _insurance_reason_codes(
        self,
        base_outcome: str,
        alternatives: list[str],
        reason_scores: dict[str, Any],
    ) -> list[str]:
        if not alternatives:
            return ["preserved_base_decision", "no_extra_cover"]

        codes: list[str] = []

        if "X" in alternatives:
            codes.append("draw_cover")

        opposite = self._opposite_outcome(base_outcome)
        if opposite and opposite in alternatives:
            codes.append("opposite_safety")
            strongest = max((self._safe_float(v) for v in reason_scores.values()), default=0.0)
            if strongest < 0.14:
                codes.append("low_justification_for_opposite")
        if any(outcome in alternatives for outcome in ("1", "2")) and not opposite:
            codes.append("adjacent_safety")

        if self._safe_float(reason_scores.get("favorite_fail_risk")) >= 0.12:
            codes.append("favorite_fail_cover")
        if self._safe_float(reason_scores.get("upset_risk")) >= 0.12:
            codes.append("upset_cover")
        if self._safe_float(reason_scores.get("public_conflict")) >= 0.10 or self._safe_float(reason_scores.get("uncertainty_conflict")) >= 0.10:
            codes.append("diversification_cover")

        if not codes:
            codes.append("no_extra_cover")

        return sorted(set(codes))

    @staticmethod
    def _strategy_reason_human(codes: list[str]) -> str:
        mapping = {
            "draw_risk": "риск ничьей",
            "favorite_fail_risk": "риск срыва фаворита",
            "upset_risk": "риск апсета",
            "pool_conflict": "конфликт с пулом",
            "history_conflict": "конфликт с историческим профилем",
            "public_bias": "перекос public-пула",
            "trap_signal": "ловушечный профиль",
            "model_uncertainty": "неуверенность модели",
            "insufficient_market_context": "недостаточный рыночный контекст",
            "bookmaker_pool_divergence": "дивергенция pool vs bookmaker",
            "no_strong_toto_signal": "сильный TOTO-сигнал не найден",
        }
        parts = [mapping.get(code, code) for code in codes]
        return ", ".join(parts) if parts else "not available"

    @staticmethod
    def _insurance_reason_human(codes: list[str]) -> str:
        mapping = {
            "adjacent_safety": "соседнее страхование",
            "opposite_safety": "страхование противоположного исхода",
            "draw_cover": "добавлено покрытие ничьи",
            "favorite_fail_cover": "покрытие риска срыва фаворита",
            "upset_cover": "покрытие риска апсета",
            "diversification_cover": "диверсификация покрытия",
            "no_extra_cover": "дополнительное покрытие не добавлено",
            "low_justification_for_opposite": "противоположный исход добавлен со слабым обоснованием",
            "preserved_base_decision": "базовое решение сохранено",
        }
        parts = [mapping.get(code, code) for code in codes]
        return ", ".join(parts) if parts else "not available"

    def _strategy_justification_strength(
        self,
        ctx: dict[str, Any],
        reason_codes: list[str],
        strategy_changed: bool,
    ) -> str:
        signal_strength = max(
            self._safe_float(ctx.get("history_adjustment")),
            self._safe_float(ctx.get("market_adjustment")),
            self._safe_float(ctx.get("history_conflict_with_model")),
            self._safe_float(ctx.get("favorite_trap_score")),
            self._safe_float(ctx.get("upset_potential")),
            self._safe_float(ctx.get("draw_tendency_score")),
            self._safe_float(ctx.get("event_history_signal_strength")),
            self._safe_float(ctx.get("pool_vs_bookmaker_divergence")),
        )

        if not strategy_changed and signal_strength < 0.10:
            return "weak"
        if signal_strength >= 0.20 or len(reason_codes) >= 4:
            return "strong"
        if signal_strength >= 0.10 or len(reason_codes) >= 2:
            return "medium"
        return "weak"

    def _insurance_justification_strength(
        self,
        reason_codes: list[str],
        reason_scores: dict[str, Any],
        insurance_changed: bool,
    ) -> str:
        if not insurance_changed:
            return "weak"

        strongest = max((self._safe_float(v) for v in reason_scores.values()), default=0.0)
        if strongest >= 0.20 or len(reason_codes) >= 4:
            return "strong"
        if strongest >= 0.10 or len(reason_codes) >= 2:
            return "medium"
        return "weak"

    def _build_match_layer_diagnostics(
        self,
        matches: list[dict],
        base_contexts: list[dict[str, Any]],
        base_decisions: list[str],
        final_coverages: list[list[str]],
        coupons: list[list[str]] | None = None,
        base_coupon_count: int | None = None,
        insurance_meta_by_match: dict[int, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        insurance_meta_by_match = insurance_meta_by_match or {}

        for idx, (match, ctx, strategy_decision, final_coverage) in enumerate(
            zip(matches, base_contexts, base_decisions, final_coverages)
        ):
            probs = self._normalised_probs(match["probs"])
            base_model_decision = str(ctx.get("model_decision", strategy_decision))
            strategy_decision = str(strategy_decision)
            strategy_changed = base_model_decision != strategy_decision

            strategy_outcomes = self._options_from_decision(strategy_decision, probs)
            insurance_meta = insurance_meta_by_match.get(idx, {})
            raw_insurance_alternatives = insurance_meta.get("alternatives", [])
            insurance_alternatives = [
                str(outcome)
                for outcome in raw_insurance_alternatives
                if str(outcome) in ALLOWED_OUTCOMES
            ] if isinstance(raw_insurance_alternatives, list) else []

            actual_outcomes_all: set[str] = set()
            actual_outcomes_insurance: set[str] = set()
            if isinstance(coupons, list) and coupons:
                for line_idx, coupon in enumerate(coupons):
                    if not isinstance(coupon, list) or idx >= len(coupon):
                        continue
                    token = str(coupon[idx]).strip()
                    if token not in ALLOWED_OUTCOMES:
                        continue
                    actual_outcomes_all.add(token)
                    if base_coupon_count is not None and line_idx >= int(base_coupon_count):
                        actual_outcomes_insurance.add(token)

            factual_coverage = sorted(actual_outcomes_all) if actual_outcomes_all else list(final_coverage)

            # Keep strategy-vs-insurance semantics strict and factual: only explicit
            # insurance alternatives that actually appear in insurance coupon lines
            # count as insurance layer additions.
            insurance_added_outcomes = sorted(
                [
                    outcome
                    for outcome in insurance_alternatives
                    if outcome in actual_outcomes_insurance and outcome not in strategy_outcomes
                ]
            )
            insurance_changed = bool(insurance_added_outcomes)

            insurance_reason_scores = insurance_meta.get("reason_scores", {})
            if not isinstance(insurance_reason_scores, dict):
                insurance_reason_scores = {}

            strategy_reason_codes = self._strategy_reason_codes(
                ctx=ctx,
                decision_class=str(ctx.get("decision_class", "unknown")),
            )
            insurance_reason_codes = (
                self._insurance_reason_codes(
                    base_outcome=self._single_from_decision(strategy_decision, probs),
                    alternatives=insurance_added_outcomes,
                    reason_scores=insurance_reason_scores,
                )
                if insurance_changed
                else ["preserved_base_decision", "no_extra_cover"]
            )

            strategy_justification_strength = self._strategy_justification_strength(
                ctx=ctx,
                reason_codes=strategy_reason_codes,
                strategy_changed=strategy_changed,
            )
            insurance_justification_strength = self._insurance_justification_strength(
                reason_codes=insurance_reason_codes,
                reason_scores=insurance_reason_scores,
                insurance_changed=insurance_changed,
            )

            rows.append(
                {
                    "match_index": idx + 1,
                    "match": self._match_label(match),
                    "base_model_probabilities": {
                        "p1": float(probs["P1"]),
                        "px": float(probs["PX"]),
                        "p2": float(probs["P2"]),
                    },
                    "base_model_decision": base_model_decision,
                    "toto_strategy_adjusted_decision": strategy_decision,
                    "strategy_changed_flag": strategy_changed,
                    "strategy_reason_codes": strategy_reason_codes,
                    "strategy_justification_strength": strategy_justification_strength,
                    "strategy_human_explanation": self._strategy_reason_human(strategy_reason_codes),
                    "match_type": str(ctx.get("match_type", "High-uncertainty match")),
                    "risk_level": str(ctx.get("risk_level", "medium-risk")),
                    "resolution_type": str(ctx.get("resolution_type", "double")),
                    "draw_risk_score": self._safe_float(ctx.get("draw_risk_score")),
                    "upset_risk_score": self._safe_float(ctx.get("upset_risk_score")),
                    "weak_favorite_score": self._safe_float(ctx.get("weak_favorite_score")),
                    "trap_match_score": self._safe_float(ctx.get("trap_match_score")),
                    "single_confidence_score": self._safe_float(ctx.get("single_confidence_score")),
                    "double_recommendation_flag": bool(ctx.get("double_recommendation_flag", False)),
                    "insurance_priority_score": self._safe_float(ctx.get("insurance_priority_score")),
                    "market_model_disagreement_score": self._safe_float(ctx.get("market_model_disagreement_score")),
                    "context_completeness_score": self._safe_float(ctx.get("context_completeness_score")),
                    "toto_layer_used": bool(ctx.get("toto_layer_used", False)),
                    "insurance_added_outcomes": insurance_added_outcomes,
                    "insurance_changed_flag": insurance_changed,
                    "insurance_reason_codes": insurance_reason_codes,
                    "insurance_justification_strength": insurance_justification_strength,
                    "insurance_human_explanation": self._insurance_reason_human(insurance_reason_codes),
                    "final_effective_coverage": factual_coverage,
                }
            )

        return rows

    def _build_coupon_entries(
        self,
        coupons: list[list[str]],
        base_coupon_count: int,
        changed_matches_by_line: list[list[int]] | None = None,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        changed_matches_by_line = changed_matches_by_line or []

        for idx, coupon in enumerate(coupons):
            is_insurance_coupon = idx >= base_coupon_count
            insured_line_idx = idx - base_coupon_count
            changed_matches = (
                list(changed_matches_by_line[insured_line_idx])
                if is_insurance_coupon and 0 <= insured_line_idx < len(changed_matches_by_line)
                else []
            )
            entries.append(
                {
                    "coupon_index": idx,
                    "coupon_type": "insurance" if is_insurance_coupon else "base",
                    "source_stage": "insurance_layer" if is_insurance_coupon else "base_strategy_layer",
                    "insurance_applied_flag": bool(is_insurance_coupon),
                    "changed_match_indexes": changed_matches,
                    "coverage": list(coupon),
                }
            )

        return entries

    def _coupon_uniqueness_stats(self, coupons: list[list[str]]) -> dict[str, int]:
        generated = len(coupons)
        unique = len({tuple(coupon) for coupon in coupons})
        return {
            "generated_coupon_count": generated,
            "unique_coupon_count": unique,
            "duplicate_coupon_count": max(generated - unique, 0),
        }

    def _ensure_unique_coupon_set(
        self,
        coupons: list[list[str]],
        target_lines: int,
        pools: list[list[str]],
        matches: list[dict],
    ) -> list[list[str]]:
        unique: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        for coupon in coupons:
            key = tuple(coupon)
            if key in seen:
                continue
            seen.add(key)
            unique.append(list(coupon))
            if len(unique) >= target_lines:
                return unique[:target_lines]

        # Refill with additional unique candidates while keeping deterministic behavior.
        for strength in (0.10, 0.18, 0.28, 0.40):
            if len(unique) >= target_lines:
                break
            candidates = self._assemble_coupons(
                pools=pools,
                matches=matches,
                target_lines=target_lines,
                diversity_strength=strength,
            )
            for candidate in candidates:
                key = tuple(candidate)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(candidate)
                if len(unique) >= target_lines:
                    break

        # Last-resort deterministic expansion by single-cell mutations.
        if len(unique) < target_lines and unique:
            mutation_seed = list(unique)
            for base_coupon in mutation_seed:
                if len(unique) >= target_lines:
                    break
                for col_idx, allowed in enumerate(pools):
                    if len(unique) >= target_lines:
                        break
                    options = [str(x) for x in allowed if str(x) in ALLOWED_OUTCOMES]
                    for alt in options:
                        if col_idx >= len(base_coupon) or alt == base_coupon[col_idx]:
                            continue
                        mutated = list(base_coupon)
                        mutated[col_idx] = alt
                        key = tuple(mutated)
                        if key in seen:
                            continue
                        seen.add(key)
                        unique.append(mutated)
                        if len(unique) >= target_lines:
                            break

        return unique[:target_lines]

    def _match_label(self, match: dict) -> str:
        home = str(match.get("home", "")).strip()
        away = str(match.get("away", "")).strip()
        if home or away:
            return f"{home or '?'} vs {away or '?'}"
        name = str(match.get("name", "")).strip()
        if name:
            return name
        return str(match.get("match_id", "?"))

    def _model_distribution(self, matches: list[dict]) -> dict[str, int]:
        distribution = {"top1_1": 0, "top1_X": 0, "top1_2": 0}
        for match in matches:
            probs = self._normalised_probs(match["probs"])
            top = max(("1", "X", "2"), key=lambda outcome: probs[_PROB_KEY[outcome]])
            distribution[f"top1_{top}"] += 1
        return distribution

    def _sort_indexes_by_risk(self, matches: list[dict]) -> list[int]:
        """Sort match indexes by risk score descending.

        Combines model entropy (uncertainty) with a quick pool-vs-model signal
        when pool_probs is available in the match dict.  Higher score = riskier
        match = more likely to be selected for insurance pool expansion.
        """
        def risk(index: int) -> float:
            match = matches[index]
            probs = self._normalised_probs(match["probs"])
            entropy = 0.0
            for value in (probs["P1"], probs["PX"], probs["P2"]):
                if value > 0:
                    entropy -= value * math.log(value, 2)

            # Quick pool divergence boost (doesn't call full strategy extraction).
            pool_raw = match.get("pool_probs")
            pool_boost = 0.0
            if isinstance(pool_raw, dict):
                pP1 = self._safe_float(pool_raw.get("P1") or pool_raw.get("p1"))
                pPX = self._safe_float(pool_raw.get("PX") or pool_raw.get("px"))
                pP2 = self._safe_float(pool_raw.get("P2") or pool_raw.get("p2"))
                total = pP1 + pPX + pP2
                if total > 0.05:
                    pP1 /= total
                    pPX /= total
                    pP2 /= total
                    divergence = max(
                        abs(probs["P1"] - pP1),
                        abs(probs["PX"] - pPX),
                        abs(probs["P2"] - pP2),
                    )
                    pool_boost = min(0.25, divergence * 1.5)

            return entropy + pool_boost

        return sorted(range(len(matches)), key=risk, reverse=True)

    def _options_from_decision(self, decision: str, probs: dict[str, float]) -> list[str]:
        candidates = [outcome for outcome in decision if outcome in ALLOWED_OUTCOMES]
        if not candidates:
            return [self._single_from_decision(decision=decision, probs=probs)]

        unique_candidates = list(dict.fromkeys(candidates))
        if len(unique_candidates) == 1:
            return unique_candidates
        return sorted(unique_candidates, key=lambda outcome: probs[_PROB_KEY[outcome]], reverse=True)

    def _single_from_decision(self, decision: str, probs: dict[str, float]) -> str:
        candidates = [outcome for outcome in decision if outcome in ALLOWED_OUTCOMES]
        if not candidates:
            return max(("1", "X", "2"), key=lambda outcome: probs[_PROB_KEY[outcome]])
        return max(candidates, key=lambda outcome: probs[_PROB_KEY[outcome]])

    def _coupon_log_weight(self, coupon: list[str], matches: list[dict]) -> float:
        weight = 0.0
        for idx, outcome in enumerate(coupon):
            prob = max(self._normalised_probs(matches[idx]["probs"])[_PROB_KEY[outcome]], 1e-12)
            weight += math.log(prob)
        return weight

    @staticmethod
    def _hamming_distance(a: list[str], b: list[str]) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    def _diversity_bonus(
        self,
        coupon: list[str],
        per_column_usage: list[dict[str, int]],
        strength: float,
    ) -> float:
        bonus = 0.0
        for idx, outcome in enumerate(coupon):
            usage = per_column_usage[idx].get(outcome, 0)
            bonus += 1.0 / (1.0 + usage)
        return bonus * strength

    def _normalised_probs(self, probs: dict[str, float]) -> dict[str, float]:
        p1 = max(float(probs.get("P1", 0.0)), 0.0)
        px = max(float(probs.get("PX", 0.0)), 0.0)
        p2 = max(float(probs.get("P2", 0.0)), 0.0)
        total = p1 + px + p2
        if total <= 0:
            return {"P1": 1.0 / 3.0, "PX": 1.0 / 3.0, "P2": 1.0 / 3.0}
        return {"P1": p1 / total, "PX": px / total, "P2": p2 / total}

    def _rank_outcomes(self, probs: dict[str, float]) -> list[tuple[str, float]]:
        return sorted(
            (("1", probs["P1"]), ("X", probs["PX"]), ("2", probs["P2"])),
            key=lambda item: item[1],
            reverse=True,
        )

    def _decision_from_outcomes(self, outcomes: list[str]) -> str:
        ordered = sorted(dict.fromkeys(outcomes), key=lambda x: ["1", "X", "2"].index(x))
        return "".join(ordered[:2]) if ordered else "1"

    def _decision_distribution(self, decisions: list[str]) -> dict[str, int]:
        distribution = {
            "single_1": 0,
            "single_X": 0,
            "single_2": 0,
            "double_1X": 0,
            "double_X2": 0,
            "double_12": 0,
            "other": 0,
        }
        for decision in decisions:
            key = "other"
            if decision == "1":
                key = "single_1"
            elif decision == "X":
                key = "single_X"
            elif decision == "2":
                key = "single_2"
            elif decision == "1X":
                key = "double_1X"
            elif decision == "X2":
                key = "double_X2"
            elif decision == "12":
                key = "double_12"
            distribution[key] += 1
        return distribution

    def _final_coupon_distribution(self, coupons: list[list[str]]) -> dict[str, int]:
        distribution = {"1": 0, "X": 0, "2": 0}
        for coupon in coupons:
            for outcome in coupon:
                if outcome in distribution:
                    distribution[outcome] += 1
        return distribution

    def _coupon_diversity_metrics(self, coupons: list[list[str]]) -> dict[str, Any]:
        if not coupons:
            return {
                "average_hamming_distance_between_coupons": 0.0,
                "median_hamming_distance": 0.0,
                "min_hamming_distance": 0.0,
                "coupon_diversity_score": 0.0,
                "per_match_entropy": {},
            }

        row_len = len(coupons[0]) if coupons and coupons[0] else 0
        pair_distances: list[int] = []
        for i in range(len(coupons)):
            for j in range(i + 1, len(coupons)):
                a = coupons[i]
                b = coupons[j]
                d = 0
                for k in range(min(len(a), len(b))):
                    if a[k] != b[k]:
                        d += 1
                pair_distances.append(d)

        if pair_distances:
            pair_distances_sorted = sorted(pair_distances)
            avg_dist = sum(pair_distances_sorted) / len(pair_distances_sorted)
            mid = len(pair_distances_sorted) // 2
            if len(pair_distances_sorted) % 2 == 0:
                med_dist = (pair_distances_sorted[mid - 1] + pair_distances_sorted[mid]) / 2
            else:
                med_dist = float(pair_distances_sorted[mid])
            min_dist = float(pair_distances_sorted[0])
        else:
            avg_dist = 0.0
            med_dist = 0.0
            min_dist = 0.0

        per_match_entropy: dict[str, float] = {}
        if row_len > 0:
            for col in range(row_len):
                counts: dict[str, int] = {"1": 0, "X": 0, "2": 0}
                total = 0
                for row in coupons:
                    if col >= len(row):
                        continue
                    token = str(row[col]).strip()
                    if token not in counts:
                        continue
                    counts[token] += 1
                    total += 1
                if total <= 0:
                    per_match_entropy[str(col)] = 0.0
                    continue
                entropy = 0.0
                for token in ("1", "X", "2"):
                    p = counts[token] / total
                    if p > 0:
                        entropy -= p * math.log(p, 2)
                per_match_entropy[str(col)] = entropy

        diversity_score = (avg_dist / row_len) if row_len > 0 else 0.0
        return {
            "average_hamming_distance_between_coupons": avg_dist,
            "median_hamming_distance": med_dist,
            "min_hamming_distance": min_dist,
            "coupon_diversity_score": diversity_score,
            "per_match_entropy": per_match_entropy,
        }

    def _safe_float(self, value: object) -> float:
        if isinstance(value, bool):
            return float(int(value))
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0
