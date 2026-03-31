from __future__ import annotations

import itertools
import logging
import math

logger = logging.getLogger("toto")

ALLOWED_OUTCOMES = {"1", "X", "2"}
_PROB_KEY = {"1": "P1", "X": "PX", "2": "P2"}


class TotoOptimizer:
    def __init__(self) -> None:
        self.last_run_summary: dict[str, object] = {}

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def optimize(self, matches: list[dict], mode: str) -> list[list[str]]:
        if mode not in {"16", "32"}:
            raise ValueError("mode must be '16' or '32'")
        if not matches:
            return []

        ordered_matches = list(matches)
        target_lines = 16 if mode == "16" else 32
        risk_count = 4 if mode == "16" else 5

        sorted_indexes = self._sort_indexes_by_risk(matches=ordered_matches)
        risk_indexes = sorted_indexes[: min(risk_count, len(matches))]

        base_contexts = [self._evaluate_base_decision(match) for match in ordered_matches]
        base_decisions = [context["decision"] for context in base_contexts]

        pools_by_index: dict[int, list[str]] = {}
        for idx, match in enumerate(ordered_matches):
            if idx in risk_indexes:
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
    ) -> list[list[str]]:
        if mode not in {"16", "32"}:
            raise ValueError("mode must be '16' or '32'")
        if not matches:
            return []
        if not (0.0 <= insurance_strength <= 1.0):
            raise ValueError("insurance_strength must be between 0.0 and 1.0")

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
        insured_matches_dict: dict[int, dict[str, object]] = {}
        strategy_risk_scores: dict[int, float] = {}

        for idx, match in enumerate(ordered_matches):
            probs = self._normalised_probs(match["probs"])
            base_single = self._single_from_decision(decision=base_decisions[idx], probs=probs)
            strategy = self._extract_strategy_signals(match=match, probs=probs)
            profile = self._decision_profile(match)

            if idx in risk_indexes:
                base_pool = self._base_pool(match=match, decision=base_decisions[idx])
                insured_pool = self._insurance_pool(
                    match=match,
                    decision=base_decisions[idx],
                    insurance_strength=insurance_strength,
                    strategy=strategy,
                )
                base_pools_by_index[idx] = base_pool
                insured_pools_by_index[idx] = insured_pool

                alternatives = [outcome for outcome in insured_pool if outcome != base_single]
                if alternatives:
                    risk_score = self._insurance_risk_score(profile=profile, strategy=strategy)
                    insured_matches_dict[idx] = {
                        "base_outcome": base_single,
                        "alternatives": alternatives,
                        "pool_size": len(insured_pool),
                        "risk_score": risk_score,
                    }
                    strategy_risk_scores[idx] = risk_score
            else:
                # At very high insurance strength, expand secondary risk matches if strategy signals present
                if insurance_strength >= 0.88 and strategy["available"]:
                    risk_score = self._insurance_risk_score(profile=profile, strategy=strategy)
                    if risk_score >= 0.60:
                        pool = self._insurance_pool(
                            match=match,
                            decision=base_decisions[idx],
                            insurance_strength=0.5,  # conservative expansion for non-risk matches
                            strategy=strategy,
                        )
                        if len(pool) > 1:
                            base_pools_by_index[idx] = pool
                            insured_pools_by_index[idx] = pool
                            alternatives = [o for o in pool if o != base_single]
                            if alternatives:
                                insured_matches_dict[idx] = {
                                    "base_outcome": base_single,
                                    "alternatives": alternatives,
                                    "pool_size": len(pool),
                                    "risk_score": risk_score,
                                }
                                strategy_risk_scores[idx] = risk_score
                            continue
                single_pool = [base_single]
                base_pools_by_index[idx] = single_pool
                insured_pools_by_index[idx] = single_pool

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
            for candidate in filler:
                key = tuple(candidate)
                if key in existing:
                    continue
                coupons.append(candidate)
                existing.add(key)
                if len(coupons) >= target_lines:
                    break

        if len(coupons) < target_lines and coupons:
            while len(coupons) < target_lines:
                coupons.append(list(coupons[-1]))

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

    def _evaluate_base_decision(self, match: dict) -> dict[str, object]:
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

        return {
            "model_decision": model_decision,
            "decision": adjusted_decision,
            "decision_class": decision_class,
            "history_available": bool(strategy["available"]),
            "history_used": history_used,
            "pool_signals_used": bool("pool_probs_dict" in strategy.get("strategy_signals_used", [])),
            "bk_signals_used": bool(
                "bookmaker_quotes" in strategy.get("strategy_signals_used", [])
                or "top_level_quotes" in strategy.get("strategy_signals_used", [])
            ),
            "strategy_reason": strategy["reason"],
            "signal_source": strategy["signal_source"],
            "strategy_signals_used": list(strategy.get("strategy_signals_used", [])),
            "history_score": float(strategy.get("history_score", 0.0)),
            "favorite_trap_score": float(strategy.get("favorite_trap_score", 0.0)),
            "upset_potential": float(strategy.get("upset_potential", 0.0)),
            "draw_tendency_score": float(strategy.get("draw_tendency_score", 0.0)),
            "public_bias_score": float(strategy.get("public_bias", 0.0)),
            "pool_vs_bookmaker_divergence": float(strategy.get("pool_vs_bookmaker_divergence", 0.0)),
            "history_conflict_with_model": float(strategy.get("history_conflict_with_model", 0.0)),
            "history_adjustment": float(history_adjusted.get("history_adjustment", 0.0)),
            "market_adjustment": float(history_adjusted.get("market_adjustment", 0.0)),
            "history_adjustment_reason": str(history_adjusted.get("reason", "")),
            "history_ready": bool(strategy.get("history_ready", False)),
            "history_state": str(strategy.get("history_state", "history_not_requested")),
            "history_empty_reason": strategy.get("history_empty_reason"),
            "history_draws_loaded_count": int(strategy.get("history_draws_loaded_count", 0)),
            "history_events_loaded_count": int(strategy.get("history_events_loaded_count", 0)),
        }

    def _decision_profile(self, match: dict) -> dict[str, object]:
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

    def _classify_decision_profile(self, profile: dict[str, object]) -> str:
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

    def _extract_strategy_signals(self, match: dict, probs: dict[str, float]) -> dict[str, object]:
        merged: dict[str, float] = {}
        strategy_signals_used: set[str] = set()

        history_stats = self._extract_history_stats(match=match)
        draws_loaded = int(history_stats["draws_count"])
        events_loaded = int(history_stats["events_count"])
        history_state = str(history_stats.get("history_state", "history_not_requested"))
        history_ready = bool(history_stats.get("history_ready", False))
        history_empty_reason = history_stats.get("history_empty_reason")
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
        if history_signals["signals_used"]:
            strategy_signals_used.update(history_signals["signals_used"])

        signal_source = (
            "pool_probs_dict" if "pool_probs_dict" in strategy_signals_used
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

        return {
            "available": has_signals,
            "history_ready": history_ready,
            "history_state": history_state,
            "history_empty_reason": history_empty_reason,
            "public_probs": public,
            "bk_probs": bk_probs,
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
            "reason": reason,
            "signal_source": signal_source,
            "strategy_signals_used": sorted(strategy_signals_used),
            "history_draws_loaded_count": draws_loaded,
            "history_events_loaded_count": events_loaded,
        }

    def _apply_history_strategy(
        self,
        decision: str,
        decision_class: str,
        profile: dict[str, object],
        strategy: dict[str, object],
    ) -> dict[str, object]:
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

        market_adjustment = min(1.0, max(0.0, public_bias + pool_vs_bk_gap + pool_vs_model_bias * 0.5))
        history_adjustment = min(1.0, max(0.0, history_score + history_conflict))

        # ------------------------------------------------------------------
        # SIGNAL 1 (highest priority): Pool vs model divergence.
        # When pool distribution clearly differs from model prediction and the
        # pool's preferred outcome is plausible, form a double covering both.
        # This is the primary mechanism for multi-signal decision making.
        # ------------------------------------------------------------------
        if pool_vs_model_bias >= 0.06 and crowd_favorite != top_outcome and crowd_favorite_competitive:
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
            if (favorite_overload >= 0.06 or upset_tendency >= 0.06) and contrarian != top_outcome and competitive_contrarian:
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
            if draw_underpricing >= 0.06 and probs["PX"] >= float(profile["top_prob"]) - 0.12:
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
        if (pool_vs_bk_gap >= 0.07 or pool_vs_bk_div >= 0.07) and underdog_neglect >= 0.06:
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
        if market_adjustment >= 0.12 and contrarian != top_outcome and competitive_contrarian:
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
        strategy: dict[str, object] | None = None,
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

        strategy_risk_boost = min(
            0.14,
            0.04 * float(strategy.get("favorite_overload", 0.0))
            + 0.04 * float(strategy.get("upset_tendency", 0.0))
            + 0.04 * float(strategy.get("pool_vs_model_bias", 0.0))
            + 0.03 * float(strategy.get("underdog_neglect", 0.0))
            + 0.03 * float(strategy.get("pool_vs_bk_gap", 0.0)),
        )

        # Tier 1 (all strengths >= 0.3): add second outcome if close enough.
        second_margin_limit = 0.16 - (0.10 * insurance_strength)
        second_prob_floor = 0.27 - (0.12 * insurance_strength)
        if second_outcome != single_pick and (
            top_gap_second <= second_margin_limit + strategy_risk_boost
            or second_prob >= second_prob_floor - strategy_risk_boost
        ):
            options.append(second_outcome)

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
                options.append(primary_alt)

        # Tier 3 (>= 0.70): add third ranked outcome.
        if insurance_strength >= 0.70 and third_outcome not in options:
            third_threshold_prob = 0.18 - 0.06 * insurance_strength - strategy_risk_boost
            third_margin_limit = 0.24 - 0.14 * insurance_strength
            if (
                third_prob >= third_threshold_prob
                or top_gap_third <= third_margin_limit + strategy_risk_boost
            ):
                options.append(third_outcome)

        # Tier 4 (>= 0.88): AGGRESSIVE - at very high strength, force all 3 outcomes
        # for matches with high history conflict or pool divergence. The qualitative
        # difference: 0.7 gives 2-3 outcomes for risk matches, 0.9 forces 3 for all
        # high-conflict matches regardless of prob margins.
        if insurance_strength >= 0.88:
            history_conflict = float(strategy.get("history_conflict_with_model", 0.0))
            pool_bias = float(strategy.get("pool_vs_model_bias", 0.0))
            if history_conflict >= 0.12 or pool_bias >= 0.08:
                # Force all 3 outcomes.
                for outcome in ("1", "X", "2"):
                    if outcome not in options:
                        options.append(outcome)

        ordered = sorted(set(options), key=lambda outcome: probs[_PROB_KEY[outcome]], reverse=True)
        # Max outcomes by strength tier:
        if insurance_strength >= 0.88:
            max_options = 3
        elif insurance_strength >= 0.70:
            max_options = 3
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

        while weighted_candidates and len(selected) < target_lines:
            best_index = 0
            best_score = float("-inf")
            search_window = min(len(weighted_candidates), max(target_lines * 12, 64))
            for idx in range(search_window):
                coupon, log_weight = weighted_candidates[idx]
                score = log_weight + self._diversity_bonus(
                    coupon=coupon,
                    per_column_usage=per_column_usage,
                    strength=diversity_strength,
                )
                if score > best_score:
                    best_score = score
                    best_index = idx

            coupon, _ = weighted_candidates.pop(best_index)
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
        insured_matches: dict[int, dict[str, object]],
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
        # 0.7: change up to 2 cells, broader target pool
        # 0.9: change up to 3 cells, deepest coverage
        if insurance_strength >= 0.88:
            max_changes_per_coupon = 3
        elif insurance_strength >= 0.65:
            max_changes_per_coupon = 2
        else:
            max_changes_per_coupon = 1

        influence = 0.25 + 0.70 * insurance_strength

        # Semantic diversification: at high strength, prioritize matches that
        # have pool divergence signals (not just highest entropy).
        # We already have ranked_matches sorted by risk_score which incorporates
        # strategy signals, so the order is already signal-aware.

        insured_coupons: list[list[str]] = []
        changed_matches_by_line: list[list[int]] = []
        seen: set[tuple[str, ...]] = {tuple(c) for c in base_coupons}

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

            for rank, match_idx in enumerate(rotated):
                if len(changed) >= max_changes_per_coupon:
                    break
                if match_idx >= len(modified):
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

            key = tuple(modified)
            if key in seen:
                continue
            seen.add(key)
            insured_coupons.append(modified)
            changed_matches_by_line.append(changed)

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

                pick_idx = (fallback_idx + rank * 2) % len(alternatives)
                alt_outcome = alternatives[pick_idx]
                if alt_outcome == modified[match_idx]:
                    if len(alternatives) > 1:
                        alt_outcome = alternatives[(pick_idx + 1) % len(alternatives)]
                    else:
                        continue

                modified[match_idx] = alt_outcome
                changed.append(match_idx)

            key = tuple(modified)
            if changed and key not in seen:
                seen.add(key)
                insured_coupons.append(modified)
                changed_matches_by_line.append(changed)

            fallback_idx += 1
            if fallback_idx > max(target_count * 12, 64):
                break

        return insured_coupons, changed_matches_by_line

    def _insurance_risk_score(self, profile: dict[str, object], strategy: dict[str, object]) -> float:
        uncertainty = float(profile.get("uncertainty", 0.0))
        margin_top_second = float(profile.get("margin_top_second", 0.0))
        closeness = max(0.0, 1.0 - min(1.0, margin_top_second / 0.25))
        pool_vs_model = float(strategy.get("pool_vs_model_bias", 0.0))
        strategy_risk = (
            0.30 * float(strategy.get("favorite_overload", 0.0))
            + 0.30 * float(strategy.get("upset_tendency", 0.0))
            + 0.15 * float(strategy.get("underdog_neglect", 0.0))
            + 0.15 * float(strategy.get("pool_vs_bk_gap", 0.0))
            + 0.10 * pool_vs_model
        )
        return (0.50 * uncertainty) + (0.25 * closeness) + (0.25 * min(1.0, strategy_risk))

    # ---------------------------------------------------------------------
    # Summary / diagnostics
    # ---------------------------------------------------------------------

    def _build_run_summary(
        self,
        matches: list[dict],
        mode: str,
        insurance_strength: float,
        base_decisions: list[str],
        base_contexts: list[dict[str, object]],
        insured_pools: list[list[str]],
        coupons: list[list[str]],
        generation_mode: str,
    ) -> dict[str, object]:
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

        history_available_count = sum(1 for ctx in base_contexts if bool(ctx.get("history_available")))
        history_used_count = sum(1 for ctx in base_contexts if bool(ctx.get("history_used")))
        pool_signals_used_count = sum(1 for ctx in base_contexts if bool(ctx.get("pool_signals_used")))
        bk_signals_used_count = sum(1 for ctx in base_contexts if bool(ctx.get("bk_signals_used")))

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
        lines_with_1 = sum(1 for coupon in coupons if "1" in coupon)
        lines_with_x = sum(1 for coupon in coupons if "X" in coupon)
        lines_with_2 = sum(1 for coupon in coupons if "2" in coupon)

        pool_fallback_count = sum(
            1
            for m in matches
            if str(m.get("prob_source", "")).lower() in ("pool_context_only", "pool_only")
        )
        weak_model_features_suspected = (n > 0) and (pool_fallback_count > n // 2)

        all_strategy_signals: set[str] = set()
        summary_draws_loaded = 0
        summary_events_loaded = 0
        for ctx in base_contexts:
            for sig in ctx.get("strategy_signals_used", []):
                all_strategy_signals.add(str(sig))
            summary_draws_loaded += int(ctx.get("history_draws_loaded_count", 0))
            summary_events_loaded += int(ctx.get("history_events_loaded_count", 0))

        if history_available_count == 0 and pool_signals_used_count == 0:
            hs_reason = "no_history_signals_in_match_payload"
        elif history_used_count == 0 and pool_signals_used_count == 0:
            hs_reason = "history_signals_present_but_no_decision_changed"
        elif pool_signals_used_count > 0 and history_used_count > 0:
            hs_reason = "history_and_pool_strategy_applied"
        elif pool_signals_used_count > 0:
            hs_reason = "pool_strategy_applied"
        else:
            hs_reason = "history_strategy_applied"

        return {
            "mode": mode,
            "generation_mode": generation_mode,
            "insurance_strength": insurance_strength,
            "stake_affects_decision": False,
            "mode_affects_decision": False,
            "insurance_is_secondary_layer": True,
            "coupon_count": len(coupons),
            "generated_coupon_count": int(uniqueness["generated_coupon_count"]),
            "unique_coupon_count": int(uniqueness["unique_coupon_count"]),
            "duplicate_coupon_count": int(uniqueness["duplicate_coupon_count"]),
            "base_coupons_count": int(uniqueness["generated_coupon_count"]),
            "insured_coupons_count": 0,
            "match_count": n,
            "model_distribution": model_dist,
            "model_decision_distribution": model_decision_dist,
            "base_decision_distribution": base_dist,
            "base_decision_class_distribution": decision_class_distribution,
            "strategy_adjusted_distribution": strategy_adjusted_dist,
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
            "history_strategy": {
                "history_strategy_available": history_available_count > 0 or pool_signals_used_count > 0,
                "history_strategy_used": history_used_count > 0,
                "pool_signals_used_count": pool_signals_used_count,
                "bk_signals_used_count": bk_signals_used_count,
                "history_draws_loaded_count": summary_draws_loaded,
                "history_events_loaded_count": summary_events_loaded,
                "matches_with_history_signals": history_available_count,
                "matches_where_history_changed_decision": history_used_count,
                "history_changed_matches": history_changed_matches,
                "strategy_signal_sources": strategy_signal_sources,
                "all_strategy_signals": sorted(all_strategy_signals),
                "applied_in_manual_mode_when_signals_present": history_used_count > 0 or pool_signals_used_count > 0,
                "reason": hs_reason,
            },
            "match_level_decision_diagnostics": [
                {
                    "match_index": idx + 1,
                    "match": self._match_label(match),
                    "model_decision": model_decisions[idx],
                    "strategy_adjusted_decision": base_decisions[idx],
                    "history_used": bool(ctx.get("history_used", False)),
                    "pool_signals_used": bool(ctx.get("pool_signals_used", False)),
                    "bk_signals_used": bool(ctx.get("bk_signals_used", False)),
                    "history_ready": bool(ctx.get("history_ready", False)),
                    "history_state": str(ctx.get("history_state", "history_not_requested")),
                    "history_adjustment": self._safe_float(ctx.get("history_adjustment")),
                    "market_adjustment": self._safe_float(ctx.get("market_adjustment")),
                    "public_bias_score": self._safe_float(ctx.get("public_bias_score")),
                    "reason": str(ctx.get("history_adjustment_reason", ctx.get("strategy_reason", ""))),
                }
                for idx, (match, ctx) in enumerate(zip(matches, base_contexts))
            ],
            "insurance_target_matches": [],
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
        base_contexts: list[dict[str, object]],
        insured_pools: list[list[str]],
        insured_matches_dict: dict[int, dict[str, object]],
        strategy_risk_scores: dict[int, float],
        coupons: list[list[str]],
        base_coupons_count: int,
        insured_coupons_count: int,
        changed_matches_by_line: list[list[int]],
        generation_mode: str,
    ) -> dict[str, object]:
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
                    affected_coupon_lines_set.add(coupon_line_idx)

            per_match_coupon_changes[match_idx] = coupons_with_alternative

        changed_matches_set: set[int] = set()
        for changed_line in changed_matches_by_line:
            for idx in changed_line:
                changed_matches_set.add(idx)

        affected_coupon_lines_count = len(affected_coupon_lines_set)
        insurance_rebalance_changes = affected_coupon_lines_count
        insured_coupon_indices = list(range(base_coupons_count, base_coupons_count + insured_coupons_count))

        strategy_signals_used: set[str] = set()
        history_draws_loaded_count = 0
        history_events_loaded_count = 0
        history_changed_decisions_count = 0
        pool_signals_used_count = 0
        bk_signals_used_count = 0

        for ctx in base_contexts:
            history_draws_loaded_count += int(ctx.get("history_draws_loaded_count", 0))
            history_events_loaded_count += int(ctx.get("history_events_loaded_count", 0))
            if bool(ctx.get("history_used")):
                history_changed_decisions_count += 1
            if bool(ctx.get("pool_signals_used")):
                pool_signals_used_count += 1
            if bool(ctx.get("bk_signals_used")):
                bk_signals_used_count += 1
            for signal_name in ctx.get("strategy_signals_used", []):
                strategy_signals_used.add(str(signal_name))

        history_changed_insurance_matches_count = sum(1 for idx in changed_matches_set if idx in insured_matches_dict)

        weak_model_features_present = False
        for m in matches:
            if bool(m.get("weak_model_features_present")):
                weak_model_features_present = True
                break

        insurance_diagnostics = {
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
            "insurance_effectiveness_ratio": (
                affected_coupon_lines_count / len(coupons)
                if len(coupons) > 0
                else 0.0
            ),
        }

        base_summary["insurance_diagnostics"] = insurance_diagnostics
        base_summary["base_coupons_count"] = base_coupons_count
        base_summary["insured_coupons_count"] = insured_coupons_count
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
        base_summary["insurance_cells_changed_count"] = insurance_cells_changed_count
        base_summary["affected_coupon_lines_count"] = affected_coupon_lines_count

        # Determine if history was truly available and used.
        history_available = (
            history_draws_loaded_count > 0
            or history_events_loaded_count > 0
            or pool_signals_used_count > 0
        )
        history_used = (
            history_changed_decisions_count > 0
            or history_changed_insurance_matches_count > 0
        )
        if not history_available:
            history_state_label = "history_not_in_match_payload"
        elif history_events_loaded_count > 0:
            history_state_label = "full_history_ready"
        elif history_draws_loaded_count > 0:
            history_state_label = "draws_loaded_no_event_propagation"
        elif pool_signals_used_count > 0:
            history_state_label = "pool_probs_used_as_public_signal"
        else:
            history_state_label = "history_available_not_used"

        base_summary["history_strategy"] = {
            "history_strategy_available": history_available,
            "history_strategy_used": history_used,
            "history_state_label": history_state_label,
            "history_draws_loaded_count": history_draws_loaded_count,
            "history_events_loaded_count": history_events_loaded_count,
            "pool_signals_used_count": pool_signals_used_count,
            "bk_signals_used_count": bk_signals_used_count,
            "history_changed_decisions_count": history_changed_decisions_count,
            "history_changed_matches_count": history_changed_decisions_count,
            "history_changed_insurance_matches_count": history_changed_insurance_matches_count,
            "strategy_signals_used": sorted(strategy_signals_used),
        }

        base_summary["weak_model_features_present"] = weak_model_features_present
        return base_summary

    # ---------------------------------------------------------------------
    # History helpers
    # ---------------------------------------------------------------------

    def _extract_history_stats(self, match: dict) -> dict[str, object]:
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

            # State when draws loaded but no events/stats available.
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
        history_stats: dict[str, object],
        public: dict[str, float] | None,
    ) -> dict[str, object]:
        events_count = int(history_stats.get("events_count", 0))
        history_ready = bool(history_stats.get("history_ready", False))

        if events_count <= 0 and not history_ready:
            return {
                "favorite_overload_signal": 0.0,
                "underdog_neglect_signal": 0.0,
                "pool_vs_bk_gap_signal": 0.0,
                "public_bias_signal": 0.0,
                "upset_history_signal": 0.0,
                "draw_underpricing_signal": 0.0,
                "history_score": 0.0,
                "favorite_trap_score": 0.0,
                "upset_potential": 0.0,
                "draw_tendency_score": 0.0,
                "history_conflict_with_model": 0.0,
                "signals_used": [],
            }

        pool_favorite_win_rate = float(history_stats.get("pool_favorite_win_rate", 0.0))
        upset_rate = float(history_stats.get("upset_rate", 0.0))
        draw_result_rate = float(history_stats.get("draw_result_rate", 0.0))
        avg_pool_vs_bk_gap = float(history_stats.get("avg_pool_vs_bk_gap", 0.0))

        favorite_overload = max(0.0, 0.60 - pool_favorite_win_rate)
        underdog_neglect = max(0.0, upset_rate - 0.28)
        pool_vs_bk_gap = min(1.0, avg_pool_vs_bk_gap)

        public_bias = 0.0
        if public is not None:
            public_bias = max(
                abs(public["P1"] - probs["P1"]),
                abs(public["PX"] - probs["PX"]),
                abs(public["P2"] - probs["P2"]),
            )

        upset_history = max(0.0, upset_rate - 0.25)
        draw_underpricing = max(0.0, draw_result_rate - probs["PX"])
        favorite_trap_score = min(1.0, max(0.0, 0.55 * favorite_overload + 0.45 * underdog_neglect))
        upset_potential = min(1.0, max(0.0, 0.55 * upset_history + 0.45 * underdog_neglect))
        draw_tendency_score = min(1.0, max(0.0, draw_underpricing))
        history_score = min(1.0, max(0.0, 0.35 * favorite_trap_score + 0.35 * upset_potential + 0.30 * draw_tendency_score))
        history_conflict_with_model = min(1.0, max(0.0, public_bias + upset_potential))

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
            "signals_used": [
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
            ],
        }

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------

    def _coupon_uniqueness_stats(self, coupons: list[list[str]]) -> dict[str, int]:
        generated = len(coupons)
        unique = len({tuple(coupon) for coupon in coupons})
        return {
            "generated_coupon_count": generated,
            "unique_coupon_count": unique,
            "duplicate_coupon_count": max(generated - unique, 0),
        }

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

    def _safe_float(self, value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
