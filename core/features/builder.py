from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Build prematch feature vectors without postmatch leakage."""

    INVALID_SENTINELS = {-1, -2, -1.0, -2.0}

    @classmethod
    def safe_div(cls, numerator: Any, denominator: Any, default: float = 0.0) -> float:
        num = cls._to_float(numerator, default=default)
        den = cls._to_float(denominator, default=0.0)
        if den == 0.0:
            return float(default)
        return float(num / den)

    @classmethod
    def calc_diff(cls, left: Any, right: Any, default: float = 0.0) -> float:
        return float(cls._to_float(left, default=default) - cls._to_float(right, default=default))

    @classmethod
    def normalize_odds(cls, odds_1: Any, odds_x: Any, odds_2: Any) -> dict[str, float]:
        o1 = cls._to_float(odds_1, default=0.0)
        ox = cls._to_float(odds_x, default=0.0)
        o2 = cls._to_float(odds_2, default=0.0)

        inv_1 = cls.safe_div(1.0, o1)
        inv_x = cls.safe_div(1.0, ox)
        inv_2 = cls.safe_div(1.0, o2)

        total = inv_1 + inv_x + inv_2
        if total == 0.0:
            logger.warning("Failed to normalize odds because all values are invalid/zero: %s, %s, %s", o1, ox, o2)
            return {"implied_prob_1": 1.0 / 3.0, "implied_prob_x": 1.0 / 3.0, "implied_prob_2": 1.0 / 3.0}

        return {
            "implied_prob_1": float(inv_1 / total),
            "implied_prob_x": float(inv_x / total),
            "implied_prob_2": float(inv_2 / total),
        }

    @staticmethod
    def calc_entropy(p1: Any, px: Any, p2: Any) -> float:
        probs = [float(p1), float(px), float(p2)]
        eps = 1e-12
        entropy = -sum(p * math.log(max(p, eps), 2) for p in probs if p > 0.0)
        max_entropy = math.log(3, 2)
        if max_entropy == 0.0:
            return 0.0
        return float(entropy / max_entropy)

    @classmethod
    def build_features(
        cls,
        match: dict[str, Any] | None,
        home_team_stats: dict[str, Any] | None,
        away_team_stats: dict[str, Any] | None,
        league_stats: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            match = match or {}
            home_team_stats = home_team_stats or {}
            away_team_stats = away_team_stats or {}
            league_stats = league_stats or {}

            odds_1 = cls._pick(match, "odds_ft_1", "odds_home", "home_odds")
            odds_x = cls._pick(match, "odds_ft_x", "odds_draw", "draw_odds")
            odds_2 = cls._pick(match, "odds_ft_2", "odds_away", "away_odds")

            implied = cls.normalize_odds(odds_1, odds_x, odds_2)
            probs = [implied["implied_prob_1"], implied["implied_prob_x"], implied["implied_prob_2"]]
            probs_sorted = sorted(probs, reverse=True)

            # --- Stats-based features with source tracking (no sentinel/None → 0.0 confusion) ---
            source_meta: dict[str, str] = {}

            # goals_diff: avg scored per game (prefer context-aware key names from team_season_stats)
            home_goals_raw = cls._pick(
                home_team_stats,
                "goals_for_avg_home", "goals_for_avg_overall",
                "goals", "goals_scored", "avg_goals",
            )
            away_goals_raw = cls._pick(
                away_team_stats,
                "goals_for_avg_away", "goals_for_avg_overall",
                "goals", "goals_scored", "avg_goals",
            )
            g_h, g_src_h = cls._to_float_with_source(home_goals_raw)
            g_a, g_src_a = cls._to_float_with_source(away_goals_raw)
            goals_diff = g_h - g_a
            source_meta["__source_goals_diff"] = cls._combined_source(g_src_h, g_src_a)

            # xg_diff: expected goals per game (often absent — stays fallback_default when NULL)
            home_xg_raw = cls._pick(
                home_team_stats,
                "xg_for_avg_home", "xg_for_avg_overall",
                "xg", "avg_xg",
            )
            away_xg_raw = cls._pick(
                away_team_stats,
                "xg_for_avg_away", "xg_for_avg_overall",
                "xg", "avg_xg",
            )
            xg_h, xg_src_h = cls._to_float_with_source(home_xg_raw)
            xg_a, xg_src_a = cls._to_float_with_source(away_xg_raw)
            xg_diff = xg_h - xg_a
            source_meta["__source_xg_diff"] = cls._combined_source(xg_src_h, xg_src_a)

            # shots_diff: prefer home-context for home team, away-context for away team
            home_shots_raw = cls._pick(
                home_team_stats,
                "shots_avg_home", "shots_avg_overall",
                "shots", "avg_shots",
            )
            away_shots_raw = cls._pick(
                away_team_stats,
                "shots_avg_away", "shots_avg_overall",
                "shots", "avg_shots",
            )
            sh_h, sh_src_h = cls._to_float_with_source(home_shots_raw)
            sh_a, sh_src_a = cls._to_float_with_source(away_shots_raw)
            shots_diff = sh_h - sh_a
            source_meta["__source_shots_diff"] = cls._combined_source(sh_src_h, sh_src_a)

            # possession_diff: prefer context-specific averages
            home_poss_raw = cls._pick(
                home_team_stats,
                "possession_avg_home", "possession_avg_overall",
                "possession", "avg_possession",
            )
            away_poss_raw = cls._pick(
                away_team_stats,
                "possession_avg_away", "possession_avg_overall",
                "possession", "avg_possession",
            )
            po_h, po_src_h = cls._to_float_with_source(home_poss_raw)
            po_a, po_src_a = cls._to_float_with_source(away_poss_raw)
            possession_diff = po_h - po_a
            source_meta["__source_possession_diff"] = cls._combined_source(po_src_h, po_src_a)

            features: dict[str, float] = {
                # 1) MATCH LEVEL
                "odds_ft_1": cls._to_float(odds_1),
                "odds_ft_x": cls._to_float(odds_x),
                "odds_ft_2": cls._to_float(odds_2),
                "implied_prob_1": implied["implied_prob_1"],
                "implied_prob_x": implied["implied_prob_x"],
                "implied_prob_2": implied["implied_prob_2"],
                # 2) TEAM DIFF
                "ppg_diff": cls.calc_diff(cls._pick(home_team_stats, "ppg"), cls._pick(away_team_stats, "ppg")),
                "goals_diff": goals_diff,
                "xg_diff": xg_diff,
                "shots_diff": shots_diff,
                "possession_diff": possession_diff,
                # 3) HOME/AWAY
                "home_home_ppg": cls._to_float(cls._pick(home_team_stats, "home_ppg", "ppg_home", "ppg")),
                "away_away_ppg": cls._to_float(cls._pick(away_team_stats, "away_ppg", "ppg_away", "ppg")),
                # 4) LEAGUE
                "draw_pct": cls._to_float(cls._pick(league_stats, "draw_pct", "draw_rate")),
                "home_advantage": cls._to_float(cls._pick(league_stats, "home_advantage", "home_win_rate")),
                "avg_goals": cls._to_float(cls._pick(league_stats, "avg_goals", "goals_per_match")),
                # 5) RISK
                "entropy": cls.calc_entropy(*probs),
                "gap": float(probs_sorted[0] - probs_sorted[1]),
                "volatility": cls._std_dev(probs),
            }

            features["split_advantage"] = cls.calc_diff(features["home_home_ppg"], features["away_away_ppg"])
            # Convert all numeric features; then merge source metadata strings separately.
            result: dict[str, Any] = {k: cls._to_float(v) for k, v in features.items()}
            result.update(source_meta)
            return result
        except Exception:
            logger.exception("Failed to build feature vector; returning safe defaults")
            return cls._default_vector()

    def build(self, match_details: dict[str, Any]) -> dict[str, float]:
        """Backward-compatible adapter to the previous single-argument API."""
        home_stats = match_details.get("home", {}) if isinstance(match_details, dict) else {}
        away_stats = match_details.get("away", {}) if isinstance(match_details, dict) else {}
        league_stats = match_details.get("league", {}) if isinstance(match_details, dict) else {}
        return self.build_features(match_details, home_stats, away_stats, league_stats)

    @classmethod
    def _to_float(cls, value: Any, default: float = 0.0) -> float:
        if value is None:
            return float(default)

        if value in cls.INVALID_SENTINELS:
            return float(default)

        try:
            num = float(value)
        except (TypeError, ValueError):
            logger.error("Invalid numeric value for feature conversion: %r", value)
            return float(default)

        if math.isnan(num) or math.isinf(num):
            return float(default)
        return float(num)

    @classmethod
    def _to_float_with_source(cls, value: Any, default: float = 0.0) -> tuple[float, str]:
        """Like _to_float() but also returns a source classification string.

        Returns (float_value, source) where source is one of:
        - ``"real"``              – a valid, non-sentinel numeric value.
        - ``"missing_sentinel"``  – value was ``None`` or a known sentinel (-1, -2).
        - ``"fallback_default"``  – value was present but non-parseable.
        """
        if value is None:
            return (float(default), "missing_sentinel")

        if value in cls.INVALID_SENTINELS:
            return (float(default), "missing_sentinel")

        try:
            num = float(value)
        except (TypeError, ValueError):
            return (float(default), "fallback_default")

        if math.isnan(num) or math.isinf(num):
            return (float(default), "missing_sentinel")

        return (float(num), "real")

    @staticmethod
    def _combined_source(src_a: str, src_b: str) -> str:
        """Combine two source labels for a diff feature.

        Priority: real > missing_sentinel > fallback_default.
        Both sides must be 'real' for the diff to be 'real'.
        """
        if src_a == "real" and src_b == "real":
            return "real"
        if src_a == "missing_sentinel" or src_b == "missing_sentinel":
            return "missing_sentinel"
        return "fallback_default"

    @staticmethod
    def _pick(source: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in source:
                return source.get(key)
        return None

    @classmethod
    def _std_dev(cls, values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return cls._to_float(math.sqrt(var))

    @classmethod
    def _default_vector(cls) -> dict[str, float]:
        base = cls.normalize_odds(0, 0, 0)
        return {
            "odds_ft_1": 0.0,
            "odds_ft_x": 0.0,
            "odds_ft_2": 0.0,
            "implied_prob_1": base["implied_prob_1"],
            "implied_prob_x": base["implied_prob_x"],
            "implied_prob_2": base["implied_prob_2"],
            "ppg_diff": 0.0,
            "goals_diff": 0.0,
            "xg_diff": 0.0,
            "shots_diff": 0.0,
            "possession_diff": 0.0,
            "home_home_ppg": 0.0,
            "away_away_ppg": 0.0,
            "split_advantage": 0.0,
            "draw_pct": 0.0,
            "home_advantage": 0.0,
            "avg_goals": 0.0,
            "entropy": cls.calc_entropy(base["implied_prob_1"], base["implied_prob_x"], base["implied_prob_2"]),
            "gap": 0.0,
            "volatility": 0.0,
        }
