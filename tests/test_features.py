import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import math

from core.features.builder import FeatureBuilder


def test_build_features_contains_required_groups() -> None:
    match = {"odds_ft_1": 2.0, "odds_ft_x": 3.4, "odds_ft_2": 3.8}
    home = {
        "ppg": 2.1,
        "goals": 1.9,
        "xg": 1.8,
        "shots": 14.0,
        "possession": 58.0,
        "home_ppg": 2.3,
    }
    away = {
        "ppg": 1.2,
        "goals": 1.1,
        "xg": 1.0,
        "shots": 8.0,
        "possession": 45.0,
        "away_ppg": 1.0,
    }
    league = {"draw_pct": 0.26, "home_advantage": 0.41, "avg_goals": 2.7}

    features = FeatureBuilder.build_features(match, home, away, league)

    expected_keys = {
        "odds_ft_1",
        "odds_ft_x",
        "odds_ft_2",
        "implied_prob_1",
        "implied_prob_x",
        "implied_prob_2",
        "ppg_diff",
        "goals_diff",
        "xg_diff",
        "shots_diff",
        "possession_diff",
        "home_home_ppg",
        "away_away_ppg",
        "split_advantage",
        "draw_pct",
        "home_advantage",
        "avg_goals",
        "entropy",
        "gap",
        "volatility",
    }
    assert expected_keys.issubset(features.keys())
    assert all(isinstance(v, float) for v in features.values())
    assert math.isclose(
        features["implied_prob_1"] + features["implied_prob_x"] + features["implied_prob_2"],
        1.0,
        rel_tol=1e-6,
    )


def test_build_features_sanitizes_invalid_values() -> None:
    match = {"odds_ft_1": -1, "odds_ft_x": -2, "odds_ft_2": "NaN"}
    home = {"ppg": -1, "goals": None, "xg": "bad", "shots": float("nan"), "possession": -2, "home_ppg": -2}
    away = {"ppg": None, "goals": -1, "xg": -2, "shots": "oops", "possession": None, "away_ppg": -1}
    league = {"draw_pct": -1, "home_advantage": None, "avg_goals": -2}

    features = FeatureBuilder.build_features(match, home, away, league)

    assert features["odds_ft_1"] == 0.0
    assert features["odds_ft_x"] == 0.0
    assert features["odds_ft_2"] == 0.0
    assert features["ppg_diff"] == 0.0
    assert features["draw_pct"] == 0.0
    assert features["split_advantage"] == 0.0
    assert all(not math.isnan(v) for v in features.values())


def test_helpers_behave_as_expected() -> None:
    assert FeatureBuilder.safe_div(10, 2) == 5.0
    assert FeatureBuilder.safe_div(10, 0) == 0.0
    assert FeatureBuilder.calc_diff(3, 8) == -5.0

    normalized = FeatureBuilder.normalize_odds(2.0, 4.0, 4.0)
    assert math.isclose(normalized["implied_prob_1"], 0.5, rel_tol=1e-6)

    entropy = FeatureBuilder.calc_entropy(1 / 3, 1 / 3, 1 / 3)
    assert 0.99 <= entropy <= 1.0
