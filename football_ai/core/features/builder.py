from __future__ import annotations

from typing import Any

from football_ai.core.features.transformers import calc_form_index, safe_mean


class FeatureBuilder:
    def build(self, match_details: dict[str, Any]) -> dict[str, float]:
        home = match_details.get("home", {})
        away = match_details.get("away", {})

        home_last = home.get("last_5_results", [0, 0, 0, 0, 0])
        away_last = away.get("last_5_results", [0, 0, 0, 0, 0])

        home_winrate = safe_mean(home_last)
        away_winrate = safe_mean(away_last)

        xg_home = float(match_details.get("xg_home", 0.0))
        xg_away = float(match_details.get("xg_away", 0.0))
        shots_home = float(match_details.get("shots_home", 0.0))
        shots_away = float(match_details.get("shots_away", 0.0))
        odds_home = float(match_details.get("odds_home", 0.0))
        odds_draw = float(match_details.get("odds_draw", 0.0))
        odds_away = float(match_details.get("odds_away", 0.0))

        return {
            "home_winrate": home_winrate,
            "away_winrate": away_winrate,
            "goals_scored_home": float(home.get("goals_scored", 0.0)),
            "goals_conceded_home": float(home.get("goals_conceded", 0.0)),
            "goals_scored_away": float(away.get("goals_scored", 0.0)),
            "goals_conceded_away": float(away.get("goals_conceded", 0.0)),
            "home_form": home_winrate,
            "away_form": away_winrate,
            "xg_diff": xg_home - xg_away,
            "shots_diff": shots_home - shots_away,
            "form_index_home": calc_form_index(home_winrate, float(home.get("goals_scored", 0.0)), float(home.get("goals_conceded", 0.0))),
            "form_index_away": calc_form_index(away_winrate, float(away.get("goals_scored", 0.0)), float(away.get("goals_conceded", 0.0))),
            "odds_home": odds_home,
            "odds_draw": odds_draw,
            "odds_away": odds_away,
            "odds_diff": odds_home - odds_away,
        }
