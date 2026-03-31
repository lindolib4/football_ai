"""One-shot fix: correct the indentation of the team/league stats loading block in loaders.py."""
from pathlib import Path

path = Path("ingestion/loaders.py")
text = path.read_text(encoding="utf-8")

old_block = (
    "\n"
    "                # Load team season stats (shotsAVG, possessionAVG, scoredAVG, xg etc.) for this season\n"
    "                # These are required for goals_diff, shots_diff, possession_diff, xg_diff features.\n"
    "                # Without this, all team-stats-based features fall back to 0.0.\n"
    "                teams_loaded = self.load_league_teams(season_id=season_id, max_time=max_time)\n"
    "                summary[\"teams_loaded\"] += teams_loaded\n"
    "                summary[\"season_summaries\"][-1][\"teams_loaded\"] = teams_loaded\n"
    "\n"
    "                # Load league season stats (draw_pct, home_advantage, avg_goals etc.)\n"
    "                league_loaded = self.load_league_season(season_id=season_id, max_time=max_time)\n"
    "                summary[\"league_seasons_loaded\"] += int(league_loaded)\n"
    "                summary[\"season_summaries\"][-1][\"league_season_loaded\"] = league_loaded\n"
)

new_block = (
    "\n"
    "            # Load team season stats (shotsAVG, possessionAVG, scoredAVG, xg etc.) for this season.\n"
    "            # Required for goals_diff, shots_diff, possession_diff, xg_diff — without this call\n"
    "            # all team-stats-based features fall back to 0.0.\n"
    "            teams_loaded = self.load_league_teams(season_id=season_id, max_time=max_time)\n"
    "            summary[\"teams_loaded\"] += teams_loaded\n"
    "            summary[\"season_summaries\"][-1][\"teams_loaded\"] = teams_loaded\n"
    "\n"
    "            # Load league season stats (draw_pct, home_advantage, avg_goals etc.)\n"
    "            league_loaded = self.load_league_season(season_id=season_id, max_time=max_time)\n"
    "            summary[\"league_seasons_loaded\"] += int(league_loaded)\n"
    "            summary[\"season_summaries\"][-1][\"league_season_loaded\"] = league_loaded\n"
)

if old_block in text:
    text = text.replace(old_block, new_block, 1)
    path.write_text(text, encoding="utf-8")
    print("FIXED: indentation corrected")
else:
    # Show what's actually around the area
    lines = text.splitlines()
    for i, line in enumerate(lines, 1):
        if "Load team season stats" in line or "Load league season stats" in line:
            print(f"Line {i}: repr={repr(line)}")
    print("FAILED: old_block not found as exact string")
