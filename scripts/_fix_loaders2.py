"""Surgical fix for load_historical_completed_matches in loaders.py."""
from pathlib import Path

path = Path("ingestion/loaders.py")
text = path.read_text(encoding="utf-8")

# The corrupted section we need to replace (all the broken lines from after the summary dict)
old_section = '''        }
            summary["teams_loaded"] = 0
            summary["league_seasons_loaded"] = 0

        if not target_seasons:
            return summary
        summary["teams_loaded"] = 0
        summary["league_seasons_loaded"] = 0
            season_id = int(season["season_id"])'''

new_section = '''        }
        summary["teams_loaded"] = 0
        summary["league_seasons_loaded"] = 0

        if not target_seasons:
            return summary

        for season in target_seasons:
            season_id = int(season["season_id"])'''

if old_section in text:
    text = text.replace(old_section, new_section, 1)
    path.write_text(text, encoding="utf-8")
    print("FIXED: restored for-loop and corrected summary init")
else:
    print("FAILED: old_section not found")
    # Show the problematic region for debugging
    for i, line in enumerate(text.splitlines(), 1):
        if 140 <= i <= 160:
            print(f"{i:3d}: {repr(line)}")
