"""Quick DB diagnostic for status, season_id join coverage and winning_team_id."""
import sys
sys.path.insert(0, ".")
from database.db import Database

db = Database()
c = db.conn

r_total = c.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
r_null_sid = c.execute("SELECT COUNT(*) FROM matches WHERE season_id IS NULL").fetchone()[0]
r_completed = c.execute("SELECT COUNT(*) FROM matches WHERE status = 'completed'").fetchone()[0]
r_completed_with_sid = c.execute("SELECT COUNT(*) FROM matches WHERE status = 'completed' AND season_id IS NOT NULL").fetchone()[0]
r_team_stats = c.execute("SELECT COUNT(*) FROM team_season_stats").fetchone()[0]
r_league_stats = c.execute("SELECT COUNT(*) FROM league_season_stats").fetchone()[0]

r_joinable_team = c.execute("""
    SELECT COUNT(DISTINCT m.match_id)
    FROM matches m
    JOIN team_season_stats htss ON htss.team_id = m.home_team_id AND htss.season_id = m.season_id
    WHERE m.status = 'completed'
""").fetchone()[0]

r_joinable_league = c.execute("""
    SELECT COUNT(DISTINCT m.match_id)
    FROM matches m
    JOIN league_season_stats lss ON lss.season_id = m.season_id
    WHERE m.status = 'completed'
""").fetchone()[0]

r_no_winner_no_goals = c.execute("""
    SELECT COUNT(*) FROM matches
    WHERE status = 'completed' AND winning_team_id IS NULL AND home_goals IS NULL
""").fetchone()[0]

r_no_winner_has_goals = c.execute("""
    SELECT COUNT(*) FROM matches
    WHERE status = 'completed' AND winning_team_id IS NULL AND home_goals IS NOT NULL
""").fetchone()[0]

r_draws_detected = c.execute("""
    SELECT COUNT(*) FROM matches
    WHERE status = 'completed' AND home_goals IS NOT NULL AND away_goals IS NOT NULL
    AND home_goals = away_goals
""").fetchone()[0]

# Build training dataset and check source stats
dataset = db.build_training_dataset_from_db()
debug = db.last_training_dataset_debug
wfc = debug.get("weak_feature_coverage", {})

print("=== DB STATUS ===")
print(f"total_matches={r_total}")
print(f"null_season_id={r_null_sid}")
print(f"completed_matches={r_completed}")
print(f"completed_with_season_id={r_completed_with_sid}")
print(f"team_season_stats_rows={r_team_stats}")
print(f"league_season_stats_rows={r_league_stats}")
print(f"completed_joinable_to_team_stats={r_joinable_team} ({round(100*r_joinable_team/r_completed,1) if r_completed else 0}%)")
print(f"completed_joinable_to_league_stats={r_joinable_league} ({round(100*r_joinable_league/r_completed,1) if r_completed else 0}%)")
print(f"completed_no_winner_no_goals={r_no_winner_no_goals}")
print(f"completed_no_winner_has_goals={r_no_winner_has_goals} (draws where winning_team_id=NULL)")
print(f"completed_draws_by_goals={r_draws_detected}")

print()
print("=== TRAINING DATASET ===")
print(f"raw_completed_rows={debug.get('raw_completed_rows')}")
print(f"unique_completed_matches={debug.get('unique_completed_matches')}")
print(f"rows_after_feature_build={debug.get('rows_after_feature_build')}")
print(f"rows_after_dedup={debug.get('rows_after_dedup')}")
print(f"skipped_due_to_invalid_odds={debug.get('skipped_due_to_invalid_odds')}")

print()
print("=== FEATURE COVERAGE ===")
for feat, info in wfc.items():
    print(f"  {feat}: fill={info.get('fill_rate',0)*100:.1f}%  zero={info.get('zero_rate',0)*100:.1f}%  fallback={info.get('fallback_rate',0)*100:.1f}%  status={info.get('source_status')}  breakdown={info.get('source_breakdown')}")

print()
audit = db.audit_match_duplicates(sample_limit=5)
print("=== DUPLICATE AUDIT ===")
print(f"total={audit['total_matches']}")
print(f"dup_match_id={audit['duplicate_match_id_groups']}")
print(f"dup_home_away_date={audit['duplicate_home_away_date_groups']}")
print(f"dup_season_home_away_date={audit['duplicate_season_home_away_date_groups']}")
print(f"mixed_status_identity_groups={audit['mixed_status_identity_groups']}")
if audit["samples"]["mixed_status_identity"]:
    print("  sample mixed-status rows:")
    for s in audit["samples"]["mixed_status_identity"][:3]:
        print(f"    {s}")
