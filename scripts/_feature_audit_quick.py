import sqlite3
from pathlib import Path

DB = str(Path(__file__).parents[1] / "database" / "footai.sqlite3")
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

def q(sql):
    return conn.execute(sql).fetchone()[0]

print("=== DB COVERAGE ===")
print("team_season_stats rows:     ", q("SELECT COUNT(*) FROM team_season_stats"))
print("league_season_stats rows:   ", q("SELECT COUNT(*) FROM league_season_stats"))
print("matches total:              ", q("SELECT COUNT(*) FROM matches"))
print("matches completed:          ", q("SELECT COUNT(*) FROM matches WHERE status='completed'"))
print("matches with season_id:     ", q("SELECT COUNT(*) FROM matches WHERE season_id IS NOT NULL"))
print("matches home_ppg non-null:  ", q("SELECT COUNT(*) FROM matches WHERE home_ppg IS NOT NULL AND home_ppg != 0"))
print("matches pre_match_home_ppg: ", q("SELECT COUNT(*) FROM matches WHERE pre_match_home_ppg IS NOT NULL AND pre_match_home_ppg != 0"))

print("\n=== team_season_stats fields ===")
for col in ["goals_for_avg_overall","goals_for_avg_home","shots_avg_overall","shots_avg_home","possession_avg_overall","possession_avg_home","xg_for_avg_overall","season_ppg_overall"]:
    cnt = q("SELECT COUNT(*) FROM team_season_stats WHERE " + col + " IS NOT NULL AND " + col + " != 0")
    print("  {:<32}  non-null: {}".format(col, cnt))

print("\n=== league_season_stats fields ===")
for col in ["draw_pct","home_advantage","season_avg_goals","home_win_pct","away_win_pct"]:
    cnt = q("SELECT COUNT(*) FROM league_season_stats WHERE " + col + " IS NOT NULL AND " + col + " != 0")
    print("  {:<32}  non-null: {}".format(col, cnt))

print("\n=== TRAINING JOIN QUALITY ===")
jq = ("SELECT COUNT(*) total, SUM(CASE WHEN htss.team_id IS NOT NULL THEN 1 ELSE 0 END) home_join_ok, SUM(CASE WHEN atss.team_id IS NOT NULL THEN 1 ELSE 0 END) away_join_ok, SUM(CASE WHEN lss.season_id IS NOT NULL THEN 1 ELSE 0 END) league_join_ok FROM matches m LEFT JOIN team_season_stats htss ON htss.team_id = m.home_team_id AND htss.season_id = m.season_id LEFT JOIN team_season_stats atss ON atss.team_id = m.away_team_id AND atss.season_id = m.season_id LEFT JOIN league_season_stats lss ON lss.season_id = m.season_id WHERE m.status = 'completed'")
r = conn.execute(jq).fetchone()
total = r["total"] or 1
print("  Completed matches:         ", r["total"])
print("  home team_season_stats hit:", r["home_join_ok"], "(" + str(100 * r["home_join_ok"] // total) + "%)")
print("  away team_season_stats hit:", r["away_join_ok"], "(" + str(100 * r["away_join_ok"] // total) + "%)")
print("  league_season_stats hit:   ", r["league_join_ok"], "(" + str(100 * r["league_join_ok"] // total) + "%)")

print("\n=== LIVE match sample (most recent) ===")
cols = "home_team_id, away_team_id, season_id, home_ppg, away_ppg, pre_match_home_ppg, pre_match_away_ppg, avg_potential"
rows = conn.execute("SELECT " + cols + " FROM matches ORDER BY date_unix DESC LIMIT 3").fetchall()
for row in rows:
    print(" ", dict(row))

conn.close()
print("\nDone.")
