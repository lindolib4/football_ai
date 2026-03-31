import sqlite3
c = sqlite3.connect("database/footai.sqlite3")
r1 = c.execute("SELECT COUNT(*) FROM team_season_stats").fetchone()[0]
r2 = c.execute("SELECT COUNT(*) FROM league_season_stats").fetchone()[0]
r3 = c.execute("SELECT COUNT(*) FROM matches WHERE status='completed'").fetchone()[0]
print(f"team_season_stats={r1}")
print(f"league_season_stats={r2}")
print(f"completed_matches={r3}")
