from database.db import Database
db = Database()
c = db.conn

total_tss = c.execute("SELECT COUNT(*) FROM team_season_stats").fetchone()[0]
print("team_season_stats rows:", total_tss)

total_lss = c.execute("SELECT COUNT(*) FROM league_season_stats").fetchone()[0]
print("league_season_stats rows:", total_lss)

total_m = c.execute("SELECT COUNT(*) FROM matches WHERE status = 'completed'").fetchone()[0]
print("completed matches:", total_m)

if total_tss > 0:
    row = c.execute("SELECT team_id, season_id, goals_for_avg_overall, shots_avg_overall FROM team_season_stats LIMIT 1").fetchone()
    print("sample tss row:", dict(row))
    match_seasons = [r[0] for r in c.execute("SELECT DISTINCT season_id FROM matches LIMIT 5").fetchall()]
    stat_seasons = [r[0] for r in c.execute("SELECT DISTINCT season_id FROM team_season_stats LIMIT 5").fetchall()]
    print("match seasons sample:", match_seasons)
    print("stat  seasons sample:", stat_seasons)
    matched = c.execute("SELECT COUNT(*) FROM matches m JOIN team_season_stats htss ON htss.team_id = m.home_team_id AND htss.season_id = m.season_id WHERE m.status = 'completed'").fetchone()[0]
    print("home team matched:", matched)
    non_null_goals = c.execute("SELECT COUNT(*) FROM team_season_stats WHERE goals_for_avg_overall IS NOT NULL").fetchone()[0]
    print("tss rows with non-null goals_for_avg_overall:", non_null_goals)
else:
    print("team_season_stats is EMPTY")

if total_lss > 0:
    non_null_draw = c.execute("SELECT COUNT(*) FROM league_season_stats WHERE draw_pct IS NOT NULL").fetchone()[0]
    print("lss rows with draw_pct:", non_null_draw)
else:
    print("league_season_stats is EMPTY")

unique_seasons = [r[0] for r in c.execute("SELECT DISTINCT season_id FROM matches LIMIT 10").fetchall()]
print("Unique season_ids in matches (sample):", unique_seasons)
