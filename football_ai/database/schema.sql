CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    date TEXT,
    league TEXT,
    home_team TEXT,
    away_team TEXT,
    home_score INTEGER,
    away_score INTEGER,
    status TEXT
);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id TEXT PRIMARY KEY,
    xg_home REAL,
    xg_away REAL,
    shots_home INTEGER,
    shots_away INTEGER,
    corners INTEGER,
    possession REAL,
    odds_home REAL,
    odds_draw REAL,
    odds_away REAL,
    FOREIGN KEY(match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS features (
    match_id TEXT PRIMARY KEY,
    features_json TEXT,
    FOREIGN KEY(match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    match_id TEXT PRIMARY KEY,
    p1 REAL,
    px REAL,
    p2 REAL,
    decision TEXT,
    confidence REAL,
    FOREIGN KEY(match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS results (
    match_id TEXT PRIMARY KEY,
    actual_result TEXT,
    correct INTEGER,
    FOREIGN KEY(match_id) REFERENCES matches(match_id)
);
