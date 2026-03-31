PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS countries (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS leagues (
    season_id INTEGER PRIMARY KEY,
    league_name TEXT,
    country_name TEXT,
    season_label TEXT,
    chosen_flag INTEGER DEFAULT 1,
    raw_json TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS league_season_stats (
    season_id INTEGER PRIMARY KEY,
    progress REAL,
    total_matches INTEGER,
    matches_completed INTEGER,
    home_win_pct REAL,
    draw_pct REAL,
    away_win_pct REAL,
    btts_pct REAL,
    season_avg_goals REAL,
    home_advantage REAL,
    corners_avg REAL,
    cards_avg REAL,
    raw_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS teams (
    team_id INTEGER,
    season_id INTEGER,
    name TEXT,
    clean_name TEXT,
    country TEXT,
    table_position INTEGER,
    performance_rank REAL,
    prediction_risk REAL,
    raw_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_id, season_id)
);

CREATE TABLE IF NOT EXISTS team_season_stats (
    team_id INTEGER,
    season_id INTEGER,
    season_ppg_overall REAL,
    season_ppg_home REAL,
    season_ppg_away REAL,
    win_pct_overall REAL,
    win_pct_home REAL,
    win_pct_away REAL,
    draw_pct_overall REAL,
    draw_pct_home REAL,
    draw_pct_away REAL,
    lose_pct_overall REAL,
    lose_pct_home REAL,
    lose_pct_away REAL,
    goals_for_avg_overall REAL,
    goals_for_avg_home REAL,
    goals_for_avg_away REAL,
    goals_against_avg_overall REAL,
    goals_against_avg_home REAL,
    goals_against_avg_away REAL,
    btts_pct_overall REAL,
    btts_pct_home REAL,
    btts_pct_away REAL,
    over25_pct_overall REAL,
    over25_pct_home REAL,
    over25_pct_away REAL,
    corners_avg_overall REAL,
    corners_avg_home REAL,
    corners_avg_away REAL,
    cards_avg_overall REAL,
    cards_avg_home REAL,
    cards_avg_away REAL,
    shots_avg_overall REAL,
    shots_avg_home REAL,
    shots_avg_away REAL,
    shots_on_target_avg_overall REAL,
    shots_on_target_avg_home REAL,
    shots_on_target_avg_away REAL,
    possession_avg_overall REAL,
    possession_avg_home REAL,
    possession_avg_away REAL,
    xg_for_avg_overall REAL,
    xg_for_avg_home REAL,
    xg_for_avg_away REAL,
    xg_against_avg_overall REAL,
    xg_against_avg_home REAL,
    xg_against_avg_away REAL,
    raw_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (team_id, season_id)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id INTEGER PRIMARY KEY,
    season_id INTEGER,
    date_unix INTEGER,
    match_date_iso TEXT,
    status TEXT,
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_team_name TEXT,
    away_team_name TEXT,
    home_goals INTEGER,
    away_goals INTEGER,
    winning_team_id INTEGER,
    odds_ft_1 REAL,
    odds_ft_x REAL,
    odds_ft_2 REAL,
    btts_potential REAL,
    o15_potential REAL,
    o25_potential REAL,
    o35_potential REAL,
    o45_potential REAL,
    corners_potential REAL,
    cards_potential REAL,
    avg_potential REAL,
    home_ppg REAL,
    away_ppg REAL,
    pre_match_home_ppg REAL,
    pre_match_away_ppg REAL,
    raw_json TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id INTEGER PRIMARY KEY,
    shots_home REAL,
    shots_away REAL,
    shots_on_target_home REAL,
    shots_on_target_away REAL,
    possession_home REAL,
    possession_away REAL,
    corners_home REAL,
    corners_away REAL,
    fouls_home REAL,
    fouls_away REAL,
    yellow_home REAL,
    yellow_away REAL,
    red_home REAL,
    red_away REAL,
    over25_flag INTEGER,
    btts_flag INTEGER,
    ht_goals_home INTEGER,
    ht_goals_away INTEGER,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE IF NOT EXISTS features (
    match_id INTEGER PRIMARY KEY,
    feature_version TEXT,
    features_json TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS training_rows (
    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    season_id INTEGER,
    feature_version TEXT,
    target_1x2 INTEGER,
    is_trainable INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    model_version TEXT,
    p1 REAL,
    px REAL,
    p2 REAL,
    entropy REAL,
    confidence REAL,
    decision TEXT,
    decision_reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS toto_draws (
    draw_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS toto_draw_matches (
    draw_id INTEGER,
    position_1_to_15 INTEGER,
    match_id INTEGER,
    selected_flag INTEGER,
    PRIMARY KEY (draw_id, position_1_to_15)
);

CREATE TABLE IF NOT EXISTS toto_coupons (
    coupon_id INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_id INTEGER,
    strategy_name TEXT,
    coupon_index INTEGER,
    line_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS results_audit (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id INTEGER,
    predicted_decision TEXT,
    actual_result TEXT,
    correct_flag INTEGER,
    error_type TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_prediction_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    predicted_at TEXT NOT NULL,
    model_version TEXT,
    model_fingerprint TEXT,
    model_mtime TEXT,
    match_id INTEGER NOT NULL,
    match_date_iso TEXT,
    home_team TEXT,
    away_team TEXT,
    season_id INTEGER,
    league_id INTEGER,
    p1 REAL NOT NULL,
    px REAL NOT NULL,
    p2 REAL NOT NULL,
    final_predicted_outcome TEXT NOT NULL,
    confidence REAL,
    calibrated_confidence REAL,
    feature_context_level TEXT,
    signal_strength TEXT,
    market_disagreement_flag INTEGER DEFAULT 0,
    weak_favorite_flag INTEGER DEFAULT 0,
    draw_risk_flag INTEGER DEFAULT 0,
    stats_override_signal_flag INTEGER DEFAULT 0,
    no_odds_mode INTEGER DEFAULT 0,
    prediction_source TEXT,
    prediction_status TEXT,
    actual_outcome TEXT,
    is_correct INTEGER,
    resolved_at TEXT,
    dedupe_key TEXT UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_model_prediction_history_match_id
    ON model_prediction_history(match_id);

CREATE INDEX IF NOT EXISTS idx_model_prediction_history_predicted_at
    ON model_prediction_history(predicted_at);

CREATE INDEX IF NOT EXISTS idx_model_prediction_history_unresolved
    ON model_prediction_history(actual_outcome, predicted_at);

CREATE INDEX IF NOT EXISTS idx_model_prediction_history_league_resolved
    ON model_prediction_history(league_id, actual_outcome);
