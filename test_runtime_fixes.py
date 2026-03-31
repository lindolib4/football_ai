#!/usr/bin/env python3
"""Test runtime fixes: avg_goals DB fallback, UI runtime snapshot, partial_context behavior."""

from pathlib import Path
from core.model.predictor import ModelPredictor
from scheduler.auto_train import AutoTrainer
from database.db import Database
import sqlite3
import json

def main():
    # Load model
    predictor = ModelPredictor()
    predictor.load(str(Path("data/models")))
    
    # Create trainer for runtime snapshot building
    auto_trainer = AutoTrainer()
    auto_trainer.predictor = predictor
    
    # Wire DB
    db = Database(db_path=Path("database/footai.sqlite3"))
    auto_trainer.db = db
    
    # Get upcoming matches with valid odds
    conn = sqlite3.connect("database/footai.sqlite3")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM matches WHERE status != 'completed' AND "
        "odds_ft_1 > 1.01 AND odds_ft_x > 1.01 AND odds_ft_2 > 1.01 "
        "ORDER BY date_unix DESC LIMIT 12"
    ).fetchall()
    
    print("Sample partial_context matches with NEW fixes:")
    print("=" * 90)
    print()
    
    results = []
    for i, raw_row in enumerate(rows):
        match_dict = dict(raw_row)
        match_name = f"{match_dict.get('home_team')} vs {match_dict.get('away_team')}"
        
        # Build runtime snapshot (NEW: uses DB avg_goals fallback)
        snapshot = auto_trainer.build_runtime_feature_snapshot(
            match=match_dict,
            required_columns=predictor.feature_columns,
        )
        
        if not isinstance(snapshot, dict):
            print(f"{i+1}. {match_name}: snapshot build failed")
            continue
        
        features = snapshot.get("features", {})
        source_meta = snapshot.get("source_meta", {})
        
        # Call predictor with diagnostics
        result = predictor.predict_with_diagnostics(features, allow_no_odds_fallback=True)
        
        probs = result.get("probs", {})
        diagnostics = result.get("feature_diagnostics", {})
        
        p1 = float(probs.get("P1", 0.0))
        px = float(probs.get("PX", 0.0))
        p2 = float(probs.get("P2", 0.0))
        
        # Store result
        record = {
            "match": match_name,
            "P1": round(p1, 4),
            "PX": round(px, 4),
            "P2": round(p2, 4),
            "source": result.get("source", "unknown"),
            "avg_goals_source": source_meta.get("avg_goals", "unknown"),
            "avg_goals_value": float(features.get("avg_goals", 0.0)),
            "missing_non_market_names": diagnostics.get("missing_non_market_names", []),
        }
        results.append(record)
        
        print(f"{i+1}. {match_name}")
        print(f"    P1={p1:.4f}  PX={px:.4f}  P2={p2:.4f}  | source={result.get('source')}")
        print(f"    avg_goals={features.get('avg_goals', 'unknown'):.2f} ({source_meta.get('avg_goals', 'unknown')})")
        missing = diagnostics.get("missing_non_market_names", [])
        if missing:
            print(f"    missing_non_market: {missing}")
        print()
    
    conn.close()
    
    # Summary stats
    px_values = [r["PX"] for r in results]
    print("\n" + "=" * 90)
    print(f"Summary: {len(results)} matches sampled")
    print(f"  PX mean={sum(px_values)/len(px_values) if px_values else 0:.4f}")
    print(f"  PX max={max(px_values) if px_values else 0:.4f}  (should be lower than before)")
    print(f"  avg_goals sources: {set(r['avg_goals_source'] for r in results)}")
    print()

if __name__ == "__main__":
    main()
