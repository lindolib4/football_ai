from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import settings
from core.features.builder import FeatureBuilder
from database.db import Database
from scheduler.auto_train import AutoTrainer


def main() -> None:
    db = Database(settings.db_path)
    try:
        api_sqlite_audit = db.audit_api_sqlite_pipeline(limit=5000)
        dataset = db.build_training_dataset_from_db()

        trainer = AutoTrainer()
        feature_columns = list(FeatureBuilder.build_features({}, {}, {}, {}).keys())
        profile = trainer._build_feature_profile(dataset, feature_columns)

        spotlight = [
            "goals_diff",
            "xg_diff",
            "shots_diff",
            "possession_diff",
            "draw_pct",
            "implied_prob_1",
            "implied_prob_x",
            "implied_prob_2",
            "volatility",
            "entropy",
        ]

        per_feature = profile.get("per_feature", {}) if isinstance(profile, dict) else {}
        spotlight_summary = {
            key: per_feature.get(key, {})
            for key in spotlight
            if key in per_feature
        }

        report = {
            "db_path": str(Path(settings.db_path).resolve()),
            "training_rows": len(dataset),
            "api_sqlite_audit": api_sqlite_audit,
            "feature_quality": {
                "weak_features": profile.get("weak_features", []),
                "healthy_features": profile.get("healthy_features", []),
                "spotlight": spotlight_summary,
            },
            "notes": [
                "fallback_rate reflects rows where feature came from fallback/default source markers.",
                "zero_rate reflects numeric zeros among non-null values.",
                "Use this report to identify whether weakness originates in upstream API coverage or feature defaults.",
            ],
        }

        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        db.close()


if __name__ == "__main__":
    main()
