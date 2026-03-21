from __future__ import annotations

import logging

from football_ai.api.footystats import FootyStatsClient
from football_ai.core.decision.decision_engine import DecisionEngine
from football_ai.core.features.builder import FeatureBuilder
from football_ai.core.model.predictor import Predictor
from football_ai.database.db import Database
from football_ai.logging_setup import configure_logging


logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    configure_logging()
    db = Database()
    db.initialize()

    api = FootyStatsClient()
    builder = FeatureBuilder()
    predictor = Predictor()
    engine = DecisionEngine()

    matches = api.get_todays_matches()
    for match in matches:
        match_id = str(match.get("id"))
        details = api.get_match_details(match_id)
        if not details:
            continue
        features = builder.build(details)
        db.save_features(match_id, features)

        probs = predictor.predict_proba(features)
        decision, confidence = engine.decide(probs["P1"], probs["PX"], probs["P2"])
        logger.info("match=%s probs=%s decision=%s confidence=%.3f", match_id, probs, decision, confidence)


if __name__ == "__main__":
    run_pipeline()
