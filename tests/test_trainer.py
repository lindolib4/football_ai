from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from core.features.builder import FeatureBuilder
from core.model.trainer import ModelTrainer
from scheduler.auto_train import AutoTrainer


def _build_dataset(size: int = 600) -> list[dict[str, float | int | str]]:
    dataset: list[dict[str, float | int | str]] = []
    for idx in range(size):
        match = {
            "odds_ft_1": 1.9 + (idx % 15) * 0.03,
            "odds_ft_x": 3.0 + (idx % 7) * 0.04,
            "odds_ft_2": 3.3 + (idx % 11) * 0.05,
        }
        home = {
            "ppg": 2.0 + (idx % 5) * 0.1,
            "goals": 1.5 + (idx % 4) * 0.1,
            "xg": 1.4 + (idx % 6) * 0.1,
            "shots": 12.0 + (idx % 7),
            "possession": 52.0 + (idx % 8),
            "home_ppg": 2.1 + (idx % 5) * 0.1,
        }
        away = {
            "ppg": 1.2 + (idx % 5) * 0.08,
            "goals": 1.0 + (idx % 4) * 0.08,
            "xg": 0.9 + (idx % 6) * 0.07,
            "shots": 9.0 + (idx % 6),
            "possession": 44.0 + (idx % 7),
            "away_ppg": 1.1 + (idx % 4) * 0.08,
        }
        league = {"draw_pct": 0.27, "home_advantage": 0.42, "avg_goals": 2.75}

        features = FeatureBuilder.build_features(match, home, away, league)

        if idx % 3 == 0:
            target = 0
        elif idx % 3 == 1:
            target = 1
        else:
            target = 2

        row: dict[str, float | int | str] = {**features, "target": target, "match_date": f"2024-01-{(idx % 28) + 1:02d}"}
        dataset.append(row)

    return dataset


def test_trainer_train_and_predict_proba_shape() -> None:
    trainer = ModelTrainer()
    dataset = _build_dataset(600)

    model = trainer.train(dataset)
    _, x_valid, _, _ = trainer.split_chronological(dataset)
    proba = model.predict_proba(x_valid)

    assert proba.shape[0] == x_valid.shape[0]
    assert proba.shape[1] == 3
    assert not np.isnan(proba).any()


def test_trainer_save_persists_feature_schema(tmp_path: Path) -> None:
    trainer = ModelTrainer(model_path=str(tmp_path / "model.pkl"))
    dataset = _build_dataset(510)

    model = trainer.train(dataset)
    trainer.save(model)

    schema_path = tmp_path / "feature_schema.json"
    assert schema_path.exists()

    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert payload == trainer.feature_columns


def test_post_train_backtest_report_has_core_blocks() -> None:
    dataset = _build_dataset(180)
    trainer = ModelTrainer()
    cleaned, _ = trainer.clean_data_with_report(dataset)

    auto = AutoTrainer()

    def _fake_predict(features: dict[str, float]) -> dict[str, float]:
        if float(features.get("ppg_diff", 0.0)) >= 0.0:
            return {"P1": 0.62, "PX": 0.21, "P2": 0.17}
        return {"P1": 0.19, "PX": 0.23, "P2": 0.58}

    auto.predict = _fake_predict  # type: ignore[method-assign]

    report = auto._run_post_train_backtest(  # type: ignore[attr-defined]
        raw_rows=cleaned,
        feature_columns=list(trainer.feature_columns),
        train_ratio=0.8,
    )

    assert report["backtest_run"] is True
    assert report["evaluated_matches"] > 0
    assert isinstance(report["metrics"]["accuracy"], float)
    assert "class_metrics" in report
    assert "context_metrics" in report
    assert "signal_quality" in report
    assert "market_vs_stats" in report
    assert "summary_lines" in report and report["summary_lines"]


def test_post_train_backtest_not_run_for_tiny_dataset() -> None:
    dataset = _build_dataset(60)
    trainer = ModelTrainer()
    cleaned, _ = trainer.clean_data_with_report(dataset)

    auto = AutoTrainer()
    report = auto._run_post_train_backtest(  # type: ignore[attr-defined]
        raw_rows=cleaned,
        feature_columns=list(trainer.feature_columns),
        train_ratio=0.8,
    )

    assert report["backtest_run"] is False
    assert report["reason"]
