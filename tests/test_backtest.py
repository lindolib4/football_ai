from core.evaluation.backtest import BacktestEngine


def test_backtest_single_bets_accuracy_and_roi() -> None:
    matches = [
        {
            "probs": {"P1": 0.65, "PX": 0.20, "P2": 0.15},
            "features": {},
            "odds": {"O1": 2.0, "OX": 3.2, "O2": 4.5},
        },
        {
            "probs": {"P1": 0.62, "PX": 0.18, "P2": 0.20},
            "features": {},
            "odds": {"O1": 1.5, "OX": 2.0, "O2": 5.7},
        },
    ]
    results = ["1", "1"]

    outcome = BacktestEngine().evaluate(matches=matches, results=results)

    assert outcome["total_bets"] == 2
    assert outcome["wins"] == 1
    assert outcome["losses"] == 1
    assert outcome["accuracy"] == 0.5
    assert outcome["ROI"] == 0.0


def test_backtest_double_bets_are_counted_as_wins() -> None:
    matches = [
        {
            "probs": {"P1": 0.40, "PX": 0.32, "P2": 0.28},
            "features": {},
            "odds": {"O1": 2.5, "OX": 3.0, "O2": 3.8, "O1X": 1.75, "OX2": 1.70, "O12": 1.60},
        },
        {
            "probs": {"P1": 0.28, "PX": 0.32, "P2": 0.40},
            "features": {},
            "odds": {"O1": 4.0, "OX": 3.0, "O2": 2.2, "O1X": 1.95, "OX2": 1.62, "O12": 1.55},
        },
        {
            "probs": {"P1": 0.34, "PX": 0.32, "P2": 0.34},
            "features": {},
            "odds": {"O1": 2.6, "OX": 3.5, "O2": 2.6, "O1X": 1.72, "OX2": 1.72, "O12": 1.58},
        },
    ]
    results = ["X", "2", "1"]

    outcome = BacktestEngine().evaluate(matches=matches, results=results)

    assert outcome["total_bets"] == 3
    assert outcome["wins"] == 3
    assert outcome["losses"] == 0
    assert outcome["accuracy"] == 1.0


def test_backtest_skips_none_bets_and_calculates_roi() -> None:
    matches = [
        {
            "probs": {"P1": 0.50, "PX": 0.25, "P2": 0.25},
            "features": {},
            "odds": {"O1": 2.0, "OX": 2.8, "O2": 3.0},
        },
        {
            "probs": {"P1": 0.65, "PX": 0.20, "P2": 0.15},
            "features": {},
            "odds": {"O1": 2.0, "OX": 3.2, "O2": 4.5},
        },
    ]
    results = ["1", "2"]

    outcome = BacktestEngine().evaluate(matches=matches, results=results)

    assert outcome["total_bets"] == 1
    assert outcome["wins"] == 0
    assert outcome["losses"] == 1
    assert outcome["accuracy"] == 0.0
    assert outcome["ROI"] == -1.0
