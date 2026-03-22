from toto.backtest import TotoBacktest


def test_evaluate_returns_expected_metrics_and_roi() -> None:
    coupons = [
        ["1"] * 15,
        ["1"] * 14 + ["X"],
        ["1"] * 13 + ["X", "X"],
        ["1"] * 12 + ["X", "X", "X"],
    ]
    results = ["1"] * 15

    report = TotoBacktest().evaluate(coupons=coupons, results=results)

    assert report["max_hits"] == 15
    assert report["avg_hits"] == 13.5
    assert report["distribution"] == {15: 1, 14: 1, 13: 1, 12: 1}
    assert report["ROI"] == (100000 + 5000 + 500) / 4


def test_evaluate_supports_combined_picks() -> None:
    coupons = [["1", "X2", "12", "1X", "X"]]
    results = ["1", "2", "2", "X", "X"]

    report = TotoBacktest().evaluate(coupons=coupons, results=results)

    assert report["max_hits"] == 5
    assert report["avg_hits"] == 5.0
    assert report["distribution"] == {5: 1}
    assert report["ROI"] == 0.0


def test_evaluate_uses_real_payouts_when_provided() -> None:
    coupons = [
        ["1"] * 15,
        ["1"] * 14 + ["X"],
        ["1"] * 13 + ["X", "X"],
    ]
    results = ["1"] * 15
    payouts = {15: 150000, 14: 7000, 13: 900}

    report = TotoBacktest().evaluate(coupons=coupons, results=results, payouts=payouts)

    assert report["distribution"] == {15: 1, 14: 1, 13: 1}
    assert report["ROI"] == (150000 + 7000 + 900) / 3


def test_evaluate_with_empty_coupons() -> None:
    report = TotoBacktest().evaluate(coupons=[], results=["1"] * 15)

    assert report == {
        "max_hits": 0,
        "avg_hits": 0.0,
        "distribution": {},
        "ROI": 0.0,
    }
