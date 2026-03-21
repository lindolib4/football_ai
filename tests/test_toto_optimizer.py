from __future__ import annotations

from toto.optimizer import TotoOptimizer


def _match(p1: float, px: float, p2: float, decision: str) -> dict:
    return {
        "probs": {"P1": p1, "PX": px, "P2": p2},
        "decision": decision,
    }


def _build_matches() -> list[dict]:
    return [
        _match(0.70, 0.20, 0.10, "1"),
        _match(0.34, 0.33, 0.33, "1X"),
        _match(0.68, 0.21, 0.11, "1"),
        _match(0.33, 0.34, 0.33, "X2"),
        _match(0.64, 0.22, 0.14, "1"),
        _match(0.33, 0.33, 0.34, "12"),
        _match(0.62, 0.23, 0.15, "1"),
        _match(0.35, 0.32, 0.33, "1X"),
        _match(0.35, 0.33, 0.32, "12"),
        _match(0.66, 0.20, 0.14, "1"),
        _match(0.65, 0.21, 0.14, "1"),
        _match(0.63, 0.22, 0.15, "1"),
        _match(0.69, 0.18, 0.13, "1"),
        _match(0.71, 0.19, 0.10, "1"),
        _match(0.72, 0.18, 0.10, "1"),
    ]


def test_optimize_16_rows_without_duplicates() -> None:
    matches = _build_matches()
    coupons = TotoOptimizer().optimize(matches=matches, mode="16")

    assert len(coupons) == 16
    assert len({tuple(row) for row in coupons}) == 16
    assert all(len(row) == 15 for row in coupons)


def test_risk_matches_are_varied_for_mode_16() -> None:
    matches = _build_matches()
    coupons = TotoOptimizer().optimize(matches=matches, mode="16")

    varied_indexes = [1, 3, 5, 7]
    fixed_indexes = [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14]

    for idx in varied_indexes:
        assert len({coupon[idx] for coupon in coupons}) >= 2

    for idx in fixed_indexes:
        assert len({coupon[idx] for coupon in coupons}) == 1


def test_optimize_32_rows_and_coverage_score() -> None:
    matches = _build_matches()
    coupons = TotoOptimizer().optimize(matches=matches, mode="32")

    assert len(coupons) == 32
    assert len({tuple(row) for row in coupons}) == 32

    score = TotoOptimizer().coverage_score(coupons=coupons, matches=matches)
    assert 0.0 <= score <= 1.0
    assert score > 0.60
