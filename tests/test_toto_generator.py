from toto.generator import TotoGenerator


def _match(probs: dict[str, float], odds: dict[str, float] | None = None) -> dict:
    return {
        "probs": probs,
        "features": {},
        "odds": odds or {"O1": 2.0, "OX": 3.2, "O2": 4.0},
    }


def _build_matches() -> list[dict]:
    return [
        _match({"P1": 0.60, "PX": 0.22, "P2": 0.18}),
        _match({"P1": 0.45, "PX": 0.33, "P2": 0.22}),
        _match({"P1": 0.42, "PX": 0.16, "P2": 0.42}),
        _match({"P1": 0.51, "PX": 0.31, "P2": 0.18}),
        _match({"P1": 0.38, "PX": 0.18, "P2": 0.44}),
        _match({"P1": 0.57, "PX": 0.24, "P2": 0.19}),
        _match({"P1": 0.34, "PX": 0.33, "P2": 0.33}),
        _match({"P1": 0.63, "PX": 0.20, "P2": 0.17}),
        _match({"P1": 0.40, "PX": 0.30, "P2": 0.30}),
        _match({"P1": 0.59, "PX": 0.22, "P2": 0.19}),
        _match({"P1": 0.43, "PX": 0.29, "P2": 0.28}),
        _match({"P1": 0.46, "PX": 0.20, "P2": 0.34}),
        _match({"P1": 0.56, "PX": 0.24, "P2": 0.20}),
        _match({"P1": 0.37, "PX": 0.21, "P2": 0.42}),
        _match({"P1": 0.49, "PX": 0.27, "P2": 0.24}),
    ]


def test_generate_16_rows_without_duplicates_and_with_valid_structure() -> None:
    matches = _build_matches()
    coupons = TotoGenerator().generate(matches=matches, mode="16")

    assert len(coupons) == 16
    assert len({tuple(row) for row in coupons}) == 16

    for row in coupons:
        assert len(row) == 15
        assert all(item in {"1", "X", "2", "1X", "X2", "12"} for item in row)
        assert all(item for item in row)


def test_generate_32_rows_without_duplicates_and_with_valid_structure() -> None:
    matches = _build_matches()
    coupons = TotoGenerator().generate(matches=matches, mode="32")

    assert len(coupons) == 32
    assert len({tuple(row) for row in coupons}) == 32

    for row in coupons:
        assert len(row) == 15
        assert all(item in {"1", "X", "2", "1X", "X2", "12"} for item in row)
        assert all(item for item in row)
