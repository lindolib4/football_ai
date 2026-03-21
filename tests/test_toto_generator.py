from toto.generator import generate_coupons


def test_generate_coupons_limit() -> None:
    coupons = generate_coupons(["1", "X2", "1X", "2"], limit=4)
    assert len(coupons) == 4
