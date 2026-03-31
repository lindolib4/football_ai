from ui.main import MainWindow


def test_coupon_changed_cells_reports_exact_before_after() -> None:
    base_coupon = ["2", "1", "1", "1", "1", "2", "1", "1", "1", "1", "1", "2", "X", "1", "2"]
    insurance_coupon = ["2", "1", "1", "1", "1", "2", "1", "1", "1", "1", "X", "2", "2", "1", "2"]

    changed = MainWindow._coupon_changed_cells(base_coupon, insurance_coupon)

    assert changed == [
        (10, "1", "X"),
        (12, "X", "2"),
    ]


def test_coupon_changed_cells_empty_when_coupon_is_identical() -> None:
    coupon = ["1", "X", "2", "1", "X", "2", "1", "X", "2", "1", "X", "2", "1", "X", "2"]

    assert MainWindow._coupon_changed_cells(coupon, list(coupon)) == []


def test_format_changed_positions_lines_with_changes() -> None:
    changed_cells = [
        (10, "1", "X"),
        (11, "1", "2"),
    ]

    assert MainWindow._format_changed_positions_lines(changed_cells) == [
        "- M11: base=1 -> insurance=X",
        "- M12: base=1 -> insurance=2",
    ]


def test_format_changed_positions_lines_without_changes() -> None:
    assert MainWindow._format_changed_positions_lines([]) == [
        "No insurance changes relative to base coupon"
    ]


def test_classify_coupon_sections_uses_explicit_coupon_entries() -> None:
    sections = MainWindow._classify_coupon_sections(
        coupon_count=4,
        coupon_entries=[
            {"coupon_type": "base"},
            {"coupon_type": "base"},
            {"coupon_type": "insurance"},
            {"coupon_type": "insurance", "insurance_applied_flag": True},
        ],
    )

    assert sections == {
        "base": [0, 1],
        "insurance": [2, 3],
        "unknown": [],
    }


def test_classify_coupon_sections_falls_back_to_insured_indices() -> None:
    sections = MainWindow._classify_coupon_sections(
        coupon_count=5,
        coupon_entries=[],
        insured_indices={3, 4},
    )

    assert sections == {
        "base": [0, 1, 2],
        "insurance": [3, 4],
        "unknown": [],
    }


def test_classify_coupon_sections_returns_unknown_without_metadata() -> None:
    sections = MainWindow._classify_coupon_sections(coupon_count=3)

    assert sections == {
        "base": [],
        "insurance": [],
        "unknown": [0, 1, 2],
    }