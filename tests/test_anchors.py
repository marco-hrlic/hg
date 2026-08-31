"""Anchor status derivation -- the distinction the whole experiment rests on."""

from __future__ import annotations

import pytest

from hg.anchors.agreement import (
    chance_corrected_agreement,
    clean_subset_warnings,
    panel_report,
    pairwise_agreement,
)
from hg.anchors.panel import (
    PanelItem,
    anchor_status,
    build_anchored_items,
    reference_label,
)
from hg.schema import ANCHOR_CONTESTED, ANCHOR_MAJORITY, ANCHOR_UNANIMOUS


def test_unanimous_panel():
    status, agreement = anchor_status(["a", "a", "a"])
    assert status == ANCHOR_UNANIMOUS
    assert agreement == 1.0


def test_clear_majority():
    status, agreement = anchor_status(["a", "a", "a", "a", "b"])
    assert status == ANCHOR_MAJORITY
    assert agreement == pytest.approx(0.8)


def test_even_split_is_contested():
    status, _ = anchor_status(["a", "a", "b", "b"])
    assert status == ANCHOR_CONTESTED


def test_three_way_split_is_contested():
    status, _ = anchor_status(["a", "b", "c"])
    assert status == ANCHOR_CONTESTED


def test_bare_majority_can_be_pushed_to_contested_by_threshold():
    """With an odd panel and two labels a tie is impossible.

    A 3-2 split is the most divided such a panel can be, at a 60% modal share.
    Raising the majority bar above it is the only way to record those items as
    contested -- and without it the contested subset is empty, which is a real
    failure mode the fixtures hit.
    """
    judgments = ["a", "a", "a", "b", "b"]
    assert anchor_status(judgments)[0] == ANCHOR_MAJORITY
    assert anchor_status(judgments, majority_threshold=0.66)[0] == ANCHOR_CONTESTED


def test_contested_margin_demotes_thin_majorities():
    judgments = ["a", "a", "a", "b", "b"]
    assert anchor_status(judgments, contested_margin=0.5)[0] == ANCHOR_CONTESTED


def test_reference_label_tie_break_is_deterministic():
    a = reference_label(["b", "a"])
    b = reference_label(["a", "b"])
    assert a == b == "a"


def test_curated_label_conflicting_with_unanimous_panel_is_contested():
    """An override against a unanimous panel is an adjudication decision.

    Letting it into the clean subset would put an item the humans agreed on,
    and were overruled on, into the population used to claim a clean systematic
    residual.
    """
    items = build_anchored_items([PanelItem("i0", "x", ["a", "a", "a"], reference="b")])
    assert items[0].status == ANCHOR_CONTESTED
    assert items[0].reference == "b"
    assert not items[0].is_clean


def test_curated_label_agreeing_with_panel_stays_clean():
    items = build_anchored_items([PanelItem("i0", "x", ["a", "a", "a"], reference="a")])
    assert items[0].status == ANCHOR_UNANIMOUS
    assert items[0].is_clean


def test_empty_panel_rejected():
    with pytest.raises(Exception):
        PanelItem("i0", "x", [])


def test_panel_report_counts_and_agreement():
    items = build_anchored_items([
        PanelItem("i0", "x", ["a", "a", "a"]),
        PanelItem("i1", "x", ["a", "a", "b"]),
        PanelItem("i2", "x", ["a", "b", "c"]),
    ])
    report = panel_report(items)
    assert report["status_counts"] == {"unanimous": 1, "majority": 1, "contested": 1}
    assert 0.0 <= report["observed_agreement"] <= 1.0
    assert report["observed_agreement"] == pytest.approx(pairwise_agreement(items))


def test_warning_when_no_contested_items():
    """Q_contested and P3 are untestable without contested items."""
    items = build_anchored_items([PanelItem(f"i{i}", "x", ["a"] * 5) for i in range(10)])
    warnings = clean_subset_warnings(panel_report(items), min_clean=1)
    assert any("no contested items" in w for w in warnings)


def test_warning_when_panel_too_small_to_define_unanimity():
    items = build_anchored_items([PanelItem(f"i{i}", "x", ["a", "b"]) for i in range(10)])
    warnings = clean_subset_warnings(panel_report(items), min_clean=1)
    assert any("panel size < 3" in w for w in warnings)
