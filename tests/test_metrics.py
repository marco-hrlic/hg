"""Section 8 metrics, on matrices whose answers are worked out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg.metrics.core import (
    capability_spread,
    clean_consensus_wrong,
    condition_metrics,
    consensus_wrong,
    contested_share_of_residual,
    disagreement_rate,
    diversity_gap,
    error_conversion,
    invisible_error_rate,
    localization_surplus,
    member_accuracy,
)
from tests.conftest import make_matrix


def test_consensus_wrong_counts_only_all_wrong_items():
    m = make_matrix([[1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1]])
    assert consensus_wrong(m) == pytest.approx(2 / 4)


def test_clean_consensus_wrong_conditions_on_unanimous_items():
    m = make_matrix(
        [[1, 1], [1, 1], [1, 1]],
        statuses=["unanimous", "contested", "unanimous"],
    )
    assert consensus_wrong(m) == 1.0
    assert clean_consensus_wrong(m) == 1.0
    m2 = make_matrix(
        [[1, 1], [1, 1], [0, 0]],
        statuses=["contested", "contested", "unanimous"],
    )
    # Both consensus failures sit on contested anchors, so the clean rate is 0.
    assert consensus_wrong(m2) == pytest.approx(2 / 3)
    assert clean_consensus_wrong(m2) == 0.0


def test_empty_clean_subset_is_nan_not_zero():
    """'No unanimous items' and 'no unanimous failures' are opposite claims."""
    m = make_matrix([[1, 1]], statuses=["contested"])
    assert math.isnan(clean_consensus_wrong(m))


def test_disagreement_is_about_predictions_not_correctness():
    """Two members can be wrong in the same way (invisible) or differently."""
    preds = np.array([["a", "a"], ["a", "b"]], dtype=object)
    m = make_matrix([[1, 1], [1, 1]], predictions=preds)
    assert consensus_wrong(m) == 1.0
    assert disagreement_rate(m) == pytest.approx(0.5)


def test_invisible_error_equals_consensus_wrong_in_binary_tasks():
    """In a binary task all-wrong implies all-agreed. Worth pinning down:
    it is why the localization race cannot test P5 on binary labels."""
    m = make_matrix([[1, 1, 1], [0, 1, 0], [0, 0, 0]])
    assert invisible_error_rate(m) == pytest.approx(consensus_wrong(m))


def test_invisible_error_below_consensus_wrong_with_three_labels():
    preds = np.array([["b", "c"], ["b", "b"]], dtype=object)
    m = make_matrix([[1, 1], [1, 1]], predictions=preds)
    assert consensus_wrong(m) == 1.0
    assert invisible_error_rate(m) == pytest.approx(0.5)


def test_contested_share_of_residual():
    m = make_matrix(
        [[1, 1], [1, 1], [1, 1], [0, 0]],
        statuses=["contested", "unanimous", "contested", "unanimous"],
    )
    assert contested_share_of_residual(m) == pytest.approx(2 / 3)


def test_contested_share_is_nan_when_no_residual():
    m = make_matrix([[0, 0], [0, 0]])
    assert math.isnan(contested_share_of_residual(m))


def test_diversity_gap_is_infinite_when_diverse_residual_vanishes():
    """Gate A is the interesting outcome, not an error to swallow."""
    assert diversity_gap(0.1, 0.0) == math.inf
    assert math.isnan(diversity_gap(0.0, 0.0))
    assert diversity_gap(0.2, 0.1) == pytest.approx(2.0)


def test_member_accuracy_and_capability_spread():
    m = make_matrix([[1, 0], [1, 0], [0, 0], [0, 0]])
    accs = member_accuracy(m)
    assert accs["m0"] == pytest.approx(0.5)
    assert accs["m1"] == pytest.approx(1.0)
    assert capability_spread(m)["range"] == pytest.approx(0.5)


def test_localization_surplus_sign():
    assert localization_surplus(0.7, 0.5) == pytest.approx(0.2)
    assert localization_surplus(0.4, 0.5) == pytest.approx(-0.1)


def test_error_conversion_requires_same_items():
    a = make_matrix([[1, 1], [0, 0]])
    b = make_matrix([[1, 1]])
    with pytest.raises(ValueError, match="same items"):
        error_conversion(a, b)


def test_error_conversion_measures_disagreement_gained_per_error_removed():
    preds_h = np.array([["b", "b"], ["b", "b"], ["a", "a"]], dtype=object)
    preds_d = np.array([["b", "b"], ["b", "a"], ["a", "a"]], dtype=object)
    h = make_matrix([[1, 1], [1, 1], [0, 0]], predictions=preds_h)
    d = make_matrix([[1, 1], [1, 0], [0, 0]], predictions=preds_d)
    conv = error_conversion(h, d)
    assert conv["delta_clean_consensus_wrong"] == pytest.approx(1 / 3)
    assert conv["delta_clean_disagreement"] == pytest.approx(1 / 3)
    assert conv["conversion_ratio"] == pytest.approx(1.0)


def test_condition_metrics_reports_every_section_8_quantity():
    m = make_matrix(
        [[1, 1, 1], [0, 1, 0], [1, 1, 1], [0, 0, 0]],
        statuses=["unanimous", "unanimous", "contested", "majority"],
    )
    d = condition_metrics(m).to_dict()
    for key in ("C_E", "C_E_clean", "D_E", "Q_contested", "invisible_error_rate"):
        assert key in d
    assert d["n_clean"] == 2
    assert d["n_contested"] == 1
