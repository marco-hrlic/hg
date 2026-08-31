"""Splits, overlap, and the survival curve."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg.diversity.overlap import jaccard, mean_overlap, overlap_matrix, overlap_report
from hg.diversity.splits import build_splits
from hg.diversity.survival import excess_survival, residual_survival, survival_shape
from hg.metrics.core import clean_consensus_wrong
from tests.conftest import make_matrix, simulate

ITEMS = [f"i{k}" for k in range(100)]


def test_jaccard_basics():
    assert jaccard(["a", "b"], ["a", "b"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard([], []) == 0.0


def test_shared_policy_gives_identical_slices():
    plan = build_splits(ITEMS, ["m0", "m1"], policy="shared", n_eval=20, n_train_per_member=30)
    assert plan.train_by_member["m0"] == plan.train_by_member["m1"]
    assert mean_overlap(overlap_matrix(plan.train_by_member)) == pytest.approx(1.0)


def test_disjoint_policy_gives_zero_overlap():
    plan = build_splits(ITEMS, ["m0", "m1", "m2"], policy="disjoint", n_eval=10, n_train_per_member=20)
    assert mean_overlap(overlap_matrix(plan.train_by_member)) == 0.0


def test_evaluation_items_never_leak_into_training():
    plan = build_splits(ITEMS, ["m0", "m1"], policy="disjoint", n_eval=40, n_train_per_member=20)
    eval_set = set(plan.eval_ids)
    for ids in plan.train_by_member.values():
        assert not (eval_set & set(ids))


def test_disjoint_requires_equal_per_member_budget_to_fit():
    """Silently shrinking per-member slices would confound information
    diversity with training-set size."""
    with pytest.raises(ValueError, match="Equal per-member budget"):
        build_splits(ITEMS, ["m0", "m1", "m2"], policy="disjoint", n_eval=50, n_train_per_member=40)


def test_splits_are_deterministic_across_calls():
    a = build_splits(ITEMS, ["m0", "m1"], policy="disjoint", n_eval=20, n_train_per_member=20, seed=7)
    b = build_splits(ITEMS, ["m0", "m1"], policy="disjoint", n_eval=20, n_train_per_member=20, seed=7)
    assert a.eval_ids == b.eval_ids
    assert a.train_by_member == b.train_by_member


def test_splits_change_with_seed():
    a = build_splits(ITEMS, ["m0"], policy="shared", n_eval=20, n_train_per_member=20, seed=1)
    b = build_splits(ITEMS, ["m0"], policy="shared", n_eval=20, n_train_per_member=20, seed=2)
    assert a.eval_ids != b.eval_ids


def test_overlap_report_flags_a_disjoint_claim_that_leaks():
    matrix = overlap_matrix({"m0": ["a", "b"], "m1": ["b", "c"]})
    report = overlap_report(matrix, "disjoint")
    assert report["warnings"]
    assert "declares disjoint" in report["warnings"][0]


def test_overlap_report_flags_a_shared_claim_that_differs():
    matrix = overlap_matrix({"m0": ["a", "b"], "m1": ["a"]})
    report = overlap_report(matrix, "shared")
    assert report["warnings"]


def test_survival_curve_falls_with_more_families():
    m, _ = simulate(n_items=3000, n_members=5, sigma_u=1.2, alpha=-1.5, seed=11)
    curve = residual_survival(m, clean_consensus_wrong)
    means = [p.mean for p in curve]
    assert len(curve) == 5
    assert all(a >= b for a, b in zip(means, means[1:])), means


def test_survival_averages_over_families_not_members():
    """A homogeneous ensemble has one family, so there is one point: k=1.

    Averaging over arbitrary member subsets would trace out a seed-variation
    curve and label it a diversity curve.
    """
    m = make_matrix(np.zeros((10, 5), dtype=bool), families=["fam"] * 5)
    curve = residual_survival(m, clean_consensus_wrong)
    assert [p.k for p in curve] == [1]


def test_closure_shape_detected():
    m = make_matrix(np.zeros((50, 3), dtype=bool))
    shape = survival_shape(residual_survival(m, clean_consensus_wrong))
    assert shape["shape"] == "closure"
    assert shape["decision_gate"] == "A"
    assert "caveat" in shape


def test_flat_nonzero_curve_is_a_plateau_at_gate_c():
    from hg.diversity.survival import SurvivalPoint

    points = [SurvivalPoint(k, 0.05, 0.001, 0.05, 0.05, 1) for k in (1, 2, 3, 4, 5)]
    shape = survival_shape(points)
    assert shape["shape"] == "apparent_plateau"
    assert shape["decision_gate"] == "C"


def test_still_falling_curve_is_gate_b_not_c():
    """A nonzero endpoint on a falling curve is an upper bound, not a floor."""
    from hg.diversity.survival import SurvivalPoint

    means = [0.30, 0.15, 0.08, 0.04, 0.02]
    points = [SurvivalPoint(k, mu, 0.002, mu, mu, 1) for k, mu in enumerate(means, start=1)]
    shape = survival_shape(points)
    assert shape["shape"] == "still_falling"
    assert shape["decision_gate"] == "B"


def test_single_point_curve_cannot_claim_a_plateau():
    from hg.diversity.survival import SurvivalPoint

    shape = survival_shape([SurvivalPoint(1, 0.1, 0.0, 0.1, 0.1, 1)])
    assert shape["shape"] == "undetermined"
    assert shape["decision_gate"] == "B"


def test_excess_survival_tracks_the_model_when_there_is_no_core():
    """The discriminating test: no core, so observed should track predicted."""
    m, _ = simulate(n_items=6000, n_members=5, sigma_u=1.2, alpha=-1.5, seed=21)
    ex = excess_survival(m)
    assert ex is not None
    assert ex.ratio_growth < 0.25
    shape = survival_shape(residual_survival(m, clean_consensus_wrong), excess=ex)
    assert shape["excess_verdict"] == "consistent_with_shared_difficulty"


def test_excess_survival_detects_a_real_core():
    """A core makes the observed curve pull away from a model that cannot
    represent a floor. This is what separates gate A from gate C at E=5, where
    the plateau test alone cannot."""
    m, _ = simulate(
        n_items=6000, n_members=5, sigma_u=1.2, alpha=-1.5, core_rate=0.03, seed=22
    )
    ex = excess_survival(m)
    assert ex is not None
    assert ex.ratio_growth > 0.25
    shape = survival_shape(residual_survival(m, clean_consensus_wrong), excess=ex)
    assert shape["excess_verdict"] == "residual_exceeds_shared_difficulty"


def test_excess_survival_needs_at_least_two_families():
    m = make_matrix(np.zeros((10, 3), dtype=bool), families=["fam"] * 3)
    assert excess_survival(m) is None
