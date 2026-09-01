"""Generic item-level bootstrap CI -- torch-free."""

from __future__ import annotations

import numpy as np
import pytest

from hg_native.eval.binary_metrics import auroc
from hg_native.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff


def _mean(x):
    return float(np.mean(x))


def test_bootstrap_ci_point_matches_metric_exactly():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = bootstrap_ci(data, metric=_mean, n_boot=500, seed=0)
    assert result.point == pytest.approx(3.0)


def test_bootstrap_ci_deterministic_given_seed():
    data = list(range(20))
    a = bootstrap_ci(data, metric=_mean, n_boot=200, seed=0)
    b = bootstrap_ci(data, metric=_mean, n_boot=200, seed=0)
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high


def test_bootstrap_ci_contains_point_estimate_typically():
    data = list(range(50))
    result = bootstrap_ci(data, metric=_mean, n_boot=1000, seed=0)
    assert result.ci_low <= result.point <= result.ci_high


def test_bootstrap_ci_shrinks_with_more_items():
    rng = np.random.default_rng(0)
    small = rng.normal(size=10)
    large = rng.normal(size=1000)
    small_result = bootstrap_ci(small, metric=_mean, n_boot=1000, seed=0)
    large_result = bootstrap_ci(large, metric=_mean, n_boot=1000, seed=0)
    assert (large_result.ci_high - large_result.ci_low) < (small_result.ci_high - small_result.ci_low)


def test_bootstrap_ci_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError, match="same length"):
        bootstrap_ci([1, 2, 3], [1, 2], metric=lambda a, b: float(np.mean(a - b)))


def test_bootstrap_ci_works_with_paired_scores_and_labels_metric():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    result = bootstrap_ci(scores, labels, metric=auroc, n_boot=200, seed=0)
    assert result.point == pytest.approx(1.0)


# --- paired_bootstrap_diff ---------------------------------------------------


def test_paired_bootstrap_diff_point_is_exact_difference():
    a = [1.0, 2.0, 3.0]
    b = [1.0, 1.0, 1.0]
    result = paired_bootstrap_diff((a,), (b,), metric=_mean, n_boot=500, seed=0)
    assert result.point == pytest.approx(2.0 - 1.0)


def test_paired_bootstrap_diff_identical_conditions_centers_near_zero():
    a = list(range(30))
    result = paired_bootstrap_diff((a,), (a,), metric=_mean, n_boot=1000, seed=0)
    assert result.point == pytest.approx(0.0)
    assert result.ci_low <= 0.0 <= result.ci_high


def test_paired_bootstrap_diff_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same number of items"):
        paired_bootstrap_diff(([1, 2, 3],), ([1, 2],), metric=_mean)
