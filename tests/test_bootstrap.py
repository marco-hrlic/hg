"""Resampling conventions from protocol section 12."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg.metrics.bootstrap import (
    align_matrices,
    bootstrap_metric,
    odds_ratio,
    paired_bootstrap,
)
from hg.metrics.core import clean_consensus_wrong, consensus_wrong, diversity_gap
from tests.conftest import make_matrix


def rate_matrix(rate: float, n: int = 400, n_members: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    all_wrong = rng.random(n) < rate
    e = np.zeros((n, n_members), dtype=bool)
    e[all_wrong] = True
    return make_matrix(e)


def test_bootstrap_interval_covers_the_point_estimate():
    m = rate_matrix(0.2, n=500)
    res = bootstrap_metric(m, consensus_wrong, n_boot=300, seed=0)
    assert res.ci_low <= res.point <= res.ci_high
    assert res.point == pytest.approx(consensus_wrong(m))


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    m = rate_matrix(0.2, n=300)
    a = bootstrap_metric(m, consensus_wrong, n_boot=200, seed=5)
    b = bootstrap_metric(m, consensus_wrong, n_boot=200, seed=5)
    assert a.draws == b.draws


def test_interval_narrows_with_more_items():
    small = bootstrap_metric(rate_matrix(0.2, n=200, seed=1), consensus_wrong, n_boot=400, seed=0)
    large = bootstrap_metric(rate_matrix(0.2, n=4000, seed=1), consensus_wrong, n_boot=400, seed=0)
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_paired_bootstrap_requires_matching_items():
    a = rate_matrix(0.2, n=100)
    b = rate_matrix(0.1, n=90)
    with pytest.raises(ValueError, match="same items"):
        paired_bootstrap(a, b, consensus_wrong)


def test_paired_bootstrap_is_tighter_than_unpaired_when_conditions_correlate():
    """The whole reason the protocol requires pairing: independent resampling
    discards the item-level correlation that makes the comparison sensitive."""
    rng = np.random.default_rng(3)
    n = 800
    hard = rng.random(n) < 0.25  # shared difficulty drives both conditions
    e_a = np.tile((hard | (rng.random(n) < 0.05))[:, None], (1, 3))
    e_b = np.tile((hard & (rng.random(n) < 0.5))[:, None], (1, 3))
    a, b = make_matrix(e_a), make_matrix(e_b)

    paired = paired_bootstrap(a, b, consensus_wrong, n_boot=600, seed=0)

    # Unpaired: resample each condition with its own indices.
    rng2 = np.random.default_rng(0)
    draws = []
    for _ in range(600):
        ia = rng2.integers(0, n, size=n)
        ib = rng2.integers(0, n, size=n)
        draws.append(consensus_wrong(a.resample_items(ia)) - consensus_wrong(b.resample_items(ib)))
    unpaired_se = float(np.std(draws, ddof=1))
    assert paired.se < unpaired_se


def test_paired_bootstrap_supports_a_ratio_via_combine():
    a = rate_matrix(0.2, n=600, seed=2)
    b = rate_matrix(0.1, n=600, seed=3)
    res = paired_bootstrap(
        a, b, consensus_wrong, combine=diversity_gap, null_value=1.0, n_boot=300, seed=0
    )
    assert res.point > 1.0
    assert res.ci_low <= res.point <= res.ci_high


def test_odds_ratio_is_finite_when_one_rate_is_zero():
    """Gate A -- a zero diverse residual -- is the interesting case, so the
    odds ratio must not become undefined exactly there."""
    a = rate_matrix(0.2, n=400, seed=4)
    b = make_matrix(np.zeros((400, 3), dtype=bool))
    res = odds_ratio(a, b, consensus_wrong, n_boot=200, seed=0)
    assert math.isfinite(res.point)
    assert res.point > 1.0


def test_align_matrices_restricts_to_shared_items_in_common_order():
    a = make_matrix(np.zeros((5, 2), dtype=bool))
    b = make_matrix(np.zeros((3, 2), dtype=bool))
    a2, b2 = align_matrices(a, b)
    assert a2.item_ids == b2.item_ids == ["i0", "i1", "i2"]


def test_align_matrices_rejects_disjoint_item_sets():
    a = make_matrix(np.zeros((2, 2), dtype=bool))
    b = make_matrix(np.zeros((2, 2), dtype=bool))
    b.item_ids = ["z0", "z1"]
    with pytest.raises(ValueError, match="share no items"):
        align_matrices(a, b)


def test_p_value_is_reported_but_interval_is_primary():
    m = rate_matrix(0.3, n=500)
    res = bootstrap_metric(m, consensus_wrong, n_boot=300, seed=0, null_value=0.0)
    assert res.p_value is not None
    assert res.null_value == 0.0
    assert res.ci_low > 0.0
