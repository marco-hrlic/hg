"""The dependence models, checked against known generative truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg.metrics.dependence import (
    excess_consensus,
    fit_item_difficulty,
    fit_random_effects,
    independence_null,
    mean_pairwise_correlation,
    pairwise_error_correlation,
    within_between_correlation,
)
from tests.conftest import make_matrix, simulate


def test_independence_null_is_product_of_marginals():
    m = make_matrix([[1, 1], [1, 0], [0, 1], [0, 0]])
    assert independence_null(m) == pytest.approx(0.5 * 0.5)


def test_correlation_is_nan_for_a_member_with_no_error_variance():
    """An undefined correlation is not independence."""
    m = make_matrix([[1, 0], [1, 0], [0, 0]])
    rho = pairwise_error_correlation(m)
    assert math.isnan(rho[0, 1])


def test_perfectly_correlated_members():
    m = make_matrix([[1, 1], [0, 0], [1, 1], [0, 0]])
    assert mean_pairwise_correlation(m) == pytest.approx(1.0)


def test_within_between_split_by_family():
    e = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1], [0, 0, 0]])
    m = make_matrix(e, families=["fam_a", "fam_a", "fam_b"])
    wb = within_between_correlation(m)
    assert wb["n_within_pairs"] == 1
    assert wb["n_between_pairs"] == 2
    assert wb["within_family"] > wb["between_family"]


def test_random_effects_recovers_sigma_u_and_alpha():
    """The hierarchical fit is the inferential baseline, so it must be right."""
    m, truth = simulate(n_items=4000, n_members=5, sigma_u=1.2, alpha=-1.5, seed=1)
    fit = fit_random_effects(m)
    assert fit.converged
    assert fit.sigma_u == pytest.approx(1.2, abs=0.15)
    assert float(np.mean(fit.alpha)) == pytest.approx(-1.5, abs=0.2)


def test_random_effects_predicts_the_observed_consensus_rate_under_the_null():
    """Data drawn from the model: predicted and observed must agree."""
    m, _ = simulate(n_items=5000, n_members=5, sigma_u=1.2, alpha=-1.5, seed=2)
    fit = fit_random_effects(m)
    observed = float(m.all_wrong().mean())
    assert fit.predicted_consensus() == pytest.approx(observed, abs=0.004)


def test_excess_over_hierarchical_is_near_zero_under_the_null():
    """Calibration check: no core, so there is no residual to find."""
    m, _ = simulate(n_items=5000, n_members=5, sigma_u=1.2, alpha=-1.5, seed=3)
    ex = excess_consensus(m)
    assert abs(ex["excess_over_hierarchical"]) < 0.005


def test_excess_over_hierarchical_detects_a_real_core():
    m, _ = simulate(
        n_items=5000, n_members=5, sigma_u=1.0, alpha=-1.6, core_rate=0.03, seed=4
    )
    ex = excess_consensus(m)
    assert ex["excess_over_hierarchical"] > 0.005
    # The naive independence null is wildly optimistic and must not be the
    # baseline a residual claim rests on.
    assert ex["excess_over_independence"] > ex["excess_over_hierarchical"]


def test_joint_fit_overpredicts_consensus_relative_to_marginal_fit():
    """Why the marginal fit is the baseline.

    The joint MLE estimates one difficulty per item from E observations, so its
    difficulty spread carries its own sampling noise; predicted consensus is
    convex in difficulty, so the noise inflates it. That bias would mask a real
    residual, which is the wrong direction to be wrong in.
    """
    m, _ = simulate(n_items=3000, n_members=5, sigma_u=1.2, alpha=-1.5, seed=5)
    joint = fit_item_difficulty(m)
    marginal = fit_random_effects(m)
    observed = float(m.all_wrong().mean())
    assert joint.predicted_consensus() > marginal.predicted_consensus()
    assert abs(marginal.predicted_consensus() - observed) < abs(
        joint.predicted_consensus() - observed
    )


def test_implied_rho_matches_observed_under_the_null():
    m, _ = simulate(n_items=5000, n_members=5, sigma_u=1.2, alpha=-1.5, seed=6)
    fit = fit_random_effects(m)
    assert fit.implied_pairwise_rho() == pytest.approx(
        mean_pairwise_correlation(m), abs=0.03
    )


def test_fit_rejects_empty_matrix():
    m = make_matrix([[1, 1]]).select_items(np.array([False]))
    with pytest.raises(ValueError, match="empty matrix"):
        fit_random_effects(m)
