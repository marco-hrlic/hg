"""Vanilla confidence-based router -- torch-free (M0-R1 kickoff sections 9-11)."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.model.config import DEFAULT_ROUTES
from hg_native.eval.vanilla_router import vanilla_route_predictions


def _one_bit_episodes(n=20):
    return [gen_static_hidden(0, "one_bit_deficit", i) for i in range(n)]


def _no_deficit_episodes(n=20):
    return [gen_static_hidden(0, "no_deficit", i) for i in range(n)]


def test_vanilla_router_receives_no_generator_ground_truth():
    # kickoff section 9/19's explicit requirement: the router may only use
    # vanilla-model/proxy quantities at test time. Assert it by construction
    # -- the function signature has no episode-truth parameter to pass.
    params = set(inspect.signature(vanilla_route_predictions).parameters)
    assert params == {"episodes", "info_deficit_prob", "accessibility_prob", "lambda_b", "lambda_k"}


def test_vanilla_router_asks_when_info_deficit_confidently_predicted():
    episodes = _one_bit_episodes(10)
    info = np.ones(10)
    access = np.zeros(10)
    preds = vanilla_route_predictions(episodes, info, access)
    assert all(p == DEFAULT_ROUTES.index("ASK") for p in preds)


def test_vanilla_router_acts_when_no_deficit_confidently_predicted():
    episodes = _one_bit_episodes(10)
    info = np.zeros(10)
    access = np.zeros(10)
    preds = vanilla_route_predictions(episodes, info, access)
    assert all(p == DEFAULT_ROUTES.index("ACT") for p in preds)


def test_vanilla_router_never_predicts_a_route_the_episode_does_not_offer():
    # no_deficit static-hidden episodes offer no ASK/THINK actions.
    episodes = _no_deficit_episodes(10)
    rng = np.random.default_rng(0)
    info = rng.uniform(size=10)
    access = rng.uniform(size=10)
    preds = vanilla_route_predictions(episodes, info, access)
    assert all(p == DEFAULT_ROUTES.index("ACT") for p in preds)


def test_vanilla_router_can_choose_think_when_available():
    episodes = [gen_accessibility(0, "canonical_compact", i) for i in range(10)]
    info = np.zeros(10)
    access = np.ones(10)  # confidently predicted accessibility failure
    preds = vanilla_route_predictions(episodes, info, access)
    assert all(p == DEFAULT_ROUTES.index("THINK") for p in preds)


def test_vanilla_router_rejects_length_mismatch():
    episodes = _one_bit_episodes(5)
    with pytest.raises(ValueError):
        vanilla_route_predictions(episodes, np.zeros(4), np.zeros(5))


def test_vanilla_router_higher_lambda_b_discourages_ask():
    episodes = _one_bit_episodes(10)
    info = np.full(10, 0.3)  # a mild, not-overwhelming deficit signal
    access = np.zeros(10)
    cheap = vanilla_route_predictions(episodes, info, access, lambda_b=0.1)
    expensive = vanilla_route_predictions(episodes, info, access, lambda_b=100.0)
    assert (cheap == DEFAULT_ROUTES.index("ASK")).sum() >= (expensive == DEFAULT_ROUTES.index("ASK")).sum()
