"""Grounding reuse eval -- torch-free."""

from __future__ import annotations

import pytest

from hg_native.data.sequential_quotient import generate_trajectory
from hg_native.eval.reuse import future_external_bits, repeated_ask_rate
from hg_native.model.config import DEFAULT_ROUTES
from hg_native.training.targets import COST_PER_ASK


def _flat_trajectories(n=20, n_steps=5):
    episodes = []
    for i in range(n):
        episodes += generate_trajectory(0, i, n_steps)
    return episodes


ASK_IDX = DEFAULT_ROUTES.index("ASK")
ACT_IDX = DEFAULT_ROUTES.index("ACT")


def test_repeated_ask_rate_zero_for_well_behaved_router():
    episodes = _flat_trajectories()
    # ASK only at t=0 (not "grounded"), ACT afterward -- the idealized policy.
    route_pred = [ASK_IDX if not ep.metadata["grounded"] else ACT_IDX for ep in episodes]
    result = repeated_ask_rate(episodes, route_pred)
    assert result["r_repeat"] == pytest.approx(0.0)
    assert result["n"] == sum(1 for ep in episodes if ep.metadata["grounded"])


def test_repeated_ask_rate_one_for_a_router_that_always_asks():
    episodes = _flat_trajectories()
    route_pred = [ASK_IDX] * len(episodes)
    result = repeated_ask_rate(episodes, route_pred)
    assert result["r_repeat"] == pytest.approx(1.0)


def test_repeated_ask_rate_excludes_t0():
    # A router that asks *only* at t=0 (correctly) must not be penalized.
    episodes = _flat_trajectories(n=1, n_steps=5)
    route_pred = [ASK_IDX if not ep.metadata["grounded"] else ACT_IDX for ep in episodes]
    result = repeated_ask_rate(episodes, route_pred)
    assert result["n"] == 4  # 5 steps, t=0 excluded
    assert result["r_repeat"] == pytest.approx(0.0)


def test_repeated_ask_rate_rejects_length_mismatch():
    episodes = _flat_trajectories(n=2)
    with pytest.raises(ValueError):
        repeated_ask_rate(episodes, [0] * (len(episodes) - 1))


def test_future_external_bits_zero_for_no_repeats():
    episodes = _flat_trajectories()
    route_pred = [ACT_IDX] * len(episodes)
    assert future_external_bits(episodes, route_pred) == pytest.approx(0.0)


def test_future_external_bits_counts_each_repeat_at_fixed_cost():
    episodes = _flat_trajectories(n=1, n_steps=5)
    route_pred = [ASK_IDX] * len(episodes)  # asks every step, including 4 repeats
    n_grounded = sum(1 for ep in episodes if ep.metadata["grounded"])
    assert future_external_bits(episodes, route_pred) == pytest.approx(n_grounded * COST_PER_ASK)
