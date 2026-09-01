"""Exhaustive ground-truth tests for the Family C (Sequential Known Quotient) generator."""

from __future__ import annotations

import pytest

from hg_native.data.sequential_quotient import generate_batch, generate_trajectory

SEEDS = (0, 1, 12345)
N_TRAJECTORIES_PER_SEED = 60
N_STEPS = 6


def _trajectories():
    for seed in SEEDS:
        for index in range(N_TRAJECTORIES_PER_SEED):
            yield generate_trajectory(seed, index, N_STEPS)


def test_rejects_zero_steps():
    with pytest.raises(ValueError, match="n_steps"):
        generate_trajectory(0, 0, 0)


def test_deterministic():
    for seed in SEEDS:
        assert generate_trajectory(seed, 3, N_STEPS) == generate_trajectory(seed, 3, N_STEPS)


def test_shorter_trajectory_is_a_strict_prefix_of_a_longer_one():
    long = generate_trajectory(0, 0, 10)
    short = generate_trajectory(0, 0, 4)
    assert short == long[:4]


def test_generate_batch_matches_generate_trajectory():
    batch = generate_batch(seed=7, n_trajectories=5, n_steps=N_STEPS)
    assert len(batch) == 5
    for i, traj in enumerate(batch):
        assert traj == generate_trajectory(7, i, N_STEPS)


def test_trajectory_length_and_shared_sequential_state_id():
    for traj in _trajectories():
        assert len(traj) == N_STEPS
        ids = {ep.sequential_state_id for ep in traj}
        assert len(ids) == 1
        episode_ids = {ep.episode_id for ep in traj}
        assert len(episode_ids) == N_STEPS  # every step has a distinct episode_id


def test_h_and_u_are_constant_across_the_whole_trajectory():
    for traj in _trajectories():
        h_values = {ep.full_state["h"] for ep in traj}
        u_values = {ep.metadata["u"] for ep in traj}
        assert len(h_values) == 1
        assert len(u_values) == 1
        assert all(ep.hidden_state == {"h": "UNKNOWN"} for ep in traj)  # never re-stated as known


def test_transition_r_next_equals_f_current():
    # The exact, generator-known dynamic: r_{t+1} = f_t.
    for traj in _trajectories():
        for t in range(len(traj) - 1):
            assert traj[t + 1].true_quotient_state["r"] == traj[t].true_quotient_state["f"]


def test_only_t0_needs_grounding():
    for traj in _trajectories():
        assert traj[0].needs_grounding is True
        assert traj[0].min_grounding_bits == 1.0
        assert traj[0].minimal_query_set == ["h"]
        assert traj[0].correct_action == "ASK_h"
        assert traj[0].metadata["grounded"] is False
        for ep in traj[1:]:
            assert ep.needs_grounding is False
            assert ep.min_grounding_bits == 0.0
            assert ep.minimal_query_set == []
            assert ep.metadata["grounded"] is True


def test_correct_action_after_grounding_matches_true_decision():
    for traj in _trajectories():
        for ep in traj[1:]:
            r, h = ep.full_state["r"], ep.full_state["h"]
            expected = "ACT_B" if (r == "YES" and h == "YES") else "ACT_A"
            assert ep.metadata["ground_truth_decision"] == expected
            assert ep.correct_action == expected


def test_distortion_profile_every_step():
    for traj in _trajectories():
        for ep in traj:
            assert set(ep.distortion_profile) == {"ACT_A", "ACT_B", "ASK_h"}
            assert ep.distortion_profile["ASK_h"] == 0.0
            gt = ep.metadata["ground_truth_decision"]
            assert ep.distortion_profile[gt] == 0.0
            other = "ACT_A" if gt == "ACT_B" else "ACT_B"
            assert ep.distortion_profile[other] == 1.0


def test_quotient_excludes_persistent_and_random_irrelevant_variables():
    for traj in _trajectories():
        for ep in traj:
            assert set(ep.true_quotient_state) == {"r", "f", "h"}
            assert "u" not in ep.true_quotient_state
            assert "n" not in ep.true_quotient_state
            # u/n are still recorded (for the quotient-probe irrelevant-variable
            # baseline, spec section 10.6) -- just outside the quotient itself.
            assert "u" in ep.metadata
            assert "n" in ep.metadata


def test_n_is_fresh_noise_not_constant():
    # A generator bug that accidentally held n_t constant (like u_t) would be
    # invisible to per-step tests; check variation across steps directly.
    for traj in _trajectories():
        if len({ep.metadata["n"] for ep in traj}) > 1:
            return
    pytest.fail("n_t never varied within any trajectory across all seeds -- looks constant, not noise")


# --- split isolation ---------------------------------------------------------


def test_seed_pools_are_disjoint_and_seed_actually_matters():
    in_dist = [generate_trajectory(s, 0, N_STEPS) for s in range(0, 20)]
    heldout = [generate_trajectory(s, 0, N_STEPS) for s in range(10_000, 10_020)]
    in_ids = {traj[0].sequential_state_id for traj in in_dist}
    held_ids = {traj[0].sequential_state_id for traj in heldout}
    assert in_ids.isdisjoint(held_ids)

    def signature(trajs):
        # h is constant within a trajectory (only 2 possible values), so it
        # alone isn't enough entropy to tell two 20-trajectory pools apart;
        # use the full (r, f, u, n) sequence instead.
        return {tuple((ep.full_state["r"], ep.full_state["f"], ep.metadata["u"], ep.metadata["n"]) for ep in traj) for traj in trajs}

    assert signature(in_dist) != signature(heldout)
