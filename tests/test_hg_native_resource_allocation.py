"""Exhaustive ground-truth tests for Family E (Resource Allocation /
Scheduling, M1 kickoff section 7) -- mirrors
test_hg_native_sequential_quotient.py's coverage, since this generator
deliberately reuses that mechanism (see resource_allocation.py's module
docstring); field names here reflect the renamed display vocabulary
(`full_state["capacity"]` etc.) except `true_quotient_state`/`metadata`,
which keep sequential_quotient's r/f/h/u/n keys on purpose."""

from __future__ import annotations

import pytest

from hg_native.data.resource_allocation import generate_batch, generate_trajectory, render_text

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
        assert len(episode_ids) == N_STEPS


def test_capacity_and_priority_site_constant_across_the_whole_trajectory():
    for traj in _trajectories():
        capacity_values = {ep.full_state["capacity"] for ep in traj}
        site_values = {ep.metadata["u"] for ep in traj}
        assert len(capacity_values) == 1
        assert len(site_values) == 1
        assert all(ep.hidden_state == {"capacity": "UNKNOWN"} for ep in traj)


def test_transition_r_next_equals_f_current():
    for traj in _trajectories():
        for t in range(len(traj) - 1):
            assert traj[t + 1].true_quotient_state["r"] == traj[t].true_quotient_state["f"]


def test_only_t0_needs_grounding():
    for traj in _trajectories():
        assert traj[0].needs_grounding is True
        assert traj[0].min_grounding_bits == 1.0
        assert traj[0].minimal_query_set == ["capacity"]
        assert traj[0].correct_action == "ASK_capacity"
        assert traj[0].metadata["grounded"] is False
        for ep in traj[1:]:
            assert ep.needs_grounding is False
            assert ep.min_grounding_bits == 0.0
            assert ep.minimal_query_set == []
            assert ep.metadata["grounded"] is True


def test_correct_action_after_grounding_matches_true_decision():
    for traj in _trajectories():
        for ep in traj[1:]:
            urgent, capacity = ep.true_quotient_state["r"], ep.true_quotient_state["h"]
            expected = "ALLOCATE_B" if (urgent == "YES" and capacity == "YES") else "ALLOCATE_A"
            assert ep.metadata["ground_truth_decision"] == expected
            assert ep.correct_action == expected


def test_distortion_profile_every_step():
    for traj in _trajectories():
        for ep in traj:
            assert set(ep.distortion_profile) == {"ALLOCATE_A", "ALLOCATE_B", "ASK_capacity"}
            assert ep.distortion_profile["ASK_capacity"] == 0.0
            gt = ep.metadata["ground_truth_decision"]
            assert ep.distortion_profile[gt] == 0.0
            other = "ALLOCATE_A" if gt == "ALLOCATE_B" else "ALLOCATE_B"
            assert ep.distortion_profile[other] == 1.0


def test_quotient_state_uses_sequential_quotient_key_names():
    # The whole point of the reused-mechanism design: r/f/h, not
    # urgent/future_urgent/capacity, so existing eval functions work
    # unmodified.
    for traj in _trajectories():
        for ep in traj:
            assert set(ep.true_quotient_state) == {"r", "f", "h"}
            assert "u" in ep.metadata
            assert "n" in ep.metadata


def test_n_is_fresh_noise_not_constant():
    for traj in _trajectories():
        if len({ep.metadata["n"] for ep in traj}) > 1:
            return
    pytest.fail("n_t never varied within any trajectory across all seeds -- looks constant, not noise")


def test_t0_prompt_never_discloses_capacity():
    for traj in _trajectories():
        assert "ANSWER=capacity:" not in render_text(traj[0])
        assert "capacity=UNKNOWN" in render_text(traj[0])


def test_later_prompts_disclose_the_true_capacity_value():
    for traj in _trajectories():
        capacity_true = traj[0].full_state["capacity"]
        for ep in traj[1:]:
            assert f"ANSWER=capacity:{capacity_true}" in render_text(ep)


def test_prompt_never_discloses_the_wrong_capacity_value():
    wrong = {"YES": "NO", "NO": "YES"}
    for traj in _trajectories():
        capacity_true = traj[0].full_state["capacity"]
        for ep in traj[1:]:
            assert f"ANSWER=capacity:{wrong[capacity_true]}" not in render_text(ep)


def test_prompt_grows_monotonically_with_t():
    traj = generate_trajectory(0, 0, N_STEPS)
    lengths = [len(render_text(ep)) for ep in traj]
    assert lengths == sorted(lengths)
    assert lengths[0] < lengths[-1]


def test_render_text_uses_scheduling_vocabulary_not_route_choice_wording():
    ep = generate_trajectory(0, 0, 1)[0]
    text = render_text(ep)
    assert "urgent" in text and "capacity" in text
    assert "A.cost" not in text and "TASK STATIC-CHOICE" not in text


def test_seed_pools_are_disjoint_and_seed_actually_matters():
    in_dist = [generate_trajectory(s, 0, N_STEPS) for s in range(0, 20)]
    heldout = [generate_trajectory(s, 0, N_STEPS) for s in range(10_000, 10_020)]
    in_ids = {traj[0].sequential_state_id for traj in in_dist}
    held_ids = {traj[0].sequential_state_id for traj in heldout}
    assert in_ids.isdisjoint(held_ids)

    def signature(trajs):
        return {
            tuple((ep.true_quotient_state["r"], ep.true_quotient_state["f"], ep.metadata["u"], ep.metadata["n"]) for ep in traj)
            for traj in trajs
        }

    assert signature(in_dist) != signature(heldout)
