"""Exhaustive ground-truth tests for the Family D (Capability Refinement) generator."""

from __future__ import annotations

import pytest

from hg_native.data.capability_refinement import generate_batch, generate_pair

SEEDS = (0, 1, 12345)
N_PER_SEED = 100


def _pairs():
    for seed in SEEDS:
        for index in range(N_PER_SEED):
            yield generate_pair(seed, index)


def test_deterministic():
    for seed in SEEDS:
        assert generate_pair(seed, 3) == generate_pair(seed, 3)


def test_generate_batch_matches_generate_pair():
    batch = generate_batch(seed=7, n_pairs=5)
    assert len(batch) == 5
    for i, pair in enumerate(batch):
        assert pair == generate_pair(7, i)


def test_pair_has_four_episodes_sharing_pair_id():
    for pair in _pairs():
        assert set(pair) == {"z_C1", "z_C2", "z_prime_C1", "z_prime_C2"}
        pair_ids = {ep.metadata["pair_id"] for ep in pair.values()}
        assert len(pair_ids) == 1


def test_zero_information_deficit():
    for pair in _pairs():
        for ep in pair.values():
            assert ep.hidden_state == {}
            assert ep.needs_grounding is False
            assert ep.min_grounding_bits == 0.0


def test_c1_has_two_actions_c2_has_three():
    for pair in _pairs():
        assert set(pair["z_C1"].available_actions) == {"ACT_A", "ACT_B"}
        assert set(pair["z_prime_C1"].available_actions) == {"ACT_A", "ACT_B"}
        assert set(pair["z_C2"].available_actions) == {"ACT_A", "ACT_B", "ACT_C"}
        assert set(pair["z_prime_C2"].available_actions) == {"ACT_A", "ACT_B", "ACT_C"}


def test_underlying_costs_identical_between_capability_tokens_for_the_same_member():
    # Only the *available action set* should change between C1/C2 for a given
    # member -- the underlying state (A.cost, B.cost, C.cost) must not.
    for pair in _pairs():
        assert pair["z_C1"].full_state == pair["z_C2"].full_state
        assert pair["z_prime_C1"].full_state == pair["z_prime_C2"].full_state


def test_z_and_zprime_equivalent_under_c1():
    for pair in _pairs():
        assert pair["z_C1"].correct_action == "ACT_A"
        assert pair["z_prime_C1"].correct_action == "ACT_A"


def test_z_and_zprime_diverge_under_c2():
    for pair in _pairs():
        assert pair["z_C2"].correct_action == "ACT_C"  # z: C is cheapest once available
        assert pair["z_prime_C2"].correct_action == "ACT_A"  # z': C never competitive


def test_a_is_always_cheaper_than_b():
    # The generator's own invariant that makes the C1 equivalence well-defined.
    for pair in _pairs():
        for ep in pair.values():
            assert ep.full_state["A.cost"] < ep.full_state["B.cost"]


def test_distortion_profile_consistent_with_correct_action():
    for pair in _pairs():
        for ep in pair.values():
            assert set(ep.distortion_profile) == set(ep.available_actions)
            assert ep.distortion_profile[ep.correct_action] == 0.0
            assert all(v == 1.0 for a, v in ep.distortion_profile.items() if a != ep.correct_action)


def test_seed_pools_are_disjoint_and_seed_actually_matters():
    in_dist = [generate_pair(s, 0) for s in range(0, 20)]
    heldout = [generate_pair(s, 0) for s in range(10_000, 10_020)]
    in_ids = {p["z_C1"].metadata["pair_id"] for p in in_dist}
    held_ids = {p["z_C1"].metadata["pair_id"] for p in heldout}
    assert in_ids.isdisjoint(held_ids)

    def signature(pairs):
        return {(p["z_C1"].full_state["A.cost"], p["z_C1"].full_state["C.cost"]) for p in pairs}

    assert signature(in_dist) != signature(heldout)
