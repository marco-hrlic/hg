"""Exhaustive ground-truth tests for the Family D (Capability Refinement)
generator's M1 rework: Z=(r,f,u,n,v) under capability tokens C1/C2."""

from __future__ import annotations

from collections import defaultdict

from hg_native.data.capability_refinement import generate_batch, generate_pair

SEEDS = (0, 1, 12345)
N_PER_SEED = 100

EPISODE_KEYS = {
    f"{member}_t{t}_{capability}"
    for member in ("z", "z_prime")
    for t in (0, 1)
    for capability in ("C1", "C2")
}


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


def test_pair_has_eight_episodes_sharing_pair_id():
    for pair in _pairs():
        assert set(pair) == EPISODE_KEYS
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
        for key, ep in pair.items():
            capability = ep.metadata["capability"]
            expected = {"ACT_A", "ACT_B"} if capability == "C1" else {"ACT_A", "ACT_B", "ACT_C"}
            assert set(ep.available_actions) == expected


def test_true_quotient_state_matches_visible_state():
    # Zero-deficit family: nothing is hidden, so the quotient-probe ground
    # truth and what the model is actually shown must agree exactly.
    for pair in _pairs():
        for ep in pair.values():
            assert ep.true_quotient_state == ep.visible_state


def test_r_determines_correct_action_under_c1():
    for pair in _pairs():
        for key, ep in pair.items():
            if ep.metadata["capability"] != "C1":
                continue
            expected = "ACT_B" if ep.true_quotient_state["r"] == "YES" else "ACT_A"
            assert ep.correct_action == expected, key


def test_v_irrelevant_under_c1_z_and_zprime_agree():
    # z (v=YES) and z' (v=NO) share r/f/u/n at each t by construction, so
    # under C1 -- where v has no effect -- their correct_action must match.
    for pair in _pairs():
        for t in (0, 1):
            z = pair[f"z_t{t}_C1"]
            zp = pair[f"z_prime_t{t}_C1"]
            assert z.correct_action == zp.correct_action


def test_v_relevant_under_c2_z_and_zprime_diverge():
    for pair in _pairs():
        for t in (0, 1):
            z = pair[f"z_t{t}_C2"]
            zp = pair[f"z_prime_t{t}_C2"]
            c1_action = pair[f"z_t{t}_C1"].correct_action
            assert z.correct_action == "ACT_C"  # v=YES: ACT_C always wins under C2
            assert zp.correct_action == c1_action  # v=NO: C2 collapses back to C1's rule
            assert z.correct_action != zp.correct_action


def test_u_never_affects_correct_action():
    # u/n are drawn randomly (not directly controllable via the public API),
    # so verify structurally: group every episode by everything that *is*
    # allowed to matter (capability, t, r, v) and confirm correct_action
    # never varies within a group despite u/n differing across the batch.
    by_condition: dict[tuple, set[str]] = defaultdict(set)
    for pair in _pairs():
        for ep in pair.values():
            key = (ep.metadata["capability"], ep.metadata["t"], ep.true_quotient_state["r"], ep.true_quotient_state["v"])
            by_condition[key].add(ep.correct_action)
    assert len(by_condition) == 2 * 2 * 2 * 2  # capability x t x r x v, all combinations observed
    for key, actions in by_condition.items():
        assert len(actions) == 1, (key, actions)


def test_transition_r1_equals_f0():
    for pair in _pairs():
        for member in ("z", "z_prime"):
            t0 = pair[f"{member}_t0_C1"]
            t1 = pair[f"{member}_t1_C1"]
            assert t1.true_quotient_state["r"] == t0.true_quotient_state["f"]


def test_u_and_v_constant_across_t_within_a_member():
    for pair in _pairs():
        for member in ("z", "z_prime"):
            t0 = pair[f"{member}_t0_C1"]
            t1 = pair[f"{member}_t1_C1"]
            assert t0.true_quotient_state["u"] == t1.true_quotient_state["u"]
            assert t0.true_quotient_state["v"] == t1.true_quotient_state["v"]


def test_distortion_profile_consistent_with_correct_action():
    for pair in _pairs():
        for ep in pair.values():
            assert set(ep.distortion_profile) == set(ep.available_actions)
            assert ep.distortion_profile[ep.correct_action] == 0.0
            assert all(v == 1.0 for a, v in ep.distortion_profile.items() if a != ep.correct_action)


def test_sequential_state_id_distinguishes_members_shares_within_member():
    for pair in _pairs():
        assert pair["z_t0_C1"].sequential_state_id == pair["z_t1_C1"].sequential_state_id
        assert pair["z_t0_C1"].sequential_state_id != pair["z_prime_t0_C1"].sequential_state_id


def test_seed_pools_are_disjoint_and_seed_actually_matters():
    in_dist = [generate_pair(s, 0) for s in range(0, 20)]
    heldout = [generate_pair(s, 0) for s in range(10_000, 10_020)]
    in_ids = {p["z_t0_C1"].metadata["pair_id"] for p in in_dist}
    held_ids = {p["z_t0_C1"].metadata["pair_id"] for p in heldout}
    assert in_ids.isdisjoint(held_ids)

    def signature(pairs):
        # Ordered, not a set: r/f are single bits, so a *set* of only 20
        # draws saturates to all 4 (r, f) combinations from either pool and
        # falsely looks identical. The full ordered sequence is what
        # actually distinguishes two independent RNG streams.
        return tuple((p["z_t0_C1"].true_quotient_state["r"], p["z_t0_C1"].true_quotient_state["f"]) for p in pairs)

    assert signature(in_dist) != signature(heldout)
