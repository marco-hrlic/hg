"""Structural OOD (M1 kickoff section 6.2) ground-truth tests --
`generate_structural_episode`/`generate_structural_batch`, a separate
additive path from the 6 named subclasses (untouched, see
test_hg_native_static_hidden.py for their own unaffected suite)."""

from __future__ import annotations

import pytest

from hg_native.data.rule_expr import And, Not, Or, Var
from hg_native.data.static_hidden import generate_structural_batch, generate_structural_episode

SEEDS = (0, 1, 12345)
N_PER_SEED = 60

SIMPLE_AND = And((Var("h1"),))
TWO_BIT_AND = And((Var("h1"), Var("h2")))
TWO_BIT_OR = Or((Var("h1"), Var("h2")))
NESTED = Or((And((Var("h1"),)), And((Var("h2"), Not(Var("h1"))))))


def _episodes(condition):
    for seed in SEEDS:
        for index in range(N_PER_SEED):
            yield generate_structural_episode(seed, index, condition)


@pytest.mark.parametrize("condition", [SIMPLE_AND, TWO_BIT_AND, TWO_BIT_OR, NESTED])
def test_deterministic(condition):
    for seed in SEEDS:
        assert generate_structural_episode(seed, 3, condition) == generate_structural_episode(seed, 3, condition)


def test_generate_batch_matches_generate_episode():
    batch = generate_structural_batch(seed=7, n_items=10, condition=TWO_BIT_OR)
    assert len(batch) == 10
    for i, ep in enumerate(batch):
        assert ep == generate_structural_episode(7, i, TWO_BIT_OR)


@pytest.mark.parametrize("condition", [SIMPLE_AND, TWO_BIT_AND, TWO_BIT_OR, NESTED])
def test_ground_truth_decision_matches_the_formula_exhaustively(condition):
    for ep in _episodes(condition):
        b_cheaper = ep.full_state["B.cost"] < ep.full_state["A.cost"]
        bits = {v: ep.full_state[v] for v in condition.free_vars()}
        expected = "ACT_B" if (b_cheaper and condition.evaluate(bits)) else "ACT_A"
        assert ep.metadata["ground_truth_decision"] == expected
        assert ep.distortion_profile[expected] == 0.0


def test_rule_vars_match_free_vars():
    for ep in _episodes(NESTED):
        assert set(ep.metadata["rule_vars"]) == {"h1", "h2"}
        assert set(ep.hidden_state) == {"h1", "h2"}


def test_single_var_condition_only_declares_one_rule_var():
    for ep in _episodes(SIMPLE_AND):
        assert ep.metadata["rule_vars"] == ["h1"]
        assert ep.min_grounding_bits == 1.0
        assert ep.correct_action == "ASK_h1"


def test_two_var_condition_declares_both_rule_vars_and_asks_the_first():
    for ep in _episodes(TWO_BIT_OR):
        assert ep.metadata["rule_vars"] == ["h1", "h2"]
        assert ep.min_grounding_bits == 2.0
        assert ep.correct_action == "ASK_h1"


def test_distortion_profile_resolves_exactly_the_asked_variable():
    for ep in _episodes(TWO_BIT_OR):
        b_cheaper = ep.full_state["B.cost"] < ep.full_state["A.cost"]
        # ASK_h1: h1 resolved to truth, h2 defaults to NO.
        resolved_h1 = "ACT_B" if (b_cheaper and TWO_BIT_OR.evaluate({"h1": ep.full_state["h1"], "h2": "NO"})) else "ACT_A"
        assert ep.distortion_profile["ASK_h1"] == (0.0 if resolved_h1 == ep.metadata["ground_truth_decision"] else 1.0)
        resolved_h2 = "ACT_B" if (b_cheaper and TWO_BIT_OR.evaluate({"h1": "NO", "h2": ep.full_state["h2"]})) else "ACT_A"
        assert ep.distortion_profile["ASK_h2"] == (0.0 if resolved_h2 == ep.metadata["ground_truth_decision"] else 1.0)


def test_condition_signature_recorded_in_metadata():
    for ep in _episodes(NESTED):
        assert ep.metadata["condition_signature"] == NESTED.signature()


def test_rejects_condition_with_no_free_vars():
    from hg_native.data.rule_expr import And as _And

    with pytest.raises(ValueError):
        generate_structural_episode(0, 0, _And(()))


def test_available_actions_match_rule_vars():
    for ep in _episodes(TWO_BIT_AND):
        assert set(ep.available_actions) == {"ACT_A", "ACT_B", "ASK_h1", "ASK_h2"}


def test_seed_pools_are_disjoint_and_seed_actually_matters():
    in_dist = [generate_structural_episode(s, 0, TWO_BIT_OR) for s in range(0, 20)]
    heldout = [generate_structural_episode(s, 0, TWO_BIT_OR) for s in range(10_000, 10_020)]
    in_ids = {ep.episode_id for ep in in_dist}
    held_ids = {ep.episode_id for ep in heldout}
    assert in_ids.isdisjoint(held_ids)
    assert [ep.full_state["h1"] for ep in in_dist] != [ep.full_state["h1"] for ep in heldout]


def test_different_conditions_produce_different_episode_ids():
    ep_and = generate_structural_episode(0, 0, SIMPLE_AND)
    ep_or = generate_structural_episode(0, 0, TWO_BIT_OR)
    assert ep_and.episode_id != ep_or.episode_id
