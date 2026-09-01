"""Exhaustive ground-truth tests for the Family B (Accessibility/Scaffolding) generator."""

from __future__ import annotations

import pytest

from hg_native.data.accessibility import (
    ACCESSIBLE,
    SUBCLASSES,
    generate_batch,
    generate_episode,
    render_text,
)

SEEDS = (0, 1, 12345)
N_PER_SEED = 150


def _cases(subclass: str):
    for seed in SEEDS:
        for index in range(N_PER_SEED):
            yield generate_episode(seed, subclass, index)


@pytest.mark.parametrize("subclass", SUBCLASSES)
def test_deterministic(subclass):
    for seed in SEEDS:
        assert generate_episode(seed, subclass, 3) == generate_episode(seed, subclass, 3)


@pytest.mark.parametrize("subclass", SUBCLASSES)
def test_generate_batch_matches_generate_episode(subclass):
    batch = generate_batch(seed=7, subclass=subclass, n_items=10)
    assert len(batch) == 10
    for i, ep in enumerate(batch):
        assert ep == generate_episode(7, subclass, i)


def test_unknown_subclass_rejected():
    with pytest.raises(ValueError, match="unknown accessibility subclass"):
        generate_episode(0, "not_a_subclass", 0)


@pytest.mark.parametrize("subclass", SUBCLASSES)
def test_zero_information_deficit_always(subclass):
    # Family B is defined to be information-closed: no hidden variable, ever.
    for ep in _cases(subclass):
        assert ep.hidden_state == {}
        assert ep.needs_grounding is False
        assert ep.min_grounding_bits == 0.0
        assert ep.minimal_query_set == []
        assert "ASK" not in "".join(ep.available_actions)


@pytest.mark.parametrize("subclass", SUBCLASSES)
def test_distortion_profile_shape(subclass):
    for ep in _cases(subclass):
        assert set(ep.available_actions) == {"ACT_A", "ACT_B", "THINK"}
        assert set(ep.distortion_profile) == {"ACT_A", "ACT_B", "THINK"}
        assert ep.distortion_profile["THINK"] == 0.0
        gt = ep.metadata["ground_truth_decision"]
        assert ep.distortion_profile[gt] == 0.0
        other = "ACT_A" if gt == "ACT_B" else "ACT_B"
        assert ep.distortion_profile[other] == 1.0


def test_information_set_identical_across_every_subclass():
    # Spec section 5.2: "all variants must preserve the underlying
    # information set" -- for a fixed (seed, index), every subclass's
    # full_state (and hence ground_truth_decision) must be byte-identical.
    for seed in SEEDS:
        for index in range(20):
            states = {sc: generate_episode(seed, sc, index).full_state for sc in SUBCLASSES}
            decisions = {sc: generate_episode(seed, sc, index).metadata["ground_truth_decision"] for sc in SUBCLASSES}
            first_state = next(iter(states.values()))
            for sc, s in states.items():
                assert s == first_state, f"{sc} diverged in full_state at seed={seed} index={index}"
            assert len(set(decisions.values())) == 1


@pytest.mark.parametrize("subclass", sorted(ACCESSIBLE))
def test_accessible_subclasses_act_directly(subclass):
    for ep in _cases(subclass):
        assert ep.correct_action == ep.metadata["ground_truth_decision"]
        assert ep.correct_action in ("ACT_A", "ACT_B")
        assert ep.metadata["accessibility_deficit"] is False
        assert ep.scaffold_variants == []


@pytest.mark.parametrize("subclass", sorted(set(SUBCLASSES) - ACCESSIBLE))
def test_inaccessible_subclasses_need_think(subclass):
    for ep in _cases(subclass):
        assert ep.correct_action == "THINK"
        assert ep.metadata["accessibility_deficit"] is True
        assert len(ep.scaffold_variants) == 1
        assert ep.scaffold_variants[0]["kind"] == "canonical"
        # The attached canonical scaffold must itself state the correct totals.
        assert f"A.cost = {ep.full_state['A.cost']}" in ep.scaffold_variants[0]["text"]
        assert f"B.cost = {ep.full_state['B.cost']}" in ep.scaffold_variants[0]["text"]


# --- render_text spot checks, one per subclass's distinguishing feature ----


def test_render_canonical_compact_shows_totals_only():
    ep = generate_episode(0, "canonical_compact", 0)
    text = render_text(ep)
    assert f"A.cost = {ep.full_state['A.cost']}" in text
    assert f"B.cost = {ep.full_state['B.cost']}" in text
    assert "STEP" not in text and "CONSTRAINT" not in text


def test_render_explicit_derived_comparison_shows_arithmetic():
    ep = generate_episode(0, "explicit_derived_comparison", 0)
    text = render_text(ep)
    assert " = " in text
    assert str(ep.full_state["A.base"]) in text and str(ep.full_state["A.adj"]) in text


def test_render_minimal_explanatory_scaffold_has_note():
    ep = generate_episode(0, "minimal_explanatory_scaffold", 0)
    assert "Note:" in render_text(ep)


def test_render_reordered_form_has_no_totals():
    ep = generate_episode(0, "reordered_form", 0)
    text = render_text(ep)
    assert f"A.cost = {ep.full_state['A.cost']}" not in text
    assert f"B.cost = {ep.full_state['B.cost']}" not in text
    assert "B.base" in text and "A.base" in text


def test_render_irrelevant_constraints_block_has_constraints():
    ep = generate_episode(0, "irrelevant_constraints_block", 0)
    assert "CONSTRAINT" in render_text(ep)


def test_render_redundant_restatement_repeats_the_fact():
    ep = generate_episode(0, "redundant_restatement", 0)
    assert "In other words" in render_text(ep)


def test_render_chain_decomposed_form_has_steps():
    ep = generate_episode(0, "chain_decomposed_form", 0)
    text = render_text(ep)
    assert "STEP 1" in text and "STEP 2" in text


def test_render_misleading_distractor_scaffold_has_distractor():
    ep = generate_episode(0, "misleading_distractor_scaffold", 0)
    assert "Distractor:" in render_text(ep)


def test_render_starts_and_ends_with_task_markers():
    for subclass in SUBCLASSES:
        text = render_text(generate_episode(0, subclass, 0))
        assert text.startswith("TASK ACCESSIBILITY-CHOICE")
        assert text.endswith("DECIDE OR THINK TO NORMALIZE.")


# --- split isolation (mirrors static_hidden's) ------------------------------


def test_seed_pools_are_disjoint_and_seed_actually_matters():
    in_dist_seeds = range(0, 20)
    heldout_seeds = range(10_000, 10_020)

    def episodes_for(seeds):
        return [generate_episode(s, "canonical_compact", 0) for s in seeds]

    in_eps = episodes_for(in_dist_seeds)
    held_eps = episodes_for(heldout_seeds)
    assert {e.episode_id for e in in_eps}.isdisjoint({e.episode_id for e in held_eps})

    def signature(eps):
        return {(e.full_state["A.cost"], e.full_state["B.cost"]) for e in eps}

    assert signature(in_eps) != signature(held_eps)
