"""Exhaustive ground-truth tests for the Family A (Static Hidden Variable) generator.

Per spec run-order step 2 ("mechanically verify generator ground truth:
correct action, distortion profile, needs_grounding, B*, minimal query set"),
every check here re-derives the expected value directly from `full_state`
independently of the generator's internal helpers, rather than asserting
"it runs" or cross-checking against the module's own `_decide`/`_resolve_decision`.
"""

from __future__ import annotations

import pytest

from hg_native.data.static_hidden import (
    SUBCLASSES,
    generate_batch,
    generate_episode,
    render_text,
)

SEEDS = (0, 1, 12345)
N_PER_SEED = 200


def _cases(subclass: str):
    for seed in SEEDS:
        for index in range(N_PER_SEED):
            yield generate_episode(seed, subclass, index)


# --- determinism and basic shape, all subclasses ----------------------------


@pytest.mark.parametrize("subclass", SUBCLASSES)
def test_deterministic(subclass):
    for seed in SEEDS:
        a = generate_episode(seed, subclass, 3)
        b = generate_episode(seed, subclass, 3)
        assert a == b


@pytest.mark.parametrize("subclass", SUBCLASSES)
def test_generate_batch_matches_generate_episode(subclass):
    batch = generate_batch(seed=7, subclass=subclass, n_items=10)
    assert len(batch) == 10
    for i, ep in enumerate(batch):
        assert ep == generate_episode(7, subclass, i)


@pytest.mark.parametrize("subclass", SUBCLASSES)
def test_distortion_profile_covers_every_available_action(subclass):
    for ep in _cases(subclass):
        assert set(ep.distortion_profile) == set(ep.available_actions)
        assert all(v in (0.0, 1.0) for v in ep.distortion_profile.values())


def test_unknown_subclass_rejected():
    with pytest.raises(ValueError, match="unknown static_hidden subclass"):
        generate_episode(0, "not_a_subclass", 0)


# --- subclass 1: no_deficit --------------------------------------------------


def test_no_deficit_is_fully_determined_by_visible_costs():
    for ep in _cases("no_deficit"):
        assert ep.hidden_state == {}
        assert ep.needs_grounding is False
        assert ep.min_grounding_bits == 0.0
        assert ep.minimal_query_set == []
        expected = "ACT_B" if ep.full_state["B.cost"] < ep.full_state["A.cost"] else "ACT_A"
        assert ep.correct_action == expected
        assert ep.distortion_profile[expected] == 0.0
        other = "ACT_A" if expected == "ACT_B" else "ACT_B"
        assert ep.distortion_profile[other] == 1.0


# --- subclass 2: one_bit_deficit ---------------------------------------------


def test_one_bit_deficit_needs_exactly_one_bit():
    for ep in _cases("one_bit_deficit"):
        assert ep.full_state["B.cost"] < ep.full_state["A.cost"]  # forced by generator
        assert ep.needs_grounding is True
        assert ep.min_grounding_bits == 1.0
        assert ep.minimal_query_set == ["h1"]
        assert ep.correct_action == "ASK_h1"
        assert ep.available_actions == ["ACT_A", "ACT_B", "ASK_h1"]
        expected_decision = "ACT_B" if ep.full_state["h1"] == "YES" else "ACT_A"
        assert ep.metadata["ground_truth_decision"] == expected_decision
        # Asking the one required bit fully resolves the episode.
        assert ep.distortion_profile["ASK_h1"] == 0.0
        assert ep.distortion_profile[expected_decision] == 0.0
        wrong = "ACT_A" if expected_decision == "ACT_B" else "ACT_B"
        assert ep.distortion_profile[wrong] == 1.0


def test_one_bit_deficit_contrast_when_a_is_cheaper_would_need_no_deficit():
    # The spec's own worked contrast (B.cost=37 > A.cost=29 -> ACT_A, h3
    # irrelevant) cannot arise from one_bit_deficit's generator, which forces
    # B cheaper by construction -- it is exercised structurally instead via
    # the rule-text always stating the B.requires=h1 dependency regardless of
    # whether a given instance's hidden bit ends up mattering (see
    # `render_text`'s docstring). This test only pins the forced invariant.
    for ep in _cases("one_bit_deficit"):
        assert ep.full_state["B.cost"] < ep.full_state["A.cost"]


# --- subclass 3: multi_bit_deficit -------------------------------------------


def test_multi_bit_deficit_needs_exactly_two_bits():
    for ep in _cases("multi_bit_deficit"):
        assert ep.needs_grounding is True
        assert ep.min_grounding_bits == 2.0
        assert ep.minimal_query_set == ["h1", "h2"]
        assert ep.correct_action == "ASK_h1"  # first of minimal_query_set, by policy
        expected_decision = (
            "ACT_B" if ep.full_state["h1"] == "YES" and ep.full_state["h2"] == "YES" else "ACT_A"
        )
        assert ep.metadata["ground_truth_decision"] == expected_decision
        assert ep.metadata["fully_resolvable_in_one_ask"] is False


def test_multi_bit_deficit_partial_ask_distortion_is_computed_honestly():
    # Re-derive, independently of the module, what a single-shot agent would
    # decide after learning only h1 (defaulting the still-unknown h2 to NO),
    # and only h2 (defaulting h1 to NO) -- and confirm distortion_profile
    # matches, rather than optimistically treating any ask as fully resolving.
    for ep in _cases("multi_bit_deficit"):
        h1, h2 = ep.full_state["h1"], ep.full_state["h2"]
        truth = "ACT_B" if h1 == "YES" and h2 == "YES" else "ACT_A"

        after_ask_h1 = "ACT_B" if h1 == "YES" and "NO" == "YES" else "ACT_A"
        assert ep.distortion_profile["ASK_h1"] == (0.0 if after_ask_h1 == truth else 1.0)

        after_ask_h2 = "ACT_B" if "NO" == "YES" and h2 == "YES" else "ACT_A"
        assert ep.distortion_profile["ASK_h2"] == (0.0 if after_ask_h2 == truth else 1.0)


def test_multi_bit_deficit_single_ask_sometimes_fails_to_resolve():
    # Sanity check that the honest partial-resolution accounting in
    # `_resolve_decision` actually bites: across enough samples, asking only
    # one of the two required bits must sometimes still leave the wrong
    # decision (distortion 1.0) -- if it never did, the "single ask fully
    # resolves a multi-bit deficit" bug this generator deliberately avoids
    # would be invisible to these tests.
    n_ask_h1_fails = sum(1 for ep in _cases("multi_bit_deficit") if ep.distortion_profile["ASK_h1"] == 1.0)
    n_ask_h1_succeeds = sum(1 for ep in _cases("multi_bit_deficit") if ep.distortion_profile["ASK_h1"] == 0.0)
    assert n_ask_h1_fails > 0
    assert n_ask_h1_succeeds > 0


# --- subclass 4: redundant_hidden --------------------------------------------


def test_redundant_hidden_collapses_to_one_bit():
    for ep in _cases("redundant_hidden"):
        assert ep.needs_grounding is True
        assert ep.min_grounding_bits == 1.0  # not 2, despite two hidden vars present
        assert ep.minimal_query_set == ["h1"]
        assert set(ep.hidden_state) == {"h1", "h2"}
        # h2 is a hard-wired copy of h1.
        assert ep.full_state["h2"] == ep.full_state["h1"]
        # Either ask fully resolves the episode.
        assert ep.distortion_profile["ASK_h1"] == 0.0
        assert ep.distortion_profile["ASK_h2"] == 0.0
        expected_decision = "ACT_B" if ep.full_state["h1"] == "YES" else "ACT_A"
        assert ep.metadata["ground_truth_decision"] == expected_decision


# --- subclass 5: irrelevant_hidden --------------------------------------------


def test_irrelevant_hidden_has_zero_deficit_despite_unknown_variables():
    for ep in _cases("irrelevant_hidden"):
        assert set(ep.hidden_state) == {"h1", "h2"}
        assert all(v == "UNKNOWN" for v in ep.hidden_state.values())
        assert ep.needs_grounding is False
        assert ep.min_grounding_bits == 0.0
        assert ep.minimal_query_set == []
        expected = "ACT_B" if ep.full_state["B.cost"] < ep.full_state["A.cost"] else "ACT_A"
        assert ep.correct_action == expected
        # Asking either irrelevant variable cannot change the decision.
        assert ep.distortion_profile["ASK_h1"] == 0.0
        assert ep.distortion_profile["ASK_h2"] == 0.0


# --- subclass 6: distractor_visible ------------------------------------------


def test_distractor_visible_ignores_extra_visible_vars():
    for ep in _cases("distractor_visible"):
        assert {"C.cost", "A.weight"} <= set(ep.visible_state)
        assert ep.full_state["B.cost"] < ep.full_state["A.cost"]
        assert ep.needs_grounding is True
        assert ep.min_grounding_bits == 1.0
        assert ep.minimal_query_set == ["h1"]
        assert ep.correct_action == "ASK_h1"
        expected_decision = "ACT_B" if ep.full_state["h1"] == "YES" else "ACT_A"
        assert ep.metadata["ground_truth_decision"] == expected_decision
        assert ep.distortion_profile["ASK_h1"] == 0.0


# --- rendered text spot-check against the spec's own worked example ---------


def test_render_text_matches_spec_worked_example_shape():
    # Construct the exact instance from spec section 5.1's deficit example
    # (A.cost=37, B.cost=29, h1 unknown) by searching a small seed range --
    # the spec's costs aren't tied to any particular seed, so we assert on
    # structure/content rather than reproducing its literal seed.
    ep = generate_episode(seed=0, subclass="one_bit_deficit", index=0)
    text = render_text(ep)
    assert text.startswith("TASK STATIC-CHOICE")
    assert f"A.cost = {ep.visible_state['A.cost']}" in text
    assert f"B.cost = {ep.visible_state['B.cost']}" in text
    assert "B.requires = h1" in text
    assert "RULE:" in text
    assert "KNOWN:" in text
    assert "h1 = UNKNOWN" in text
    assert text.endswith("DECIDE OR REQUEST INFORMATION.")


def test_render_text_no_deficit_has_no_requires_line():
    ep = generate_episode(seed=0, subclass="no_deficit", index=0)
    text = render_text(ep)
    assert "B.requires" not in text


# --- split isolation (foreshadows spec section 17) --------------------------


def test_seed_pools_are_disjoint_and_seed_actually_matters():
    in_dist_seeds = range(0, 20)
    heldout_seeds = range(10_000, 10_020)

    def episodes_for(seeds):
        return [generate_episode(s, "one_bit_deficit", 0) for s in seeds]

    in_eps = episodes_for(in_dist_seeds)
    held_eps = episodes_for(heldout_seeds)

    in_ids = {e.episode_id for e in in_eps}
    held_ids = {e.episode_id for e in held_eps}
    assert in_ids.isdisjoint(held_ids)

    def signature(eps):
        return {(e.full_state["A.cost"], e.full_state["B.cost"], e.full_state["h1"]) for e in eps}

    # A generator that ignored `seed` would produce the identical set of
    # underlying instances from two disjoint seed pools.
    assert signature(in_eps) != signature(held_eps)
