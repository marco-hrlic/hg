"""M1 kickoff section 6 OOD split machinery -- Family A (all 4 axes),
Family B and Family C (parameter only). Family D and the other axes for
Families B/C are follow-up slices or (for Family B/C's lexical/distractor
axes) a deliberate non-fit -- see hg_native/data/ood.py's module
docstring."""

from __future__ import annotations

import pytest

from hg_native.data import accessibility, sequential_quotient, static_hidden
from hg_native.data.ood import (
    ACCESSIBILITY_IID_ADJ_RANGE,
    ACCESSIBILITY_IID_BASE_RANGE,
    ACCESSIBILITY_PARAMETER_OOD_ADJ_RANGE,
    ACCESSIBILITY_PARAMETER_OOD_BASE_RANGE,
    DISTRACTOR_OOD_RANGE,
    IID_COST_RANGE,
    PARAMETER_OOD_COST_RANGE,
    SEQUENTIAL_QUOTIENT_IID_N_STEPS_RANGE,
    SEQUENTIAL_QUOTIENT_PARAMETER_OOD_N_STEPS_RANGE,
    STRUCTURAL_OOD_CONDITION,
    STRUCTURAL_TRAIN_CONDITIONS,
    accessibility_iid_pool,
    accessibility_parameter_ood_pool,
    assert_disjoint_ranges,
    assert_no_shared_composition,
    assert_no_shared_lexical_templates,
    distractor_ood_pool,
    iid_pool,
    lexical_ood_pool,
    parameter_ood_pool,
    sequential_quotient_iid_pool,
    sequential_quotient_parameter_ood_pool,
    structural_iid_pool,
    structural_ood_pool,
)
from hg_native.data.rule_expr import And, Var

SUBCLASSES_WITH_A_RULE = ("one_bit_deficit", "no_deficit")  # a couple representative subclasses


# --- lexical OOD -----------------------------------------------------------


def test_lexical_ood_renders_different_text_same_ground_truth():
    ep = static_hidden.generate_episode(0, "one_bit_deficit", 0)
    canonical = static_hidden.render_text(ep, surface_form="canonical")
    lexical = static_hidden.render_text(ep, surface_form="lexical_ood")
    assert canonical != lexical
    # ground truth is per-episode data, untouched by surface_form
    assert ep.visible_state and ep.correct_action and ep.distortion_profile


def test_lexical_ood_default_surface_form_is_canonical():
    ep = static_hidden.generate_episode(0, "no_deficit", 0)
    assert static_hidden.render_text(ep) == static_hidden.render_text(ep, surface_form="canonical")


def test_lexical_ood_rejects_unknown_surface_form():
    ep = static_hidden.generate_episode(0, "no_deficit", 0)
    with pytest.raises(ValueError):
        static_hidden.render_text(ep, surface_form="nonsense")


def test_assert_no_shared_lexical_templates_passes_on_the_real_renderers():
    episodes = static_hidden.generate_batch(0, "one_bit_deficit", 30)
    canonical_texts = [static_hidden.render_text(ep, surface_form="canonical") for ep in episodes]
    lexical_texts = [static_hidden.render_text(ep, surface_form="lexical_ood") for ep in episodes]
    # rule_text is deliberately shared across surface forms -- it's the
    # substantive rule statement, not decorative wording (see
    # static_hidden.render_text's docstring / ood.py's own docstring).
    rule_texts = frozenset(ep.metadata["rule_text"] for ep in episodes)
    assert_no_shared_lexical_templates(canonical_texts, lexical_texts, allowed_shared=rule_texts)  # must not raise


def test_assert_no_shared_lexical_templates_still_catches_a_leak_beyond_allowed_shared():
    episodes = static_hidden.generate_batch(0, "one_bit_deficit", 30)
    canonical_texts = [static_hidden.render_text(ep, surface_form="canonical") for ep in episodes]
    lexical_texts = [static_hidden.render_text(ep, surface_form="lexical_ood") for ep in episodes]
    rule_texts = frozenset(ep.metadata["rule_text"] for ep in episodes)
    # A genuine scaffold leak (not covered by allowed_shared) must still raise.
    leaking = [t + "\nTASK STATIC-CHOICE" for t in lexical_texts]
    with pytest.raises(ValueError, match="TASK STATIC-CHOICE"):
        assert_no_shared_lexical_templates(canonical_texts, leaking, allowed_shared=rule_texts)


def test_assert_no_shared_lexical_templates_catches_a_real_leak():
    train_texts = ["HEADER\nA.cost = 3\nFOOTER"] * 5
    ood_texts = ["HEADER\nA.cost = 9\nOTHER"] * 5  # "HEADER" recurs in both pools
    with pytest.raises(ValueError, match="HEADER"):
        assert_no_shared_lexical_templates(train_texts, ood_texts)


def test_assert_no_shared_lexical_templates_ignores_coincidental_single_line_overlap():
    # A data line that happens to match once in each pool (not recurring
    # within either pool) is not scaffold leakage.
    train_texts = ["HEADER\nA.cost = 42\nFOOTER_TRAIN"]
    ood_texts = ["A.cost = 42\nFOOTER_OOD"]
    assert_no_shared_lexical_templates(train_texts, ood_texts)  # must not raise


def test_lexical_ood_pool_tags_condition_metadata():
    episodes = lexical_ood_pool(seed=0, subclass="no_deficit", n_items=5)
    assert len(episodes) == 5
    assert all(ep.metadata["ood_condition"] == "lexical" for ep in episodes)


# --- parameter OOD -----------------------------------------------------------


def test_parameter_ood_default_cost_range_matches_pre_slice_behavior():
    # Regression: calling without cost_range must reproduce exactly the same
    # costs as before this parameter existed.
    ep_default = static_hidden.generate_episode(0, "one_bit_deficit", 0)
    ep_explicit_iid = static_hidden.generate_episode(0, "one_bit_deficit", 0, cost_range=IID_COST_RANGE)
    assert ep_default.visible_state == ep_explicit_iid.visible_state


def test_parameter_ood_costs_within_new_bounds_and_outside_iid_bounds():
    episodes = static_hidden.generate_batch(0, "one_bit_deficit", 50, cost_range=PARAMETER_OOD_COST_RANGE)
    lo, hi = PARAMETER_OOD_COST_RANGE
    iid_lo, iid_hi = IID_COST_RANGE
    for ep in episodes:
        for key in ("A.cost", "B.cost"):
            cost = ep.visible_state[key]
            assert lo <= cost <= hi
            assert not (iid_lo <= cost <= iid_hi)


def test_parameter_ood_decision_rule_unaffected_by_cost_range():
    # Same rule, different cost distribution: correct_action is still
    # exactly "cheaper option, gated by the rule's hidden bits".
    for ep in static_hidden.generate_batch(0, "no_deficit", 30, cost_range=PARAMETER_OOD_COST_RANGE):
        expected = "ACT_B" if ep.visible_state["B.cost"] < ep.visible_state["A.cost"] else "ACT_A"
        assert ep.correct_action == expected


def test_assert_disjoint_ranges_raises_on_overlap():
    with pytest.raises(ValueError):
        assert_disjoint_ranges((10, 99), (50, 150))


def test_assert_disjoint_ranges_passes_on_the_real_iid_ood_pair():
    assert_disjoint_ranges(IID_COST_RANGE, PARAMETER_OOD_COST_RANGE)  # must not raise


def test_parameter_ood_pool_tags_condition_metadata():
    episodes = parameter_ood_pool(seed=0, subclass="no_deficit", n_items=5)
    assert len(episodes) == 5
    assert all(ep.metadata["ood_condition"] == "parameter" for ep in episodes)


# --- distractor-density OOD --------------------------------------------------


def test_distractor_density_two_is_byte_identical_to_pre_slice_behavior():
    ep_default = static_hidden.generate_episode(0, "distractor_visible", 0)
    ep_explicit = static_hidden.generate_episode(0, "distractor_visible", 0, n_distractor_visible=2)
    assert ep_default.visible_state == ep_explicit.visible_state


def test_distractor_density_sweep_never_changes_the_decision():
    for n in (0, 1, 2, 5, 10, 20, 30):
        ep = static_hidden.generate_episode(0, "one_bit_deficit", 0, n_distractor_visible=n)
        assert set(ep.visible_state) - {"A.cost", "B.cost"} == set(static_hidden.DISTRACTOR_NAMES[:n])
        # decision/distortion ground truth must not depend on decoy count
        ep0 = static_hidden.generate_episode(0, "one_bit_deficit", 0, n_distractor_visible=0)
        assert ep.correct_action == ep0.correct_action
        assert ep.distortion_profile == ep0.distortion_profile


def test_distractor_density_rejects_pool_overflow():
    with pytest.raises(ValueError):
        static_hidden.generate_episode(0, "no_deficit", 0, n_distractor_visible=len(static_hidden.DISTRACTOR_NAMES) + 1)


def test_distractor_ood_pool_spans_the_declared_range_and_tags_metadata():
    episodes = distractor_ood_pool(seed=0, subclass="no_deficit", n_items=60)
    lo, hi = DISTRACTOR_OOD_RANGE
    counts = {len(ep.visible_state) - 2 for ep in episodes}  # minus A.cost, B.cost
    assert counts <= set(range(lo, hi + 1))
    assert len(counts) > 1  # actually varies, not a single fixed point
    assert all(ep.metadata["ood_condition"] == "distractor" for ep in episodes)


# --- iid_pool ----------------------------------------------------------------


def test_iid_pool_tags_condition_metadata():
    episodes = iid_pool(seed=0, subclass="no_deficit", n_items=5)
    assert len(episodes) == 5
    assert all(ep.metadata["ood_condition"] == "iid" for ep in episodes)


# --- Family B (accessibility): parameter OOD only -----------------------------


def test_accessibility_parameter_ood_default_range_matches_pre_slice_behavior():
    ep_default = accessibility.generate_episode(0, "canonical_compact", 0)
    ep_explicit_iid = accessibility.generate_episode(
        0, "canonical_compact", 0, base_range=ACCESSIBILITY_IID_BASE_RANGE, adj_range=ACCESSIBILITY_IID_ADJ_RANGE
    )
    assert ep_default.visible_state == ep_explicit_iid.visible_state


def test_accessibility_parameter_ood_components_within_new_bounds_and_outside_iid_bounds():
    episodes = accessibility.generate_batch(
        0, "canonical_compact", 50, base_range=ACCESSIBILITY_PARAMETER_OOD_BASE_RANGE, adj_range=ACCESSIBILITY_PARAMETER_OOD_ADJ_RANGE
    )
    base_lo, base_hi = ACCESSIBILITY_PARAMETER_OOD_BASE_RANGE
    adj_lo, adj_hi = ACCESSIBILITY_PARAMETER_OOD_ADJ_RANGE
    iid_base_lo, iid_base_hi = ACCESSIBILITY_IID_BASE_RANGE
    iid_adj_lo, iid_adj_hi = ACCESSIBILITY_IID_ADJ_RANGE
    for ep in episodes:
        for prefix in ("A", "B"):
            base = ep.visible_state[f"{prefix}.base"]
            adj = ep.visible_state[f"{prefix}.adj"]
            assert base_lo <= base <= base_hi
            assert not (iid_base_lo <= base <= iid_base_hi)
            assert adj_lo <= adj <= adj_hi
            assert not (iid_adj_lo <= adj <= iid_adj_hi)


def test_accessibility_parameter_ood_decision_rule_unaffected_by_range():
    for ep in accessibility.generate_batch(
        0, "canonical_compact", 30, base_range=ACCESSIBILITY_PARAMETER_OOD_BASE_RANGE, adj_range=ACCESSIBILITY_PARAMETER_OOD_ADJ_RANGE
    ):
        expected = "ACT_B" if ep.visible_state["B.cost"] < ep.visible_state["A.cost"] else "ACT_A"
        assert ep.metadata["ground_truth_decision"] == expected
        assert ep.correct_action == expected  # canonical_compact is ACCESSIBLE: no THINK needed


def test_accessibility_parameter_ood_pool_validates_disjointness_and_tags_metadata():
    episodes = accessibility_parameter_ood_pool(seed=0, subclass="canonical_compact", n_items=5)
    assert len(episodes) == 5
    assert all(ep.metadata["ood_condition"] == "parameter" for ep in episodes)


def test_accessibility_parameter_ood_pool_rejects_an_overlapping_range():
    with pytest.raises(ValueError):
        accessibility_parameter_ood_pool(seed=0, subclass="canonical_compact", n_items=5, base_range=(30, 60))


def test_accessibility_iid_pool_tags_condition_metadata():
    episodes = accessibility_iid_pool(seed=0, subclass="canonical_compact", n_items=5)
    assert len(episodes) == 5
    assert all(ep.metadata["ood_condition"] == "iid" for ep in episodes)


# --- Family C (sequential_quotient): parameter OOD only -----------------------


def test_sequential_quotient_iid_and_ood_ranges_are_disjoint():
    assert_disjoint_ranges(SEQUENTIAL_QUOTIENT_IID_N_STEPS_RANGE, SEQUENTIAL_QUOTIENT_PARAMETER_OOD_N_STEPS_RANGE)  # must not raise


def test_sequential_quotient_parameter_ood_pool_rejects_an_overlapping_range():
    with pytest.raises(ValueError):
        sequential_quotient_parameter_ood_pool(seed=0, n_trajectories=5, n_steps_range=(5, 8))


def test_sequential_quotient_iid_pool_n_steps_stays_within_declared_range_and_varies():
    episodes = sequential_quotient_iid_pool(seed=0, n_trajectories=60)
    lo, hi = SEQUENTIAL_QUOTIENT_IID_N_STEPS_RANGE
    trajectory_lengths = {}
    for ep in episodes:
        trajectory_lengths.setdefault(ep.sequential_state_id, 0)
        trajectory_lengths[ep.sequential_state_id] += 1
    assert set(trajectory_lengths.values()) <= set(range(lo, hi + 1))
    assert len(set(trajectory_lengths.values())) > 1  # actually varies
    assert all(ep.metadata["ood_condition"] == "iid" for ep in episodes)


def test_sequential_quotient_parameter_ood_pool_n_steps_stays_within_declared_range_and_varies():
    episodes = sequential_quotient_parameter_ood_pool(seed=0, n_trajectories=60)
    lo, hi = SEQUENTIAL_QUOTIENT_PARAMETER_OOD_N_STEPS_RANGE
    trajectory_lengths = {}
    for ep in episodes:
        trajectory_lengths.setdefault(ep.sequential_state_id, 0)
        trajectory_lengths[ep.sequential_state_id] += 1
    assert set(trajectory_lengths.values()) <= set(range(lo, hi + 1))
    assert len(set(trajectory_lengths.values())) > 1
    assert all(ep.metadata["ood_condition"] == "parameter" for ep in episodes)


def test_sequential_quotient_ood_pool_ground_truth_invariants_hold_regardless_of_length():
    episodes = sequential_quotient_parameter_ood_pool(seed=0, n_trajectories=20)
    by_trajectory: dict[str, list] = {}
    for ep in episodes:
        by_trajectory.setdefault(ep.sequential_state_id, []).append(ep)
    for steps in by_trajectory.values():
        steps.sort(key=lambda ep: ep.metadata["t"])
        assert steps[0].needs_grounding is True
        assert all(not ep.needs_grounding for ep in steps[1:])
        for prev, nxt in zip(steps, steps[1:]):
            assert nxt.true_quotient_state["r"] == prev.true_quotient_state["f"]


def test_sequential_quotient_iid_pool_matches_direct_generation_for_a_fixed_n_steps():
    # The pool's own n_steps draw is a separate RNG stream (see
    # ood._draw_n_steps) from sequential_quotient's internal draws, so for
    # whatever n_steps a given index actually gets, the trajectory itself
    # must be byte-identical to calling generate_trajectory directly.
    episodes = sequential_quotient_iid_pool(seed=0, n_trajectories=5)
    by_trajectory: dict[str, list] = {}
    for ep in episodes:
        by_trajectory.setdefault(ep.sequential_state_id, []).append(ep)
    for sid, steps in by_trajectory.items():
        index = int(sid.rsplit("-", 1)[-1])
        direct = sequential_quotient.generate_trajectory(0, index, len(steps))
        assert [ep.true_quotient_state for ep in sorted(steps, key=lambda e: e.metadata["t"])] == [
            ep.true_quotient_state for ep in direct
        ]


# --- Family A: structural OOD ------------------------------------------------


def test_assert_no_shared_composition_passes_on_the_real_train_ood_pair():
    assert_no_shared_composition(STRUCTURAL_TRAIN_CONDITIONS, (STRUCTURAL_OOD_CONDITION,))  # must not raise


def test_assert_no_shared_composition_raises_on_a_real_overlap():
    with pytest.raises(ValueError, match="shared rule compositions"):
        assert_no_shared_composition(STRUCTURAL_TRAIN_CONDITIONS, (And((Var("h1"),)),))  # exact match to a train condition


def test_structural_iid_pool_tags_condition_metadata():
    episodes = structural_iid_pool(seed=0, n_items=5)
    assert len(episodes) == 5
    assert all(ep.metadata["ood_condition"] == "iid" for ep in episodes)


def test_structural_ood_pool_tags_condition_metadata_and_validates_disjointness():
    episodes = structural_ood_pool(seed=0, n_items=5)
    assert len(episodes) == 5
    assert all(ep.metadata["ood_condition"] == "structural" for ep in episodes)
    assert all(ep.metadata["condition_signature"] == STRUCTURAL_OOD_CONDITION.signature() for ep in episodes)


def test_structural_ood_pool_rejects_a_condition_seen_in_training():
    with pytest.raises(ValueError):
        structural_ood_pool(seed=0, n_items=5, condition=STRUCTURAL_TRAIN_CONDITIONS[0])
