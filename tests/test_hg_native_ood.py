"""M1 kickoff section 6 OOD split machinery -- Family A only (lexical,
parameter, distractor-density). Structural OOD and other families are
follow-up slices, see hg_native/data/ood.py's module docstring."""

from __future__ import annotations

import pytest

from hg_native.data import static_hidden
from hg_native.data.ood import (
    DISTRACTOR_OOD_RANGE,
    IID_COST_RANGE,
    PARAMETER_OOD_COST_RANGE,
    assert_disjoint_ranges,
    assert_no_shared_lexical_templates,
    distractor_ood_pool,
    iid_pool,
    lexical_ood_pool,
    parameter_ood_pool,
)

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
