"""Per-episode head target computation, across all four generator families."""

from __future__ import annotations

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.capability_refinement import generate_pair
from hg_native.data.sequential_quotient import generate_trajectory
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.model.config import DEFAULT_DECISION_LABELS, DEFAULT_ROUTES
from hg_native.training.actions import ACTIONS, action_id
from hg_native.training.targets import (
    accessibility_target,
    decision_target,
    distortion_target,
    info_deficit_target,
    route_target,
)


# --- decision_target ---------------------------------------------------


def test_decision_target_static_hidden_no_deficit():
    ep = gen_static_hidden(0, "no_deficit", 0)
    label, defined = decision_target(ep)
    assert defined is True
    assert DEFAULT_DECISION_LABELS[label] == ep.correct_action[-1]  # "ACT_A" -> "A"


def test_decision_target_static_hidden_one_bit_deficit_uses_ground_truth_not_ask():
    ep = gen_static_hidden(0, "one_bit_deficit", 0)
    assert ep.correct_action.startswith("ASK")
    label, defined = decision_target(ep)
    assert defined is True
    assert DEFAULT_DECISION_LABELS[label] == ep.metadata["ground_truth_decision"][-1]


def test_decision_target_capability_refinement_c2_z_is_undefined():
    pair = generate_pair(0, 0)
    assert pair["z_C2"].correct_action == "ACT_C"
    label, defined = decision_target(pair["z_C2"])
    assert defined is False


def test_decision_target_capability_refinement_c1_is_defined():
    pair = generate_pair(0, 0)
    label, defined = decision_target(pair["z_C1"])
    assert defined is True
    assert DEFAULT_DECISION_LABELS[label] == "A"


# --- distortion_target ---------------------------------------------------


def test_distortion_target_masks_absent_actions():
    ep = gen_static_hidden(0, "one_bit_deficit", 0)  # only ACT_A, ACT_B, ASK_h1
    vec, mask = distortion_target(ep)
    assert len(vec) == len(mask) == len(ACTIONS)
    for name in ACTIONS:
        idx = action_id(name)
        if name in ep.available_actions:
            assert mask[idx] is True
            assert vec[idx] == ep.distortion_profile[name]
        else:
            assert mask[idx] is False


def test_distortion_target_sequential_quotient_uses_ask_h():
    traj = generate_trajectory(0, 0, 4)
    vec, mask = distortion_target(traj[0])
    assert mask[action_id("ASK_h")] is True
    assert mask[action_id("ASK_h1")] is False


# --- info_deficit_target ---------------------------------------------------


def test_info_deficit_target_matches_needs_grounding():
    for ep in [gen_static_hidden(0, "one_bit_deficit", 0), gen_static_hidden(0, "no_deficit", 0)]:
        assert info_deficit_target(ep) == float(ep.needs_grounding)


# --- accessibility_target ---------------------------------------------------


def test_accessibility_target_undefined_when_grounding_needed():
    ep = gen_static_hidden(0, "one_bit_deficit", 0)
    assert ep.needs_grounding is True
    value, defined = accessibility_target(ep)
    assert defined is False


def test_accessibility_target_defined_for_accessibility_family():
    accessible_ep = gen_accessibility(0, "canonical_compact", 0)
    inaccessible_ep = gen_accessibility(0, "reordered_form", 0)
    v1, d1 = accessibility_target(accessible_ep)
    v2, d2 = accessibility_target(inaccessible_ep)
    assert d1 is True and v1 == 0.0
    assert d2 is True and v2 == 1.0


def test_accessibility_target_defined_zero_for_zero_deficit_outside_family_b():
    ep = gen_static_hidden(0, "no_deficit", 0)  # H(J|M)=0, but not Family B
    value, defined = accessibility_target(ep)
    assert defined is True
    assert value == 0.0


# --- route_target ---------------------------------------------------


def test_route_target_static_hidden_deficit_picks_ask():
    ep = gen_static_hidden(0, "one_bit_deficit", 0)  # no THINK available; ASK resolves, ACT risks distortion 1
    label, defined = route_target(ep)
    assert defined is True
    assert DEFAULT_ROUTES[label] == "ASK"


def test_route_target_static_hidden_no_deficit_picks_act():
    ep = gen_static_hidden(0, "no_deficit", 0)  # ACT is free and correct; no ASK/THINK available anyway
    label, defined = route_target(ep)
    assert DEFAULT_ROUTES[label] == "ACT"


def test_route_target_accessibility_inaccessible_picks_think():
    ep = gen_accessibility(0, "reordered_form", 0)  # ACT risks distortion 1; THINK is free and resolves
    label, defined = route_target(ep)
    assert DEFAULT_ROUTES[label] == "THINK"


def test_route_target_accessibility_canonical_picks_act():
    ep = gen_accessibility(0, "canonical_compact", 0)
    label, defined = route_target(ep)
    assert DEFAULT_ROUTES[label] == "ACT"


def test_route_target_capability_refinement_has_no_think_or_ask():
    # Family D never offers THINK/ASK; ACT is the only route representable.
    for ep in generate_pair(0, 0).values():
        label, defined = route_target(ep)
        assert defined is True
        assert DEFAULT_ROUTES[label] == "ACT"


def test_route_target_high_lambda_b_can_flip_ask_to_act():
    ep = gen_static_hidden(0, "one_bit_deficit", 0)
    label_cheap, _ = route_target(ep, lambda_b=0.1, lambda_k=1.0)
    label_expensive, _ = route_target(ep, lambda_b=10.0, lambda_k=1.0)
    assert DEFAULT_ROUTES[label_cheap] == "ASK"
    # At a high enough external-info price, risking the (at most) 1.0
    # distortion of acting blind beats paying for the ask.
    assert DEFAULT_ROUTES[label_expensive] == "ACT"
