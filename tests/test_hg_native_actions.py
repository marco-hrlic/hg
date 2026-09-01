"""The canonical action vocabulary shared across all four generators."""

from __future__ import annotations

import pytest

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.capability_refinement import generate_pair
from hg_native.data.sequential_quotient import generate_trajectory
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.training.actions import ACTIONS, action_id, action_name, route_of


def test_action_id_round_trips():
    for i, name in enumerate(ACTIONS):
        assert action_id(name) == i
        assert action_name(i) == name


def test_action_id_rejects_unknown():
    with pytest.raises(ValueError, match="not in the canonical ACTIONS"):
        action_id("ACT_Z")


def test_route_of():
    assert route_of("ACT_A") == "ACT"
    assert route_of("ACT_C") == "ACT"
    assert route_of("THINK") == "THINK"
    assert route_of("ASK_h1") == "ASK"
    assert route_of("ASK_h") == "ASK"


def test_route_of_rejects_unknown():
    with pytest.raises(ValueError, match="does not map to a known route"):
        route_of("FROBNICATE")


@pytest.mark.parametrize(
    "family_actions",
    [
        {"ACT_A", "ACT_B", "ASK_h1"},  # static_hidden one_bit_deficit
        {"ACT_A", "ACT_B", "ASK_h1", "ASK_h2"},  # static_hidden multi_bit_deficit / redundant_hidden
        {"ACT_A", "ACT_B", "THINK"},  # accessibility
        {"ACT_A", "ACT_B", "ASK_h"},  # sequential_quotient
        {"ACT_A", "ACT_B", "ACT_C"},  # capability_refinement (C2)
    ],
)
def test_every_generator_action_is_in_the_canonical_vocabulary(family_actions):
    assert family_actions <= set(ACTIONS)


def test_actions_vocabulary_covers_every_generator_output_directly():
    # A direct check against live generator output, not just the hand-picked
    # sets above: every action any generator actually produces must resolve.
    all_actions: set[str] = set()
    for ep in [gen_static_hidden(0, sc, 0) for sc in ("no_deficit", "one_bit_deficit", "multi_bit_deficit", "redundant_hidden", "irrelevant_hidden", "distractor_visible")]:
        all_actions |= set(ep.available_actions)
    all_actions |= set(gen_accessibility(0, "canonical_compact", 0).available_actions)
    all_actions |= set(generate_trajectory(0, 0, 3)[0].available_actions)
    for ep in generate_pair(0, 0).values():
        all_actions |= set(ep.available_actions)
    for name in all_actions:
        action_id(name)  # raises if unknown
