"""Accessibility/scaffolding eval -- torch-free."""

from __future__ import annotations

import math

import pytest

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.eval.scaffolding import accessibility_discrimination, accessibility_labels_and_mask, route_rates_on_zero_deficit
from hg_native.model.config import DEFAULT_ROUTES

ACCESSIBLE = ("canonical_compact", "explicit_derived_comparison", "minimal_explanatory_scaffold")
INACCESSIBLE = ("reordered_form", "irrelevant_constraints_block", "redundant_restatement", "chain_decomposed_form", "misleading_distractor_scaffold")


def _accessibility_episodes(n=10):
    episodes = []
    for sc in ACCESSIBLE + INACCESSIBLE:
        episodes += [gen_accessibility(0, sc, i) for i in range(n)]
    return episodes


def test_accessibility_labels_and_mask_excludes_genuine_deficit():
    deficit_ep = gen_static_hidden(0, "one_bit_deficit", 0)
    accessible_ep = gen_accessibility(0, "canonical_compact", 0)
    labels, mask = accessibility_labels_and_mask([deficit_ep, accessible_ep])
    assert list(mask) == [False, True]
    assert labels[1] == 0.0


def test_accessibility_discrimination_perfect_oracle():
    episodes = _accessibility_episodes(20)
    scores = [float(ep.metadata["accessibility_deficit"]) for ep in episodes]
    result = accessibility_discrimination(episodes, scores)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["n_zero_deficit"] == len(episodes)


def test_accessibility_discrimination_empty_when_all_genuine_deficit():
    episodes = [gen_static_hidden(0, "one_bit_deficit", i) for i in range(5)]
    result = accessibility_discrimination(episodes, [0.5] * 5)
    assert result["n_zero_deficit"] == 0


def test_accessibility_discrimination_rejects_length_mismatch():
    episodes = _accessibility_episodes(5)
    with pytest.raises(ValueError):
        accessibility_discrimination(episodes, [0.5] * (len(episodes) - 1))


def test_route_rates_desired_behavior_think_over_ask():
    episodes = _accessibility_episodes(10)
    # Simulate a well-behaved router: ACT on accessible items, THINK on
    # inaccessible ones, never ASK (Family B never offers ASK anyway).
    act_idx, think_idx = DEFAULT_ROUTES.index("ACT"), DEFAULT_ROUTES.index("THINK")
    route_pred = []
    for ep in episodes:
        route_pred.append(think_idx if ep.metadata["accessibility_deficit"] else act_idx)
    result = route_rates_on_zero_deficit(episodes, route_pred)
    assert result["unnecessary_ask_rate"] == pytest.approx(0.0)
    assert result["think_rate_on_accessibility_deficit"] == pytest.approx(1.0)


def test_route_rates_empty_when_no_zero_deficit_episodes():
    episodes = [gen_static_hidden(0, "one_bit_deficit", i) for i in range(5)]
    result = route_rates_on_zero_deficit(episodes, [0] * 5)
    assert result["n"] == 0


def test_route_rates_rejects_length_mismatch():
    episodes = _accessibility_episodes(5)
    with pytest.raises(ValueError):
        route_rates_on_zero_deficit(episodes, [0] * (len(episodes) - 1))
