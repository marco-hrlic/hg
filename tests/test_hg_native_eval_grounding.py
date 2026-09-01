"""Grounding discrimination eval -- torch-free (synthetic scores, real episodes)."""

from __future__ import annotations

import pytest

from hg_native.data.static_hidden import generate_episode
from hg_native.eval.grounding import grounding_auroc_improvement, grounding_discrimination, needs_grounding_labels

DEFICIT_SUBCLASSES = ("one_bit_deficit", "multi_bit_deficit", "redundant_hidden", "distractor_visible")
NO_DEFICIT_SUBCLASSES = ("no_deficit", "irrelevant_hidden")


def _mixed_episodes(n_per_subclass=20):
    episodes = []
    for subclass in DEFICIT_SUBCLASSES + NO_DEFICIT_SUBCLASSES:
        episodes += [generate_episode(0, subclass, i) for i in range(n_per_subclass)]
    return episodes


def test_needs_grounding_labels_matches_subclass():
    episodes = _mixed_episodes(5)
    labels = needs_grounding_labels(episodes)
    for ep, label in zip(episodes, labels):
        assert label == float(ep.needs_grounding)


def test_grounding_discrimination_perfect_oracle_scores():
    episodes = _mixed_episodes(20)
    scores = needs_grounding_labels(episodes)  # the oracle "score" is the label itself
    result = grounding_discrimination(episodes, scores)
    assert result["auroc"] == pytest.approx(1.0)
    assert result["brier"] == pytest.approx(0.0)
    assert result["n"] == len(episodes)


def test_grounding_discrimination_constant_score_is_uninformative():
    episodes = _mixed_episodes(20)
    scores = [0.5] * len(episodes)
    result = grounding_discrimination(episodes, scores)
    assert result["auroc"] == pytest.approx(0.5)


def test_grounding_discrimination_rejects_length_mismatch():
    episodes = _mixed_episodes(5)
    with pytest.raises(ValueError):
        grounding_discrimination(episodes, [0.5] * (len(episodes) - 1))


def test_grounding_discrimination_by_subclass_breakdown_present():
    episodes = _mixed_episodes(10)
    scores = needs_grounding_labels(episodes)
    result = grounding_discrimination(episodes, scores)
    assert set(result["by_subclass"]) == set(DEFICIT_SUBCLASSES + NO_DEFICIT_SUBCLASSES)
    # no_deficit is single-class (never needs grounding) -> AUROC is undefined.
    import math

    assert math.isnan(result["by_subclass"]["no_deficit"]["auroc"])


def test_grounding_discrimination_mean_score_split_by_deficit():
    episodes = _mixed_episodes(20)
    labels = needs_grounding_labels(episodes)
    result = grounding_discrimination(episodes, labels)
    assert result["mean_score_with_deficit"] == pytest.approx(1.0)
    assert result["mean_score_without_deficit"] == pytest.approx(0.0)


def test_grounding_auroc_improvement_positive_when_hg_beats_baseline():
    episodes = _mixed_episodes(20)
    labels = needs_grounding_labels(episodes)
    hg_scores = labels  # perfect
    baseline_scores = [0.5] * len(episodes)  # uninformative
    result = grounding_auroc_improvement(episodes, hg_scores, baseline_scores, n_boot=200, seed=0)
    assert result.point == pytest.approx(0.5)
    assert result.ci_low > 0.0
