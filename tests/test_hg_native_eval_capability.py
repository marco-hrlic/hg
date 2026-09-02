"""Capability floor eval -- torch-free."""

from __future__ import annotations

import pytest

from hg_native.data.capability_refinement import generate_pair
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.eval.capability import business_decision_labels_and_mask, capability_floor
from hg_native.model.config import DEFAULT_DECISION_LABELS


def _mixed_episodes(n=20):
    episodes = []
    for sc in ("no_deficit", "one_bit_deficit", "multi_bit_deficit"):
        episodes += [gen_static_hidden(0, sc, i) for i in range(n)]
    return episodes


def test_business_decision_labels_excludes_family_d_third_action():
    pair = generate_pair(0, 0)
    labels, mask = business_decision_labels_and_mask([pair["z_t0_C1"], pair["z_t0_C2"]])
    assert bool(mask[0]) is True  # z_C1's decision is ACT_A -> defined
    assert bool(mask[1]) is False  # z_C2's decision is ACT_C -> undefined


def test_capability_floor_perfect_predictions():
    episodes = _mixed_episodes(20)
    labels, mask = business_decision_labels_and_mask(episodes)
    result = capability_floor(episodes, labels.clip(min=0))  # oracle predictions (labels are already 0/1)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["balanced_accuracy"] == pytest.approx(1.0)
    assert result["constant_label_collapse"] is False


def test_capability_floor_detects_constant_label_collapse():
    episodes = _mixed_episodes(20)
    always_a = [0] * len(episodes)
    result = capability_floor(episodes, always_a)
    assert result["constant_label_collapse"] is True


def test_capability_floor_balanced_accuracy_handles_class_imbalance():
    # 18 A's, 2 B's; a naive "always A" predictor gets high raw accuracy but
    # should be penalized by balanced accuracy.
    episodes = [gen_static_hidden(0, "no_deficit", i) for i in range(200)]
    labels, mask = business_decision_labels_and_mask(episodes)
    always_a = [0] * len(episodes)
    result = capability_floor(episodes, always_a)
    assert result["balanced_accuracy"] <= result["accuracy"] + 1e-9


def test_capability_floor_empty_when_no_defined_labels():
    pair = generate_pair(0, 0)
    result = capability_floor([pair["z_t0_C2"]], [0])
    assert result["n"] == 0


def test_capability_floor_rejects_length_mismatch():
    episodes = _mixed_episodes(5)
    with pytest.raises(ValueError):
        capability_floor(episodes, [0] * (len(episodes) - 1))
