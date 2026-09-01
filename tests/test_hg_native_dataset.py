"""The synthetic episode-mixture sampler -- torch-free, runs anywhere."""

from __future__ import annotations

import pytest

from hg_native.training.dataset import FAMILIES, SyntheticMix, sample_episodes


def test_rejects_zero_n():
    with pytest.raises(ValueError, match="n must be"):
        sample_episodes(seed=0, n=0)


def test_rejects_all_zero_mix():
    with pytest.raises(ValueError, match="sum to > 0"):
        SyntheticMix(static_hidden=0, accessibility=0, sequential_quotient=0, capability_refinement=0)


def test_deterministic():
    a = sample_episodes(seed=0, n=20)
    b = sample_episodes(seed=0, n=20)
    assert [ep.episode_id for ep in a] == [ep.episode_id for ep in b]


def test_start_index_changes_the_batch():
    a = sample_episodes(seed=0, n=10, start_index=0)
    b = sample_episodes(seed=0, n=10, start_index=10)
    assert [ep.episode_id for ep in a] != [ep.episode_id for ep in b]


def test_returns_only_episodes_from_the_four_synthetic_families():
    episodes = sample_episodes(seed=0, n=200)
    assert len(episodes) == 200
    assert {ep.family for ep in episodes} <= set(FAMILIES)


def test_zero_weight_family_never_sampled():
    mix = SyntheticMix(static_hidden=1.0, accessibility=0.0, sequential_quotient=0.0, capability_refinement=0.0)
    episodes = sample_episodes(seed=0, n=100, mix=mix)
    assert {ep.family for ep in episodes} == {"static_hidden"}


def test_mix_weights_are_renormalized():
    mix = SyntheticMix(static_hidden=2.0, accessibility=2.0, sequential_quotient=0.0, capability_refinement=0.0)
    assert mix.weights() == pytest.approx([0.5, 0.5, 0.0, 0.0])
