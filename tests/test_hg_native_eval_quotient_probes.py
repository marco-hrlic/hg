"""Linear probes for quotient recovery -- torch-free (synthetic latents)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg_native.data.sequential_quotient import generate_batch
from hg_native.eval.quotient_probes import probe_accuracy, quotient_recovery


def _episodes(n=300):
    trajs = generate_batch(seed=0, n_trajectories=n, n_steps=1)
    return [traj[0] for traj in trajs]


# --- probe_accuracy ---------------------------------------------------


def test_probe_accuracy_recovers_a_perfectly_encoded_label():
    rng = np.random.default_rng(0)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(float)
    latents = rng.normal(size=(n, 5))
    latents[:, 0] = labels * 4 - 2 + rng.normal(scale=0.1, size=n)
    result = probe_accuracy(latents, labels, seed=0)
    assert result["accuracy"] > 0.9


def test_probe_accuracy_near_chance_on_pure_noise():
    rng = np.random.default_rng(1)
    n = 200
    labels = rng.integers(0, 2, size=n).astype(float)
    latents = rng.normal(size=(n, 5))
    result = probe_accuracy(latents, labels, seed=0)
    assert 0.3 < result["accuracy"] < 0.7


def test_probe_accuracy_rejects_too_few_examples():
    with pytest.raises(ValueError, match="at least 10"):
        probe_accuracy(np.zeros((5, 3)), np.array([0, 1, 0, 1, 0]))


def test_probe_accuracy_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        probe_accuracy(np.zeros((10, 3)), np.zeros(9))


def test_probe_accuracy_nan_for_single_class_training_labels():
    latents = np.random.default_rng(0).normal(size=(20, 3))
    labels = np.zeros(20)
    result = probe_accuracy(latents, labels, seed=0)
    assert math.isnan(result["accuracy"])


def test_probe_accuracy_deterministic_given_seed():
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(100, 4))
    labels = rng.integers(0, 2, size=100).astype(float)
    a = probe_accuracy(latents, labels, seed=0)
    b = probe_accuracy(latents, labels, seed=0)
    assert a == b


# --- quotient_recovery ---------------------------------------------------


def test_quotient_recovery_synthetic_latents_show_high_selectivity():
    episodes = _episodes(300)
    rng = np.random.default_rng(0)
    n = len(episodes)
    latents = rng.normal(scale=1.0, size=(n, 5))
    for i, ep in enumerate(episodes):
        r = 1.0 if ep.true_quotient_state["r"] == "YES" else -1.0
        f = 1.0 if ep.true_quotient_state["f"] == "YES" else -1.0
        h = 1.0 if ep.true_quotient_state["h"] == "YES" else -1.0
        latents[i, 0] += r * 4
        latents[i, 1] += f * 4
        latents[i, 2] += h * 4
        # dims 3, 4 (which would carry u, n) are left as pure noise.

    result = quotient_recovery(latents, episodes, seed=0)
    for var in ("r", "f", "h"):
        assert result["probes"][var]["accuracy"] > 0.85, var
    for var in ("u", "n"):
        assert result["probes"][var]["accuracy"] < 0.7, var
    assert result["q_sel"] > 0.2


def test_quotient_recovery_rejects_length_mismatch():
    episodes = _episodes(20)
    with pytest.raises(ValueError):
        quotient_recovery(np.zeros((10, 5)), episodes)


def test_quotient_recovery_has_every_variable():
    episodes = _episodes(50)
    latents = np.random.default_rng(0).normal(size=(len(episodes), 4))
    result = quotient_recovery(latents, episodes, seed=0)
    assert set(result["probes"]) == {"r", "f", "h", "u", "n"}
