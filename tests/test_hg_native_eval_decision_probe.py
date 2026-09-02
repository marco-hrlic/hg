"""Vanilla decision-probe control -- M1 kickoff section 12. Torch-free
(synthetic hidden states + hand-built episodes); `extract_hidden_states`
itself needs torch, so real-model wiring is verified on the Spark."""

from __future__ import annotations

import numpy as np
import pytest

from hg_native.data.schema import Episode
from hg_native.eval.decision_probe import DECISION_PROBE_FEATURE_NAMES, decision_probe_features, fit_decision_probe


def _episode(correct_action: str) -> Episode:
    return Episode(
        episode_id=f"e-{np.random.default_rng().integers(1_000_000)}",
        family="static_hidden",
        visible_state={},
        hidden_state={},
        full_state={},
        available_actions=["ACT_A", "ACT_B", "ACT_C"],
        correct_action=correct_action,
        distortion_profile={"ACT_A": 0.0, "ACT_B": 1.0, "ACT_C": 1.0},
        needs_grounding=False,
        min_grounding_bits=0.0,
    )


def _dataset(n=300, seed=0, undefined_frac=0.0, signal_strength=6.0):
    rng = np.random.default_rng(seed)
    episodes, hidden = [], rng.normal(size=(n, 5))
    for i in range(n):
        if rng.random() < undefined_frac:
            episodes.append(_episode("ACT_C"))
            continue
        label = "ACT_B" if rng.integers(0, 2) == 1 else "ACT_A"
        episodes.append(_episode(label))
        hidden[i, 0] += (signal_strength if label == "ACT_B" else -signal_strength)
    return hidden, episodes


def test_fit_decision_probe_recovers_a_clean_signal():
    hidden, episodes = _dataset(300, seed=0)
    probe = fit_decision_probe(hidden, episodes, seed=0)
    pred = (probe.predict_proba(hidden) > 0.5).astype(float)
    true = np.array([1.0 if ep.correct_action == "ACT_B" else 0.0 for ep in episodes])
    assert (pred == true).mean() > 0.9


def test_fit_decision_probe_near_chance_on_pure_noise():
    hidden, episodes = _dataset(300, seed=1, signal_strength=0.0)
    probe = fit_decision_probe(hidden, episodes, seed=0)
    pred = (probe.predict_proba(hidden) > 0.5).astype(float)
    true = np.array([1.0 if ep.correct_action == "ACT_B" else 0.0 for ep in episodes])
    assert 0.3 < (pred == true).mean() < 0.7


def test_fit_decision_probe_ignores_undefined_decisions():
    # A dataset that's mostly undefined (Family-D-style ACT_C) but with
    # enough defined A/B examples to still fit cleanly -- the undefined
    # ones must not corrupt the fit or crash it.
    hidden, episodes = _dataset(300, seed=2, undefined_frac=0.6, signal_strength=6.0)
    probe = fit_decision_probe(hidden, episodes, seed=0)
    defined = [(h, ep) for h, ep in zip(hidden, episodes) if ep.correct_action != "ACT_C"]
    defined_hidden = np.array([h for h, _ in defined])
    defined_true = np.array([1.0 if ep.correct_action == "ACT_B" else 0.0 for _, ep in defined])
    pred = (probe.predict_proba(defined_hidden) > 0.5).astype(float)
    assert (pred == defined_true).mean() > 0.85


def test_fit_decision_probe_rejects_too_few_defined_examples():
    hidden, episodes = _dataset(5, seed=3, undefined_frac=1.0)  # all undefined
    with pytest.raises(ValueError, match="at least 10"):
        fit_decision_probe(hidden, episodes, seed=0)


def test_fit_decision_probe_rejects_length_mismatch():
    hidden, episodes = _dataset(20, seed=4)
    with pytest.raises(ValueError):
        fit_decision_probe(hidden[:10], episodes, seed=0)


def test_decision_probe_features_shape_and_ranges():
    hidden, episodes = _dataset(200, seed=0)
    probe = fit_decision_probe(hidden, episodes, seed=0)
    features = decision_probe_features(probe, hidden)
    assert features.shape == (len(hidden), len(DECISION_PROBE_FEATURE_NAMES))
    decision_probability, decision_entropy, top1_top2_margin = features.T
    assert np.all((decision_probability >= 0.5) & (decision_probability <= 1.0))
    assert np.all((decision_entropy >= 0.0) & (decision_entropy <= np.log(2) + 1e-6))
    assert np.all((top1_top2_margin >= 0.0) & (top1_top2_margin <= 1.0))
    # A confident probe (clean signal) should show low entropy on average.
    assert decision_entropy.mean() < 0.3


def test_decision_probe_features_high_entropy_when_probe_is_near_chance():
    hidden, episodes = _dataset(200, seed=1, signal_strength=0.0)
    probe = fit_decision_probe(hidden, episodes, seed=0)
    features = decision_probe_features(probe, hidden)
    decision_entropy = features[:, DECISION_PROBE_FEATURE_NAMES.index("decision_entropy")]
    assert decision_entropy.mean() > 0.5
