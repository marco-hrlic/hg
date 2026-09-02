"""Delta_refine metric machinery -- torch-free (synthetic latents + hand-built
episodes), verified before trusting it on a real trained model (M1 kickoff
section 16 Stage M1-0: "verify refinement metric behaves correctly")."""

from __future__ import annotations

import numpy as np
import pytest

from hg_native.data.schema import Episode
from hg_native.eval.refinement import delta_refine, probe_by_capability, refinement_report


def _episode(capability: str, v: str, u: str) -> Episode:
    return Episode(
        episode_id=f"e-{capability}-{v}-{u}-{np.random.default_rng().integers(1_000_000)}",
        family="capability_refinement",
        visible_state={"v": v, "u": u},
        hidden_state={},
        full_state={"v": v, "u": u},
        available_actions=["ACT_A", "ACT_B"],
        correct_action="ACT_A",
        distortion_profile={"ACT_A": 0.0, "ACT_B": 1.0},
        needs_grounding=False,
        min_grounding_bits=0.0,
        true_quotient_state={"v": v, "u": u},
        metadata={"capability": capability},
    )


def _refining_dataset(n=200, seed=0):
    """`v` is linearly encoded in latent dim 0 only under C2; `u` is never
    encoded under either capability -- the exact pattern H-M1-4/H-M1-5
    predict for a genuinely refining variable vs. an always-irrelevant one."""
    rng = np.random.default_rng(seed)
    episodes = []
    latents = rng.normal(scale=1.0, size=(n, 3))
    for i in range(n):
        capability = "C1" if i % 2 == 0 else "C2"
        v = "YES" if rng.integers(0, 2) == 1 else "NO"
        u = "YES" if rng.integers(0, 2) == 1 else "NO"
        episodes.append(_episode(capability, v, u))
        if capability == "C2":
            latents[i, 0] += (1.0 if v == "YES" else -1.0) * 4
        # u never perturbs any latent dim, under either capability.
    return latents, episodes


def test_probe_by_capability_splits_correctly():
    latents, episodes = _refining_dataset()
    result = probe_by_capability(latents, episodes, "v", seed=0)
    assert set(result) == {"C1", "C2"}
    assert result["C1"]["n_train"] + result["C1"]["n_test"] == sum(
        1 for ep in episodes if ep.metadata["capability"] == "C1"
    )


def test_delta_refine_positive_for_a_variable_that_becomes_relevant_under_c2():
    latents, episodes = _refining_dataset()
    delta = delta_refine(latents, episodes, "v", seed=0)
    assert delta > 0.3, delta


def test_delta_refine_near_zero_for_a_variable_irrelevant_under_both():
    latents, episodes = _refining_dataset()
    delta = delta_refine(latents, episodes, "u", seed=0)
    assert abs(delta) < 0.25, delta


def test_refinement_report_has_both_variables_and_deltas():
    latents, episodes = _refining_dataset()
    report = refinement_report(latents, episodes, refining_var="v", control_var="u", seed=0)
    assert set(report) == {"v", "u", "delta_refine"}
    assert set(report["delta_refine"]) == {"v", "u"}
    assert report["delta_refine"]["v"] > report["delta_refine"]["u"]


def test_probe_by_capability_rejects_length_mismatch():
    _, episodes = _refining_dataset(n=20)
    with pytest.raises(ValueError):
        probe_by_capability(np.zeros((5, 3)), episodes, "v")


def test_probe_by_capability_requires_both_capabilities_present():
    latents, episodes = _refining_dataset(n=20)
    c1_only = [ep for ep in episodes if ep.metadata["capability"] == "C1"]
    with pytest.raises(ValueError, match="C2"):
        probe_by_capability(latents[: len(c1_only)], c1_only, "v")
