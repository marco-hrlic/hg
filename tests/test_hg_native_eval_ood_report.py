"""Condition-matrix reporting -- kickoff section 10/11/20. Torch-free
(synthetic latents first, then real Family C OOD-pool episodes with
fabricated-but-structured latents), verified before this is ever wired
against a real trained checkpoint."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg_native.data.ood import sequential_quotient_iid_pool, sequential_quotient_parameter_ood_pool
from hg_native.data.schema import Episode
from hg_native.eval.ood_report import condition_matrix, format_matrix_table, rho_ood

RELEVANT = ("r", "f", "h")
IRRELEVANT = ("u", "n")


def _episode(r: str, f: str, h: str, u: str, n: str) -> Episode:
    return Episode(
        episode_id=f"e-{np.random.default_rng().integers(1_000_000)}",
        family="sequential_quotient",
        visible_state={"r": r, "f": f, "u": u, "n": n},
        hidden_state={"h": "UNKNOWN"},
        full_state={"r": r, "f": f, "u": u, "n": n, "h": h},
        available_actions=["ACT_A", "ACT_B", "ASK_h"],
        correct_action="ACT_A",
        distortion_profile={"ACT_A": 0.0, "ACT_B": 1.0, "ASK_h": 0.0},
        needs_grounding=False,
        min_grounding_bits=0.0,
        true_quotient_state={"r": r, "f": f, "h": h},
        metadata={"u": u, "n": n},
    )


def _synthetic_condition(n: int, seed: int, encode_strength: float):
    """`r`/`f`/`h` are encoded in latent dims 0-2 with `encode_strength`;
    `u`/`n` are always pure noise. `encode_strength=0` gives a chance-level
    (Q_sel ~= 0) condition; a strong positive value gives high selectivity."""
    rng = np.random.default_rng(seed)
    episodes, latents = [], rng.normal(scale=1.0, size=(n, 5))
    for i in range(n):
        bits = {v: ("YES" if rng.integers(0, 2) == 1 else "NO") for v in ("r", "f", "h", "u", "n")}
        episodes.append(_episode(**bits))
        for dim, var in enumerate(("r", "f", "h")):
            latents[i, dim] += (1.0 if bits[var] == "YES" else -1.0) * encode_strength
    return latents, episodes


# --- synthetic verification ---------------------------------------------------


def test_condition_matrix_recovers_expected_q_sel_ordering():
    # Strengths chosen far apart (well below and at the probe-accuracy
    # ceiling) so the ordering isn't noise-sensitive: ~1.0 relevant-var
    # accuracy at 8.0, a clearly partial signal at 0.5, and pure chance at 0.0.
    latents_by_condition = {
        "iid": _synthetic_condition(150, seed=0, encode_strength=8.0)[0],
        "weaker_ood": _synthetic_condition(150, seed=1, encode_strength=0.5)[0],
        "chance_ood": _synthetic_condition(150, seed=2, encode_strength=0.0)[0],
    }
    episodes_by_condition = {
        "iid": _synthetic_condition(150, seed=0, encode_strength=8.0)[1],
        "weaker_ood": _synthetic_condition(150, seed=1, encode_strength=0.5)[1],
        "chance_ood": _synthetic_condition(150, seed=2, encode_strength=0.0)[1],
    }
    matrix = condition_matrix(latents_by_condition, episodes_by_condition, RELEVANT, IRRELEVANT, n_boot=20, seed=0)
    assert set(matrix) == {"iid", "weaker_ood", "chance_ood"}
    assert matrix["iid"]["q_sel"] > matrix["weaker_ood"]["q_sel"] > matrix["chance_ood"]["q_sel"]
    assert matrix["iid"]["q_sel"] > 0.3


def test_rho_ood_ratio_reflects_relative_selectivity():
    latents_by_condition = {
        "iid": _synthetic_condition(200, seed=0, encode_strength=4.0)[0],
        "half": _synthetic_condition(200, seed=1, encode_strength=4.0)[0],
    }
    episodes_by_condition = {
        "iid": _synthetic_condition(200, seed=0, encode_strength=4.0)[1],
        "half": _synthetic_condition(200, seed=1, encode_strength=4.0)[1],
    }
    matrix = condition_matrix(latents_by_condition, episodes_by_condition, RELEVANT, IRRELEVANT, n_boot=50, seed=0)
    ratio = rho_ood(matrix, "half")
    assert 0.5 < ratio < 1.5  # both conditions have the same encode_strength


def test_rho_ood_returns_nan_when_iid_q_sel_is_not_positive():
    latents_by_condition = {
        "iid": _synthetic_condition(200, seed=0, encode_strength=0.0)[0],
        "ood": _synthetic_condition(200, seed=1, encode_strength=4.0)[0],
    }
    episodes_by_condition = {
        "iid": _synthetic_condition(200, seed=0, encode_strength=0.0)[1],
        "ood": _synthetic_condition(200, seed=1, encode_strength=4.0)[1],
    }
    matrix = condition_matrix(latents_by_condition, episodes_by_condition, RELEVANT, IRRELEVANT, n_boot=20, seed=0)
    # chance-level IID Q_sel can land at or below 0; the guard must fire whenever it does.
    if matrix["iid"]["q_sel"] <= 0:
        assert math.isnan(rho_ood(matrix, "ood"))


def test_condition_matrix_requires_iid_key_present():
    latents_by_condition = {"parameter": _synthetic_condition(20, seed=0, encode_strength=1.0)[0]}
    episodes_by_condition = {"parameter": _synthetic_condition(20, seed=0, encode_strength=1.0)[1]}
    with pytest.raises(ValueError, match="iid"):
        condition_matrix(latents_by_condition, episodes_by_condition, RELEVANT, IRRELEVANT, n_boot=10)


def test_format_matrix_table_lists_iid_first_and_one_row_per_condition():
    latents_by_condition = {
        "iid": _synthetic_condition(50, seed=0, encode_strength=3.0)[0],
        "zzz_last_alphabetically": _synthetic_condition(50, seed=1, encode_strength=3.0)[0],
        "aaa_first_alphabetically": _synthetic_condition(50, seed=2, encode_strength=3.0)[0],
    }
    episodes_by_condition = {
        "iid": _synthetic_condition(50, seed=0, encode_strength=3.0)[1],
        "zzz_last_alphabetically": _synthetic_condition(50, seed=1, encode_strength=3.0)[1],
        "aaa_first_alphabetically": _synthetic_condition(50, seed=2, encode_strength=3.0)[1],
    }
    matrix = condition_matrix(latents_by_condition, episodes_by_condition, RELEVANT, IRRELEVANT, n_boot=20, seed=0)
    table = format_matrix_table(matrix)
    lines = table.splitlines()[2:]  # skip header + separator
    assert len(lines) == 3
    assert lines[0].startswith("iid")
    assert lines[1].startswith("aaa_first_alphabetically")
    assert lines[2].startswith("zzz_last_alphabetically")


# --- real Family C OOD pools, fabricated-but-structured latents ---------------


def test_condition_matrix_against_real_sequential_quotient_ood_pools():
    iid_episodes = sequential_quotient_iid_pool(seed=0, n_trajectories=40)
    ood_episodes = sequential_quotient_parameter_ood_pool(seed=0, n_trajectories=40)
    rng = np.random.default_rng(0)

    def _fabricate_latents(episodes):
        latents = rng.normal(scale=1.0, size=(len(episodes), 5))
        for i, ep in enumerate(episodes):
            for dim, var in enumerate(("r", "f", "h")):
                value = ep.true_quotient_state[var]
                latents[i, dim] += (1.0 if value == "YES" else -1.0) * 3.0
        return latents

    latents_by_condition = {"iid": _fabricate_latents(iid_episodes), "parameter": _fabricate_latents(ood_episodes)}
    episodes_by_condition = {"iid": iid_episodes, "parameter": ood_episodes}
    matrix = condition_matrix(latents_by_condition, episodes_by_condition, RELEVANT, IRRELEVANT, n_boot=30, seed=0)
    assert set(matrix) == {"iid", "parameter"}
    for condition in matrix:
        assert set(matrix[condition]["probes"]) == {"r", "f", "h", "u", "n"}
    assert not math.isnan(rho_ood(matrix, "parameter"))
