from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.metrics.matrix import ErrorMatrix  # noqa: E402


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


def make_matrix(
    errors,
    predictions=None,
    statuses=None,
    families=None,
    condition="DD",
    confidences=None,
) -> ErrorMatrix:
    """Build an ErrorMatrix directly, bypassing the row pipeline."""
    e = np.asarray(errors, dtype=bool)
    n_i, n_m = e.shape
    if predictions is None:
        # Binary convention: wrong -> "b", right -> "a".
        predictions = np.where(e, "b", "a").astype(object)
    else:
        predictions = np.asarray(predictions, dtype=object)
    if statuses is None:
        statuses = np.array(["unanimous"] * n_i)
    else:
        statuses = np.asarray(statuses, dtype=str)
    if families is None:
        families = [f"fam{j}" for j in range(n_m)]
    if confidences is None:
        confidences = np.full((n_i, n_m), np.nan)
    return ErrorMatrix(
        condition=condition,
        item_ids=[f"i{k}" for k in range(n_i)],
        member_ids=[f"m{j}" for j in range(n_m)],
        errors=e,
        predictions=predictions,
        statuses=statuses,
        confidences=np.asarray(confidences, dtype=float),
        parse_ok=np.ones((n_i, n_m), dtype=bool),
        families=list(families),
        task="test",
    )


@pytest.fixture
def matrix_factory():
    return make_matrix


def simulate(
    n_items=2000, n_members=5, sigma_u=1.2, alpha=-1.5, core_rate=0.0, seed=0,
    contested_rate=0.0,
):
    """Draw an error matrix from the hierarchical model, optionally with a core.

    The generative truth is known, so tests can assert that the analysis
    recovers it rather than merely that it runs.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, sigma_u, n_items)
    alphas = np.full(n_members, alpha) if np.isscalar(alpha) else np.asarray(alpha)
    core = rng.random(n_items) < core_rate
    z = u[:, None] + alphas[None, :] + 8.0 * core[:, None]
    p = 1.0 / (1.0 + np.exp(-z))
    e = rng.random((n_items, n_members)) < p
    statuses = np.where(rng.random(n_items) < contested_rate, "contested", "unanimous")
    return make_matrix(e, statuses=statuses), {"u": u, "alpha": alphas, "core": core}


@pytest.fixture
def simulator():
    return simulate
