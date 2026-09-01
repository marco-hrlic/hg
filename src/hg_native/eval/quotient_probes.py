"""Quotient recovery via linear probes -- spec section 10.6, gate N4.

Torch-free: takes a saved latent array (e.g. from `eval.latents.extract_latents`)
plus Family C episodes, not a model, so probe results are recomputable from
`latents.npy` + `episodes.jsonl` alone (spec section 4). No sklearn
dependency -- `_fit_logistic_regression` is a small hand-rolled batch
gradient descent, since a d<=100-ish linear probe needs nothing fancier.
"""

from __future__ import annotations

import math

import numpy as np

from hg_native.data.schema import Episode

# spec section 5.3 / 10.6: S_t* = (r_t, f_t, h) is what the latent should
# recover; u_t (persistent irrelevant) and n_t (random irrelevant) are the
# baseline it should *not* need.
RELEVANT_VARS = ("r", "f", "h")
IRRELEVANT_VARS = ("u", "n")


def _fit_logistic_regression(X: np.ndarray, y: np.ndarray, n_iters: int, lr: float, l2: float, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad_w = X.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def probe_accuracy(
    latents,
    labels,
    test_frac: float = 0.3,
    n_iters: int = 500,
    lr: float = 0.5,
    l2: float = 1e-3,
    seed: int = 0,
) -> dict:
    """Fit a logistic-regression probe on a train split, report held-out accuracy.

    A probe fit and scored on the same examples would overstate what the
    latent actually encodes (memorizing a specific z, not the true concept),
    so this always evaluates on a disjoint split.
    """
    X = np.asarray(latents, dtype=float)
    y = np.asarray(labels, dtype=float)
    n = len(y)
    if len(X) != n:
        raise ValueError(f"{len(X)} latents but {n} labels")
    if n < 10:
        raise ValueError(f"need at least 10 examples to fit+evaluate a probe, got {n}")

    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    y_train, y_test = y[train_idx], y[test_idx]

    if len(np.unique(y_train)) < 2:
        return {"accuracy": float("nan"), "n_train": len(train_idx), "n_test": len(test_idx)}

    mean, std = X[train_idx].mean(axis=0), X[train_idx].std(axis=0) + 1e-8
    X_train, X_test = (X[train_idx] - mean) / std, (X[test_idx] - mean) / std
    w, b = _fit_logistic_regression(X_train, y_train, n_iters=n_iters, lr=lr, l2=l2, seed=seed)
    pred = ((X_test @ w + b) > 0).astype(float)
    return {"accuracy": float((pred == y_test).mean()), "n_train": len(train_idx), "n_test": len(test_idx)}


def _variable_label(episode: Episode, var: str) -> float:
    value = episode.true_quotient_state[var] if var in RELEVANT_VARS else episode.metadata[var]
    return 1.0 if value == "YES" else 0.0


def quotient_recovery(latents, episodes: list[Episode], **probe_kwargs) -> dict:
    """Probe accuracy for every quotient/irrelevant variable, plus
    `Q_sel = mean(ProbeAcc_relevant) - mean(ProbeAcc_irrelevant)` (spec
    section 10.6's optional selectivity score)."""
    if len(latents) != len(episodes):
        raise ValueError(f"{len(latents)} latents but {len(episodes)} episodes")
    probes = {}
    for var in RELEVANT_VARS + IRRELEVANT_VARS:
        labels = np.array([_variable_label(ep, var) for ep in episodes])
        probes[var] = probe_accuracy(latents, labels, **probe_kwargs)

    relevant = [probes[v]["accuracy"] for v in RELEVANT_VARS if not math.isnan(probes[v]["accuracy"])]
    irrelevant = [probes[v]["accuracy"] for v in IRRELEVANT_VARS if not math.isnan(probes[v]["accuracy"])]
    q_sel = float(np.mean(relevant) - np.mean(irrelevant)) if relevant and irrelevant else float("nan")
    return {"probes": probes, "q_sel": q_sel}
