"""AUROC / AUPRC / Brier / ECE for a binary label against a continuous score.

No `sklearn`/`scipy` dependency (neither is in `pyproject.toml`) -- each
metric is a direct, exact implementation from its definition, not an
approximation. Torch-free: these operate on plain arrays of scores/labels,
whatever produced them (spec section 10.3's grounding discrimination and
section 10.5's accessibility discrimination both reduce to this).
"""

from __future__ import annotations

import numpy as np


def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)
    if s.shape != y.shape:
        raise ValueError(f"scores and labels must have the same shape, got {s.shape} vs {y.shape}")
    return s, y


def auroc(scores, labels) -> float:
    """P(score_pos > score_neg) + 0.5*P(score_pos == score_neg), computed exactly.

    Equivalent to the Mann-Whitney U statistic / trapezoidal ROC AUC.
    `nan` if either class is empty (undefined, not "wrong" -- the spec
    section 10.3 subclass breakdown will legitimately hit this whenever a
    subclass is single-class by construction, e.g. `no_deficit`).
    """
    s, y = _as_arrays(scores, labels)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def auprc(scores, labels) -> float:
    """Average precision: sum_n (R_n - R_{n-1}) * P_n over descending-score thresholds."""
    s, y = _as_arrays(scores, labels)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1.0 - y_sorted)
    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_pos
    recall_prev = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - recall_prev) * precision))


def brier_score(scores, labels) -> float:
    """Mean squared error between predicted probability and the binary label."""
    s, y = _as_arrays(scores, labels)
    return float(np.mean((s - y) ** 2))


def ece(scores, labels, n_bins: int = 10) -> float:
    """Expected calibration error: |accuracy - confidence|, bin-weighted by count."""
    s, y = _as_arrays(scores, labels)
    if not (0 <= s.min() and s.max() <= 1):
        raise ValueError("ece expects scores in [0, 1] (a probability, not a raw logit)")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(s)
    error = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (s >= lo) & (s <= hi) if i == n_bins - 1 else (s >= lo) & (s < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        error += (n_bin / total) * abs(y[mask].mean() - s[mask].mean())
    return float(error)


def summary(scores, labels, n_bins: int = 10) -> dict[str, float]:
    """All four metrics plus n/n_positive, in one call (spec section 10.3's report line)."""
    s, y = _as_arrays(scores, labels)
    return {
        "auroc": auroc(s, y),
        "auprc": auprc(s, y),
        "brier": brier_score(s, y),
        "ece": ece(s, y, n_bins=n_bins),
        "n": len(s),
        "n_positive": int(y.sum()),
    }
