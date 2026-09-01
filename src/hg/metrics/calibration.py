"""Calibration of a decision-state estimator, separately from the policy it feeds.

Improved plan section 2.3 is the reason this module exists: routing regret alone
can hide estimator error, because a thresholded policy is unchanged by any
estimate that stays on the same side of the threshold. Phase 1 already measured
that effect -- its learned router and the exact Bayes posterior produced
*identical* routes while holding different numbers -- and reported it as a
divergence-window width. This module measures the estimates themselves.

Everything here takes plain sequences rather than episodes, so the same
functions score a router's P(acting now is wrong), a synthesizer's predicted
soundness, or anything else with a probability and an outcome.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(slots=True)
class CalibrationReport:
    """Every quantity plan section 2.3 asks for, in one object."""

    n: int
    brier: float
    expected_calibration_error: float
    max_calibration_error: float
    mean_absolute_error: float
    mean_predicted: float
    mean_observed: float
    curve: list[dict[str, Any]]
    spearman: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "brier": self.brier,
            "expected_calibration_error": self.expected_calibration_error,
            "max_calibration_error": self.max_calibration_error,
            "mean_absolute_error": self.mean_absolute_error,
            "mean_predicted": self.mean_predicted,
            "mean_observed": self.mean_observed,
            "spearman": self.spearman,
            "curve": list(self.curve),
        }


def brier_score(predicted: Sequence[float], outcome: Sequence[float]) -> float:
    """Mean squared error of a probabilistic forecast."""
    p = np.asarray(list(predicted), dtype=float)
    y = np.asarray(list(outcome), dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"{p.shape} predictions but {y.shape} outcomes")
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def calibration_curve(
    predicted: Sequence[float], outcome: Sequence[float], *, n_bins: int = 10
) -> list[dict[str, Any]]:
    """Observed frequency against predicted probability, in equal-width bins.

    Equal-width rather than equal-count bins: the question is whether "0.3"
    means 0.3, which is a statement about the value, not about the ranking. A
    quantile binning would hide a region the estimator never visits, and the
    regions it never visits are exactly where a threshold might sit.
    """
    p = np.asarray(list(predicted), dtype=float)
    y = np.asarray(list(outcome), dtype=float)
    if p.size == 0:
        return []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # The last bin is closed on the right so a prediction of exactly 1.0 lands
    # in it rather than in a bin of its own.
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[dict[str, Any]] = []
    for b in range(n_bins):
        mask = index == b
        n = int(mask.sum())
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": n,
                "mean_predicted": float(p[mask].mean()) if n else None,
                "observed_frequency": float(y[mask].mean()) if n else None,
            }
        )
    return rows


def expected_calibration_error(curve: Sequence[dict[str, Any]], n_total: int) -> tuple[float, float]:
    """(ECE, MCE) from a calibration curve. Empty bins contribute nothing."""
    if n_total <= 0:
        return float("nan"), float("nan")
    ece = 0.0
    mce = 0.0
    for row in curve:
        if not row["n"]:
            continue
        gap = abs(row["mean_predicted"] - row["observed_frequency"])
        ece += (row["n"] / n_total) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Rank correlation, ties averaged. None when either side is constant.

    None rather than NaN: a constant predictor has no rank correlation to
    report, and that is a fact about the estimator worth keeping distinguishable
    from a failed computation.
    """
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    if a.size < 2 or a.shape != b.shape:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="stable")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    # Average the ranks within each tie group, so a coarse estimator with many
    # ties is not credited with an ordering it did not express.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(unique.size)
        np.add.at(sums, inverse, ranks)
        ranks = (sums / counts)[inverse]
    return ranks


def calibration_report(
    predicted: Sequence[float],
    outcome: Sequence[float],
    *,
    n_bins: int = 10,
    against: Sequence[float] | None = None,
) -> CalibrationReport:
    """Brier, ECE, MCE, MAE and the curve, plus an optional rank correlation.

    `against` is the quantity the estimate is supposed to order -- the empirical
    value of asking, in the plan's section 13 -- when that differs from the
    binary outcome being scored.
    """
    p = np.asarray(list(predicted), dtype=float)
    y = np.asarray(list(outcome), dtype=float)
    curve = calibration_curve(p, y, n_bins=n_bins)
    ece, mce = expected_calibration_error(curve, p.size)
    return CalibrationReport(
        n=int(p.size),
        brier=brier_score(p, y),
        expected_calibration_error=ece,
        max_calibration_error=mce,
        mean_absolute_error=float(np.mean(np.abs(p - y))) if p.size else float("nan"),
        mean_predicted=float(p.mean()) if p.size else float("nan"),
        mean_observed=float(y.mean()) if y.size else float("nan"),
        curve=curve,
        spearman=spearman(p, against if against is not None else y),
    )
