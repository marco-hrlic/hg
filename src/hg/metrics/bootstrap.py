"""Resampling (protocol section 12).

Rules the protocol fixes, implemented here and nowhere else:

* **Bootstrap over items.** Evaluation items are the primary independent
  sampling unit. Resampling members instead would treat five model families as
  a random draw from a population of families, which they are not.
* **Paired resampling for comparisons.** For `C_HS - C_DD`, the same sampled
  items must appear in both conditions. Independent resampling inflates the
  variance of the difference by discarding the item-level pairing, which is
  precisely the correlation that makes the comparison sensitive.
* **Report effect size, 95% CI, and odds ratio; do not rely on p-values.**
  So `BootstrapResult` carries the point estimate and interval, and the
  bootstrap p-value is present but explicitly secondary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .matrix import ErrorMatrix

Metric = Callable[[ErrorMatrix], float]
PairedMetric = Callable[[ErrorMatrix, ErrorMatrix], float]


@dataclass(slots=True)
class BootstrapResult:
    name: str
    point: float
    ci_low: float
    ci_high: float
    se: float
    n_boot: int
    n_valid: int
    alpha: float = 0.05
    # Two-sided bootstrap p-value for H0: value = null_value. Secondary to the
    # interval by design; reported so it cannot be quietly recomputed elsewhere
    # with a different convention.
    p_value: float | None = None
    null_value: float | None = None
    draws: list[float] = field(default_factory=list, repr=False)

    def to_dict(self, include_draws: bool = False) -> dict[str, object]:
        d: dict[str, object] = {
            "name": self.name,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "se": self.se,
            "n_boot": self.n_boot,
            "n_valid": self.n_valid,
            "alpha": self.alpha,
            "p_value": self.p_value,
            "null_value": self.null_value,
        }
        if include_draws:
            d["draws"] = self.draws
        return d

    def format(self) -> str:
        return (
            f"{self.name}: {self.point:.4f} "
            f"[{self.ci_low:.4f}, {self.ci_high:.4f}] (95% CI, {self.n_valid} draws)"
        )


def _percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    lo = float(np.nanpercentile(draws, 100 * alpha / 2))
    hi = float(np.nanpercentile(draws, 100 * (1 - alpha / 2)))
    return lo, hi


def _bootstrap_p(draws: np.ndarray, null_value: float) -> float:
    """Two-sided bootstrap p-value by interval inversion.

    Computed as twice the smaller tail mass on the null side, clipped to 1.
    Deliberately a coarse instrument: with 2000 draws the resolution floor is
    about 1e-3, so it is reported as `< 1/n_boot` territory rather than as a
    precise small number.
    """
    valid = draws[~np.isnan(draws)]
    if valid.size == 0:
        return float("nan")
    below = float((valid <= null_value).mean())
    above = float((valid >= null_value).mean())
    return min(1.0, 2.0 * min(below, above))


def bootstrap_metric(
    matrix: ErrorMatrix,
    metric: Metric,
    *,
    name: str = "metric",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    null_value: float | None = None,
) -> BootstrapResult:
    """Item-level bootstrap of a single-condition metric."""
    rng = np.random.default_rng(seed)
    point = metric(matrix)
    n = matrix.n_items
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = metric(matrix.resample_items(idx))
    valid = draws[~np.isnan(draws)]
    lo, hi = _percentile_ci(draws, alpha)
    return BootstrapResult(
        name=name,
        point=point,
        ci_low=lo,
        ci_high=hi,
        se=float(np.nanstd(draws, ddof=1)) if valid.size > 1 else float("nan"),
        n_boot=n_boot,
        n_valid=int(valid.size),
        alpha=alpha,
        p_value=(_bootstrap_p(draws, null_value) if null_value is not None else None),
        null_value=null_value,
        draws=[float(x) for x in draws],
    )


def paired_bootstrap(
    a: ErrorMatrix,
    b: ErrorMatrix,
    metric: Metric,
    *,
    name: str = "difference",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    null_value: float | None = 0.0,
    combine: Callable[[float, float], float] | None = None,
) -> BootstrapResult:
    """Paired item bootstrap of `metric(a) - metric(b)`.

    Both matrices must cover the same items in the same order; the same
    resampled item indices are applied to both, which is what preserves the
    pairing the protocol asks for.

    `combine` overrides the default difference, e.g. `lambda x, y: x / y` for
    the Delta_div ratio. The ratio's percentile interval is the honest one to
    report -- a normal-approximation interval on a ratio near zero is not.
    """
    if a.item_ids != b.item_ids:
        raise ValueError(
            "paired bootstrap requires the same items in both conditions; "
            f"{len(a.item_ids)} vs {len(b.item_ids)} items, order-sensitive. "
            "Use hg.metrics.bootstrap.align_matrices first."
        )
    comb = combine or (lambda x, y: x - y)
    rng = np.random.default_rng(seed)
    point = comb(metric(a), metric(b))
    n = a.n_items
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = comb(metric(a.resample_items(idx)), metric(b.resample_items(idx)))
    valid = draws[np.isfinite(draws)]
    lo, hi = _percentile_ci(draws, alpha)
    return BootstrapResult(
        name=name,
        point=point,
        ci_low=lo,
        ci_high=hi,
        se=float(np.nanstd(draws, ddof=1)) if valid.size > 1 else float("nan"),
        n_boot=n_boot,
        n_valid=int(valid.size),
        alpha=alpha,
        p_value=(_bootstrap_p(draws, null_value) if null_value is not None else None),
        null_value=null_value,
        draws=[float(x) for x in draws],
    )


def align_matrices(a: ErrorMatrix, b: ErrorMatrix) -> tuple[ErrorMatrix, ErrorMatrix]:
    """Restrict two matrices to their shared items, in a common order.

    Needed whenever conditions were run at different times or with different
    `unparsed_policy` settings. Alignment is explicit rather than automatic so
    that dropping items is a visible step: silently intersecting item sets can
    turn a like-for-like comparison into a comparison on a filtered population.
    """
    shared = [i for i in a.item_ids if i in set(b.item_ids)]
    if not shared:
        raise ValueError("the two conditions share no items")
    order = {item: k for k, item in enumerate(shared)}

    def restrict(m: ErrorMatrix) -> ErrorMatrix:
        keep = np.array([i in order for i in m.item_ids], dtype=bool)
        sub = m.select_items(keep)
        perm = np.argsort([order[i] for i in sub.item_ids])
        return sub.resample_items(perm)

    return restrict(a), restrict(b)


def odds_ratio(
    a: ErrorMatrix,
    b: ErrorMatrix,
    metric: Metric,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    haldane: float = 0.5,
) -> BootstrapResult:
    """Odds ratio of a rate metric between two conditions, with a paired CI.

    A Haldane-Anscombe correction of `haldane` is applied when either rate hits
    0 or 1, since an undefined odds ratio is common exactly in the interesting
    case where the diverse condition's clean residual goes to zero. The
    correction is reported in the name so it is not mistaken for a raw estimate.
    """
    def rate_to_odds(p: float, n: int) -> float:
        if math.isnan(p):
            return float("nan")
        if p <= 0.0 or p >= 1.0:
            k = p * n
            p = (k + haldane) / (n + 2 * haldane)
        return p / (1.0 - p)

    def combine(pa: float, pb: float) -> float:
        oa = rate_to_odds(pa, a.n_items)
        ob = rate_to_odds(pb, b.n_items)
        if math.isnan(oa) or math.isnan(ob) or ob == 0:
            return float("nan")
        return oa / ob

    res = paired_bootstrap(
        a, b, metric,
        name=f"odds_ratio(haldane={haldane})",
        n_boot=n_boot, alpha=alpha, seed=seed, null_value=1.0,
        combine=combine,
    )
    return res


def summarize(results: Sequence[BootstrapResult]) -> list[dict[str, object]]:
    return [r.to_dict() for r in results]
