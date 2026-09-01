"""Item-level percentile bootstrap CIs (spec section 18).

Generic over any scalar metric computed from item-indexed arrays -- unlike
`hg.metrics.bootstrap`, which is tightly coupled to the M2-LLM `ErrorMatrix`
and not reusable here (see [[hg-v2-project-state]]). Mirrors that module's
design for the same reasons, on plain arrays instead: item-level resampling
(episodes are the independent sampling unit, spec section 18), paired
resampling for a HG-vs-baseline comparison (the same resampled episodes in
both, so the comparison is sensitive to the pairing rather than diluted by
independent noise), and a percentile CI rather than a normal approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

MetricFn = Callable[..., float]


@dataclass(slots=True)
class BootstrapResult:
    name: str
    point: float
    ci_low: float
    ci_high: float
    n_boot: int
    n_valid: int
    alpha: float = 0.05

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_boot": self.n_boot,
            "n_valid": self.n_valid,
            "alpha": self.alpha,
        }

    def format(self) -> str:
        return f"{self.name}: {self.point:.4f} [{self.ci_low:.4f}, {self.ci_high:.4f}] (95% CI, {self.n_valid} draws)"


def _percentile_ci(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    return float(np.nanpercentile(draws, 100 * alpha / 2)), float(np.nanpercentile(draws, 100 * (1 - alpha / 2)))


def bootstrap_ci(
    *arrays,
    metric: MetricFn,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    name: str = "metric",
) -> BootstrapResult:
    """Bootstrap CI for `metric(*arrays)`. Every array must have the same length
    (one entry per episode); the same resampled indices are applied to all of
    them together, so `metric` can be e.g. `auroc` needing paired
    (scores, labels)."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays passed to bootstrap_ci must have the same length")
    rng = np.random.default_rng(seed)
    point = metric(*arrays)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = metric(*[a[idx] for a in arrays])
    valid = draws[np.isfinite(draws)]
    lo, hi = _percentile_ci(draws, alpha)
    return BootstrapResult(name=name, point=point, ci_low=lo, ci_high=hi, n_boot=n_boot, n_valid=int(valid.size), alpha=alpha)


def paired_bootstrap_diff(
    arrays_a: tuple,
    arrays_b: tuple,
    metric: MetricFn,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    name: str = "difference",
) -> BootstrapResult:
    """Bootstrap CI for `metric(*arrays_a) - metric(*arrays_b)` over the same
    episodes in both conditions (`arrays_a`/`arrays_b` must be the same
    length and in matching episode order -- the caller aligns them, this
    function does not silently intersect item sets, per
    `hg.metrics.bootstrap.align_matrices`'s own rationale for making that an
    explicit step)."""
    arrays_a = [np.asarray(a) for a in arrays_a]
    arrays_b = [np.asarray(a) for a in arrays_b]
    n = len(arrays_a[0])
    if any(len(a) != n for a in arrays_a) or any(len(a) != n for a in arrays_b):
        raise ValueError("paired bootstrap requires the same number of items in both conditions")
    rng = np.random.default_rng(seed)
    point = metric(*arrays_a) - metric(*arrays_b)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = metric(*[a[idx] for a in arrays_a]) - metric(*[a[idx] for a in arrays_b])
    valid = draws[np.isfinite(draws)]
    lo, hi = _percentile_ci(draws, alpha)
    return BootstrapResult(name=name, point=point, ci_low=lo, ci_high=hi, n_boot=n_boot, n_valid=int(valid.size), alpha=alpha)
