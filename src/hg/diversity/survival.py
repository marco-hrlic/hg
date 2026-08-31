"""The residual survival curve (protocol section 10).

Compute C_k^clean -- clean consensus failure -- as a function of k, the number
of independent model families in the ensemble, averaging over family subsets.
Then read the shape:

    C_k^clean -> 0            closure: internal diversity is sufficient here
    C_k^clean -> c > 0        candidate persistent residual; extend the axis
    C_k -> 0, contested stays  the remaining problem is anchor adjudication

Two details that decide whether the curve means anything:

* **Subsets are over families, not members.** Averaging over arbitrary member
  subsets in a homogeneous condition would trace out a seed-variation curve and
  call it a diversity curve. When a family contributes several members, one is
  chosen per family per subset (deterministically), so k really is the number
  of independent lineages.
* **A nonzero endpoint at k=5 is an upper bound, not a floor.** Section 18
  lists "insufficient diversity range" as a confound: a plateau claim needs the
  curve to actually flatten. `survival_shape` therefore refuses to report a
  plateau on the basis of the endpoint alone and looks at the trailing slope.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Iterable, Sequence

import numpy as np

from ..metrics.core import clean_consensus_wrong, consensus_wrong, disagreement_rate
from ..metrics.matrix import ErrorMatrix

Metric = Callable[[ErrorMatrix], float]


@dataclass(slots=True)
class SurvivalPoint:
    k: int
    mean: float
    sd: float
    lo: float
    hi: float
    n_subsets: int
    values: list[float] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.k,
            "mean": self.mean,
            "sd": self.sd,
            "min": self.lo,
            "max": self.hi,
            "n_subsets": self.n_subsets,
        }


def _family_groups(matrix: ErrorMatrix) -> dict[str, list[str]]:
    """member ids grouped by family, order preserved."""
    groups: dict[str, list[str]] = {}
    for member, family in zip(matrix.member_ids, matrix.families):
        groups.setdefault(family, []).append(member)
    return groups


def _pick_representative(members: Sequence[str], salt: str) -> str:
    """Deterministically choose one member to represent its family.

    Hash-based rather than "first", so repeated subsets do not all reuse the
    same seed and thereby understate within-family variation.
    """
    def key(m: str) -> str:
        return hashlib.blake2b(f"{salt}:{m}".encode("utf-8"), digest_size=8).hexdigest()

    return sorted(members, key=key)[0]


def residual_survival(
    matrix: ErrorMatrix,
    metric: Metric = clean_consensus_wrong,
    *,
    max_subsets: int = 200,
    seed: int = 0,
    min_k: int = 1,
) -> list[SurvivalPoint]:
    """C_k for k = min_k .. n_families, averaged over family subsets.

    When the number of subsets of size k exceeds `max_subsets`, a deterministic
    random sample of subsets is used. The sample is drawn without replacement so
    a subset cannot be double-counted into the mean.
    """
    groups = _family_groups(matrix)
    families = sorted(groups)
    n_fam = len(families)
    if n_fam == 0:
        return []

    rng = np.random.default_rng(seed)
    points: list[SurvivalPoint] = []

    for k in range(min_k, n_fam + 1):
        all_subsets = list(combinations(families, k))
        if len(all_subsets) > max_subsets:
            idx = rng.choice(len(all_subsets), size=max_subsets, replace=False)
            subsets = [all_subsets[i] for i in sorted(idx)]
        else:
            subsets = all_subsets

        values: list[float] = []
        for subset in subsets:
            salt = f"{seed}:{k}:{'|'.join(subset)}"
            members = [_pick_representative(groups[f], salt) for f in subset]
            values.append(metric(matrix.select_members(members)))

        arr = np.array(values, dtype=float)
        finite = arr[~np.isnan(arr)]
        points.append(
            SurvivalPoint(
                k=k,
                mean=float(finite.mean()) if finite.size else float("nan"),
                sd=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
                lo=float(finite.min()) if finite.size else float("nan"),
                hi=float(finite.max()) if finite.size else float("nan"),
                n_subsets=len(subsets),
                values=[float(v) for v in values],
            )
        )
    return points


@dataclass(slots=True)
class ExcessSurvival:
    """The observed survival curve against the shared-difficulty model's own.

    Why this replaced a curve-shape extrapolation: the obvious way to ask
    "where is this curve heading" is to fit `C_k = c + a*exp(-b*k)` and read
    off `c`. That does not work here. Under conditional independence given one
    shared difficulty factor, C_k decays like a mixture of geometrics -- not an
    exponential plus a constant -- so a free constant absorbs the model
    mismatch and reports a floor whether or not one exists. Measured on
    simulated curves with and without a real core, the fitted floor and the
    nested-model comparison came out at 0.027 vs 0.012 and 0.98 vs 0.93: the
    same answer to two different truths.

    The principled version compares the observed curve to the *fitted
    hierarchical model's* curve over the same member subsets. That model has one
    shared item-difficulty factor and conditional independence given it, so its
    C_k necessarily falls to zero as k grows: it cannot represent a floor. Two
    signatures follow, and they separate the cases the shape test cannot:

    * no core -- the ratio observed/predicted stays near 1 at every k;
    * a real core -- the ratio grows with k, because the model's curve keeps
      falling while the observed one does not.

    At k = 1 the ratio is ~1 by construction (the fit reproduces the marginals),
    so the informative quantity is the *growth* of the ratio across k, not its
    level at any single k.
    """

    ks: list[int]
    observed: list[float]
    predicted: list[float]
    ratio: list[float]
    gap: list[float]
    sigma_u: float
    converged: bool

    @property
    def ratio_growth(self) -> float:
        """Ratio at the largest k minus the ratio at the smallest k."""
        finite = [
            (k, r) for k, r in zip(self.ks, self.ratio)
            if not (math.isnan(r) or math.isinf(r))
        ]
        if len(finite) < 2:
            return float("nan")
        return finite[-1][1] - finite[0][1]

    def to_dict(self) -> dict[str, object]:
        return {
            "k": self.ks,
            "observed": self.observed,
            "predicted_by_difficulty_model": self.predicted,
            "ratio": self.ratio,
            "gap": self.gap,
            "ratio_growth": self.ratio_growth,
            "sigma_u": self.sigma_u,
            "converged": self.converged,
            "note": (
                "Observed C_k^clean against the fitted hierarchical model's own "
                "C_k over the same family subsets. That model cannot produce a "
                "floor, so a ratio that grows with k is evidence of a residual "
                "the shared difficulty axis does not explain."
            ),
        }


def excess_survival(
    matrix: ErrorMatrix,
    *,
    metric_status: str | None = "unanimous",
    max_subsets: int = 200,
    seed: int = 0,
) -> ExcessSurvival | None:
    """Compare the observed survival curve to the shared-difficulty model's.

    The model is fitted on the same item subset the curve is measured on
    (`metric_status`), using the full ensemble; its per-member severities are
    then used to predict C_k for each family subset. Fitting per subset instead
    would let the model re-absorb the very structure being tested.
    """
    from ..metrics.dependence import fit_random_effects

    groups = _family_groups(matrix)
    families = sorted(groups)
    if len(families) < 2:
        return None

    mask = matrix.item_mask(metric_status)
    if not mask.any():
        return None
    sub = matrix.select_items(mask)
    fit = fit_random_effects(sub)
    member_index = {m: j for j, m in enumerate(sub.member_ids)}

    rng = np.random.default_rng(seed)
    ks: list[int] = []
    observed: list[float] = []
    predicted: list[float] = []

    for k in range(1, len(families) + 1):
        all_subsets = list(combinations(families, k))
        if len(all_subsets) > max_subsets:
            idx = rng.choice(len(all_subsets), size=max_subsets, replace=False)
            subsets = [all_subsets[i] for i in sorted(idx)]
        else:
            subsets = all_subsets

        obs_vals: list[float] = []
        member_subsets: list[list[int]] = []
        for subset in subsets:
            salt = f"{seed}:{k}:{'|'.join(subset)}"
            members = [_pick_representative(groups[f], salt) for f in subset]
            obs_vals.append(float(sub.select_members(members).all_wrong().mean()))
            member_subsets.append([member_index[m] for m in members])

        ks.append(k)
        observed.append(float(np.mean(obs_vals)))
        predicted.append(fit.predicted_survival(member_subsets))

    ratio = [
        (o / p) if p and not math.isnan(p) and p > 0 else float("nan")
        for o, p in zip(observed, predicted)
    ]
    gap = [o - p for o, p in zip(observed, predicted)]
    return ExcessSurvival(
        ks=ks, observed=observed, predicted=predicted, ratio=ratio, gap=gap,
        sigma_u=fit.sigma_u, converged=fit.converged,
    )


def survival_shape(
    points: Sequence[SurvivalPoint],
    *,
    zero_tol: float = 0.002,
    plateau_slope_tol: float = 0.05,
    excess: "ExcessSurvival | None" = None,
    ratio_growth_tol: float = 0.25,
    seed: int = 0,
) -> dict[str, object]:
    """Classify the curve into the protocol's three shapes, conservatively.

    `closure` requires the endpoint to be at or below `zero_tol`.

    `plateau` requires *both* a nonzero endpoint and a trailing relative slope
    smaller than `plateau_slope_tol` -- i.e. the last step actually flattened.
    A nonzero endpoint that is still falling steeply is reported as
    `still_falling`, which maps to decision gate B ("extend the diversity axis;
    do not yet claim an information-limited floor"), not gate C.

    The `caveat` field is always populated, because even a flat trailing slope
    at k=5 bounds the floor from above rather than establishing one.
    """
    if not points:
        return {"shape": "undetermined", "reason": "no survival points"}

    endpoint = points[-1]
    k_max = endpoint.k
    result: dict[str, object] = {
        "k_max": k_max,
        "endpoint": endpoint.mean,
        "curve": [p.to_dict() for p in points],
    }

    if excess is not None:
        result["excess_survival"] = excess.to_dict()
        growth = excess.ratio_growth
        if math.isnan(growth):
            result["excess_verdict"] = "undetermined"
        elif growth > ratio_growth_tol:
            result["excess_verdict"] = "residual_exceeds_shared_difficulty"
        else:
            result["excess_verdict"] = "consistent_with_shared_difficulty"

    if math.isnan(endpoint.mean):
        result.update(shape="undetermined", reason="endpoint metric is undefined (empty subset?)")
        return result

    if endpoint.mean <= zero_tol:
        result.update(
            shape="closure",
            reason=f"C_{k_max}^clean = {endpoint.mean:.4f} <= {zero_tol}",
            decision_gate="A",
            caveat=(
                "Closure on this task and this anchor definition only. A clean "
                "residual can reappear on a task with harder items or a stricter "
                "clean subset."
            ),
        )
        return result

    if len(points) < 2:
        result.update(
            shape="undetermined",
            reason="a single k gives no slope; the diversity axis was not varied",
            decision_gate="B",
            caveat="One point cannot distinguish a plateau from a decline.",
        )
        return result

    prev, last = points[-2], points[-1]
    step = prev.mean - last.mean
    rel_slope = step / prev.mean if prev.mean > 0 else 0.0
    result["trailing_step"] = step
    result["trailing_relative_slope"] = rel_slope

    if rel_slope <= plateau_slope_tol:
        result.update(
            shape="apparent_plateau",
            reason=(
                f"C_{k_max}^clean = {last.mean:.4f} > 0 and the last step removed "
                f"only {rel_slope:.1%} of the residual"
            ),
            decision_gate="C",
            caveat=(
                f"A flat trailing slope at k={k_max} bounds an information-limited "
                "floor from above; it does not establish one. Extend the diversity "
                "axis before treating this as a floor (protocol section 18)."
            ),
        )
    else:
        verdict = result.get("excess_verdict")
        caveat = (
            "The curve has not flattened. Add families before interpreting the "
            "endpoint as a residual."
        )
        if verdict == "residual_exceeds_shared_difficulty":
            growth = (result.get("excess_survival") or {}).get("ratio_growth")
            caveat += (
                f" The observed curve is pulling away from the shared-difficulty "
                f"model's curve (ratio growth {growth:.2f} across k), which that "
                "model cannot produce and is the signature of a genuine residual. "
                "The gate stays B until the curve itself flattens, but this is the "
                "case worth spending more families on."
            )
        elif verdict == "consistent_with_shared_difficulty":
            caveat += (
                " The observed curve tracks the shared-difficulty model's curve "
                "across k, and that model has no floor: these data are consistent "
                "with the residual going to zero given enough families."
            )
        result.update(
            shape="still_falling",
            reason=(
                f"C_{k_max}^clean = {last.mean:.4f} > 0 but the last step still "
                f"removed {rel_slope:.1%} of the residual"
            ),
            decision_gate="B",
            caveat=caveat,
        )
    return result


def contested_transition(
    matrix: ErrorMatrix,
    *,
    max_subsets: int = 200,
    seed: int = 0,
) -> dict[str, object]:
    """Detect the third shape: clean residual closes while contested persists.

    Runs the survival curve twice -- once on unanimous items, once on contested
    ones -- because the protocol's third pattern is a statement about the two
    subsets moving in different directions, not about either alone.
    """
    clean_curve = residual_survival(
        matrix, clean_consensus_wrong, max_subsets=max_subsets, seed=seed
    )
    contested_metric: Metric = lambda m: consensus_wrong(m, "contested")
    contested_curve = residual_survival(
        matrix, contested_metric, max_subsets=max_subsets, seed=seed
    )
    disagreement_curve = residual_survival(
        matrix, disagreement_rate, max_subsets=max_subsets, seed=seed
    )

    clean_shape = survival_shape(clean_curve)
    clean_end = clean_curve[-1].mean if clean_curve else float("nan")
    contested_end = contested_curve[-1].mean if contested_curve else float("nan")

    transition = (
        clean_shape.get("shape") == "closure"
        and not math.isnan(contested_end)
        and contested_end > 0.01
    )
    out: dict[str, object] = {
        "clean_curve": [p.to_dict() for p in clean_curve],
        "contested_curve": [p.to_dict() for p in contested_curve],
        "disagreement_curve": [p.to_dict() for p in disagreement_curve],
        "clean_endpoint": clean_end,
        "contested_endpoint": contested_end,
        "contested_transition": transition,
    }
    if transition:
        out["interpretation"] = (
            "Clean consensus failure closes under internal diversity while "
            "contested-anchor failure persists: the remaining problem is anchor "
            "formation and adjudication, not epistemic grounding (protocol "
            "section 10, third row)."
        )
        out["decision_gate"] = "A"
    return out
