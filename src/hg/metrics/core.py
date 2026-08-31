"""The primary metrics (protocol section 8).

    C_E            = P(E_i1 = ... = E_iE = 1)          consensus-wrong rate
    C_E^clean      = P(all wrong | A_i = unanimous)    clean consensus-wrong
    D_E            = P(exists j,k : y_hat_ij != y_hat_ik)  disagreement rate
    Q_contested    = P(A_i = contested | all wrong)    contested share of residual
    Delta_div      = C_homogeneous^clean / C_diverse^clean   internal diversity gap
    localization surplus = V_targeted - V_disagreement

Every function takes an `ErrorMatrix` and returns plain floats, so all of them
compose with the item bootstrap in `hg.metrics.bootstrap` without special
cases. NaN is returned for an empty subset rather than 0.0: "no unanimous items
survived the filter" and "no unanimous item defeated the whole ensemble" are
opposite conclusions and must not share a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..schema import ANCHOR_CONTESTED, ANCHOR_UNANIMOUS
from .matrix import ErrorMatrix


def _rate(numerator: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        denom = numerator.size
        num = int(numerator.sum())
    else:
        denom = int(mask.sum())
        num = int(numerator[mask].sum())
    if denom == 0:
        return float("nan")
    return num / denom


def consensus_wrong(matrix: ErrorMatrix, status: str | Sequence[str] | None = None) -> float:
    """C_E, optionally restricted to items of a given anchor status."""
    return _rate(matrix.all_wrong(), matrix.item_mask(status))


def clean_consensus_wrong(matrix: ErrorMatrix) -> float:
    """C_E^clean: consensus failure on unanimous-anchor items only.

    This is the protocol's primary empirical proxy for a clean systematic
    residual. Restricting to unanimous items is what separates "the ensemble
    is missing information" from "the humans did not agree what the answer is".
    """
    return _rate(matrix.all_wrong(), matrix.clean_mask)


def disagreement_rate(matrix: ErrorMatrix, status: str | Sequence[str] | None = None) -> float:
    """D_E: how often the ensemble visibly disagrees with itself."""
    return _rate(matrix.any_disagreement(), matrix.item_mask(status))


def contested_share_of_residual(matrix: ErrorMatrix) -> float:
    """Q_contested = P(A_i = contested | all wrong).

    Rising Q_contested under added diversity is prediction P3: what remains
    stubborn becomes increasingly a question about the anchor rather than about
    the models.
    """
    all_wrong = matrix.all_wrong()
    n_residual = int(all_wrong.sum())
    if n_residual == 0:
        return float("nan")
    contested_and_wrong = int((all_wrong & matrix.contested_mask).sum())
    return contested_and_wrong / n_residual


def clean_share_of_residual(matrix: ErrorMatrix) -> float:
    """P(A_i = unanimous | all wrong). The complement view of Q_contested."""
    all_wrong = matrix.all_wrong()
    n_residual = int(all_wrong.sum())
    if n_residual == 0:
        return float("nan")
    return int((all_wrong & matrix.clean_mask).sum()) / n_residual


def invisible_error_rate(matrix: ErrorMatrix) -> float:
    """P(all wrong AND no disagreement).

    The failure mode the architecture cares about: the ensemble is wrong and
    has no internal signal saying so. Distinct from C_E because members can be
    unanimously wrong on the label while still disagreeing (impossible in a
    binary task, common with three or more labels).
    """
    return _rate(matrix.all_wrong() & ~matrix.any_disagreement())


def member_accuracy(matrix: ErrorMatrix, status: str | Sequence[str] | None = None) -> dict[str, float]:
    """Per-member standalone accuracy, for the capability-matching check.

    Section 18: diversity can look useful merely because one weak member
    manufactures disagreement. Reporting the spread here is what makes that
    visible before it is interpreted as a diversity effect.
    """
    mask = matrix.item_mask(status)
    if not mask.any():
        return {m: float("nan") for m in matrix.member_ids}
    correct = ~matrix.errors[mask]
    return {
        m: float(correct[:, j].mean())
        for j, m in enumerate(matrix.member_ids)
    }


def capability_spread(matrix: ErrorMatrix) -> dict[str, float]:
    """Min/max/range of standalone accuracy across members."""
    accs = np.array(list(member_accuracy(matrix).values()), dtype=float)
    if accs.size == 0 or np.all(np.isnan(accs)):
        return {"min": float("nan"), "max": float("nan"), "range": float("nan")}
    return {
        "min": float(np.nanmin(accs)),
        "max": float(np.nanmax(accs)),
        "range": float(np.nanmax(accs) - np.nanmin(accs)),
    }


def diversity_gap(
    c_homogeneous_clean: float, c_diverse_clean: float
) -> float:
    """Delta_div = C_homogeneous^clean / C_diverse^clean.

    A ratio, so it is read as "diversity removed a factor of X of the apparent
    shared error". Returns +inf when the diverse condition's clean residual is
    exactly zero -- which is the interesting outcome (decision gate A), not an
    error, so it must not be swallowed by a guard clause.
    """
    if np.isnan(c_homogeneous_clean) or np.isnan(c_diverse_clean):
        return float("nan")
    if c_diverse_clean == 0.0:
        return float("inf") if c_homogeneous_clean > 0 else float("nan")
    return c_homogeneous_clean / c_diverse_clean


def diversity_gap_difference(c_homogeneous_clean: float, c_diverse_clean: float) -> float:
    """C_homogeneous^clean - C_diverse^clean.

    Reported alongside the ratio because the ratio is unstable near zero while
    the absolute reduction is what determines whether the effect matters
    practically.
    """
    return c_homogeneous_clean - c_diverse_clean


def localization_surplus(v_targeted: float, v_disagreement: float) -> float:
    """V_targeted - V_disagreement.

    Positive means anchored targeting bought something that diverse
    disagreement-based localization did not. Prediction P5 says this is
    expected to be <= 0 *unless* a clean shared core survives, which is why it
    is only computed on tasks that reach decision gate C.
    """
    return v_targeted - v_disagreement


@dataclass(slots=True)
class ConditionMetrics:
    """Every section 8 metric for one condition, plus the diagnostics needed
    to keep it from being over-read."""

    condition: str
    n_items: int
    n_members: int
    n_clean: int
    n_contested: int
    consensus_wrong: float
    clean_consensus_wrong: float
    contested_consensus_wrong: float
    disagreement_rate: float
    clean_disagreement_rate: float
    contested_share_of_residual: float
    clean_share_of_residual: float
    invisible_error_rate: float
    mean_member_error: float
    member_accuracy: dict[str, float] = field(default_factory=dict)
    capability_spread: dict[str, float] = field(default_factory=dict)
    parse_failure_rate: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "condition": self.condition,
            "n_items": self.n_items,
            "n_members": self.n_members,
            "n_clean": self.n_clean,
            "n_contested": self.n_contested,
            "C_E": self.consensus_wrong,
            "C_E_clean": self.clean_consensus_wrong,
            "C_E_contested": self.contested_consensus_wrong,
            "D_E": self.disagreement_rate,
            "D_E_clean": self.clean_disagreement_rate,
            "Q_contested": self.contested_share_of_residual,
            "Q_clean": self.clean_share_of_residual,
            "invisible_error_rate": self.invisible_error_rate,
            "mean_member_error": self.mean_member_error,
            "member_accuracy": self.member_accuracy,
            "capability_spread": self.capability_spread,
            "parse_failure_rate": self.parse_failure_rate,
        }


def condition_metrics(matrix: ErrorMatrix) -> ConditionMetrics:
    clean = matrix.clean_mask
    contested = matrix.contested_mask
    return ConditionMetrics(
        condition=matrix.condition,
        n_items=matrix.n_items,
        n_members=matrix.n_members,
        n_clean=int(clean.sum()),
        n_contested=int(contested.sum()),
        consensus_wrong=consensus_wrong(matrix),
        clean_consensus_wrong=clean_consensus_wrong(matrix),
        contested_consensus_wrong=consensus_wrong(matrix, ANCHOR_CONTESTED),
        disagreement_rate=disagreement_rate(matrix),
        clean_disagreement_rate=disagreement_rate(matrix, ANCHOR_UNANIMOUS),
        contested_share_of_residual=contested_share_of_residual(matrix),
        clean_share_of_residual=clean_share_of_residual(matrix),
        invisible_error_rate=invisible_error_rate(matrix),
        mean_member_error=float(matrix.errors.mean()),
        member_accuracy=member_accuracy(matrix),
        capability_spread=capability_spread(matrix),
        parse_failure_rate=float((~matrix.parse_ok).mean()),
    )


def error_conversion(
    homogeneous: ErrorMatrix, diverse: ErrorMatrix
) -> dict[str, float]:
    """Prediction P4: does diversity *convert* invisible failure into
    disagreement, rather than merely removing error?

    Requires the two matrices to cover the same items, since the quantity is a
    decomposition of the same item population under two ensembles.

    Reported on the clean subset, where the label is not in dispute:

        d_consensus     C_E^clean(H) - C_E^clean(D)     error removed from consensus
        d_disagreement  D_E^clean(D) - D_E^clean(H)     disagreement gained
        conversion      d_disagreement / d_consensus    >~1 means converted, not just removed
    """
    if homogeneous.item_ids != diverse.item_ids:
        raise ValueError(
            "error conversion compares two ensembles over the same items; "
            "matrices have different item sets"
        )
    d_consensus = clean_consensus_wrong(homogeneous) - clean_consensus_wrong(diverse)
    d_disagree = disagreement_rate(diverse, ANCHOR_UNANIMOUS) - disagreement_rate(
        homogeneous, ANCHOR_UNANIMOUS
    )
    conversion = float("nan")
    if not np.isnan(d_consensus) and d_consensus > 0:
        conversion = d_disagree / d_consensus
    return {
        "delta_clean_consensus_wrong": d_consensus,
        "delta_clean_disagreement": d_disagree,
        "conversion_ratio": conversion,
    }
