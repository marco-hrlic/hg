"""The error matrix E_ij = 1[y_hat_ij != y_i^ref].

Derived from item-level rows, never stored as the primary artifact. Keeping
the dense array secondary is deliberate: an array cannot record which cells
are missing, which member produced which column, or what a row's anchor status
was, and all three are needed to interpret a consensus metric.

`ErrorMatrix` carries the alignment that the metrics rely on:

    errors      (n_items, n_members) bool, E_ij
    predictions (n_items, n_members) object, y_hat_ij -- needed for D_E, which
                asks whether members *agreed*, not whether they were right
    statuses    (n_items,) anchor status A_i
    families    per-column model family / lineage, for family-subset averaging
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..schema import (
    ANCHOR_CONTESTED,
    ANCHOR_UNANIMOUS,
    ItemResult,
    SchemaError,
    check_rectangular,
)


@dataclass(slots=True)
class ErrorMatrix:
    condition: str
    item_ids: list[str]
    member_ids: list[str]
    errors: np.ndarray  # (n_items, n_members) bool
    predictions: np.ndarray  # (n_items, n_members) object
    statuses: np.ndarray  # (n_items,) str
    confidences: np.ndarray  # (n_items, n_members) float, NaN when absent
    parse_ok: np.ndarray  # (n_items, n_members) bool
    families: list[str] = field(default_factory=list)
    lineages: list[str] = field(default_factory=list)
    task: str | None = None

    # -- shape -----------------------------------------------------------
    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def n_members(self) -> int:
        return len(self.member_ids)

    def __post_init__(self) -> None:
        if self.errors.shape != (self.n_items, self.n_members):
            raise SchemaError(
                f"errors shape {self.errors.shape} does not match "
                f"{self.n_items} items x {self.n_members} members"
            )
        if not self.families:
            self.families = list(self.member_ids)
        if not self.lineages:
            self.lineages = list(self.families)

    # -- subsetting ------------------------------------------------------
    def item_mask(self, status: str | Sequence[str] | None) -> np.ndarray:
        """Boolean mask over items for one or more anchor statuses."""
        if status is None:
            return np.ones(self.n_items, dtype=bool)
        wanted = {status} if isinstance(status, str) else set(status)
        return np.isin(self.statuses, list(wanted))

    @property
    def clean_mask(self) -> np.ndarray:
        """Unanimous items: the subset C_E^clean is defined on (section 6)."""
        return self.item_mask(ANCHOR_UNANIMOUS)

    @property
    def contested_mask(self) -> np.ndarray:
        return self.item_mask(ANCHOR_CONTESTED)

    def select_members(self, member_ids: Sequence[str]) -> "ErrorMatrix":
        """A sub-ensemble, used to average over family subsets of size k."""
        index = {m: j for j, m in enumerate(self.member_ids)}
        missing = [m for m in member_ids if m not in index]
        if missing:
            raise KeyError(f"unknown members: {missing}")
        cols = [index[m] for m in member_ids]
        return ErrorMatrix(
            condition=self.condition,
            item_ids=list(self.item_ids),
            member_ids=list(member_ids),
            errors=self.errors[:, cols],
            predictions=self.predictions[:, cols],
            statuses=self.statuses,
            confidences=self.confidences[:, cols],
            parse_ok=self.parse_ok[:, cols],
            families=[self.families[c] for c in cols],
            lineages=[self.lineages[c] for c in cols],
            task=self.task,
        )

    def select_items(self, mask: np.ndarray) -> "ErrorMatrix":
        mask = np.asarray(mask, dtype=bool)
        return ErrorMatrix(
            condition=self.condition,
            item_ids=[i for i, keep in zip(self.item_ids, mask) if keep],
            member_ids=list(self.member_ids),
            errors=self.errors[mask],
            predictions=self.predictions[mask],
            statuses=self.statuses[mask],
            confidences=self.confidences[mask],
            parse_ok=self.parse_ok[mask],
            families=list(self.families),
            lineages=list(self.lineages),
            task=self.task,
        )

    def resample_items(self, index: np.ndarray) -> "ErrorMatrix":
        """Take items by integer index, with replacement, for the bootstrap.

        Items are the primary independent sampling unit (section 12), so this
        is the only resampling axis the bootstrap uses.
        """
        index = np.asarray(index, dtype=int)
        return ErrorMatrix(
            condition=self.condition,
            item_ids=[self.item_ids[i] for i in index],
            member_ids=list(self.member_ids),
            errors=self.errors[index],
            predictions=self.predictions[index],
            statuses=self.statuses[index],
            confidences=self.confidences[index],
            parse_ok=self.parse_ok[index],
            families=list(self.families),
            lineages=list(self.lineages),
            task=self.task,
        )

    # -- derived per-item quantities -------------------------------------
    def all_wrong(self) -> np.ndarray:
        """(n_items,) bool -- every member wrong. The consensus-wrong event."""
        return self.errors.all(axis=1)

    def any_disagreement(self) -> np.ndarray:
        """(n_items,) bool -- some pair of members predicted different labels.

        Defined on predictions, not errors: two members can both be wrong in
        the same way (no disagreement, invisible failure) or wrong in different
        ways (disagreement, internally addressable). Separating those is the
        mechanism prediction P4 is about.
        """
        first = self.predictions[:, :1]
        return (self.predictions != first).any(axis=1)

    def error_count(self) -> np.ndarray:
        return self.errors.sum(axis=1)


def build_matrix(rows: Sequence[ItemResult], condition: str | None = None) -> ErrorMatrix:
    """Assemble an `ErrorMatrix` from item-level rows.

    Raises if the rows are ragged: a missing cell silently biases every
    consensus metric, so it is a hard error rather than an imputation.
    """
    if not rows:
        raise SchemaError("cannot build an error matrix from zero rows")
    conditions = {r.condition for r in rows}
    if condition is None:
        if len(conditions) != 1:
            raise SchemaError(
                f"rows span multiple conditions {sorted(conditions)}; pass condition="
            )
        condition = conditions.pop()
    else:
        rows = [r for r in rows if r.condition == condition]
        if not rows:
            raise SchemaError(f"no rows for condition {condition!r}")

    item_ids, member_ids = check_rectangular(rows)
    item_index = {v: i for i, v in enumerate(item_ids)}
    member_index = {v: j for j, v in enumerate(member_ids)}

    n_i, n_m = len(item_ids), len(member_ids)
    errors = np.zeros((n_i, n_m), dtype=bool)
    preds = np.empty((n_i, n_m), dtype=object)
    confidences = np.full((n_i, n_m), np.nan, dtype=float)
    parse_ok = np.ones((n_i, n_m), dtype=bool)
    statuses = np.empty(n_i, dtype=object)
    families: list[str | None] = [None] * n_m
    lineages: list[str | None] = [None] * n_m
    tasks = set()

    for r in rows:
        i, j = item_index[r.item_id], member_index[r.member_id]
        if r.correct is None:
            raise SchemaError(
                f"row (item={r.item_id}, member={r.member_id}) has correct=None; "
                "scoring must happen before the matrix is built"
            )
        errors[i, j] = not r.correct
        # Unparsed output is not a label. Using a sentinel rather than None
        # keeps it from silently colliding with a real prediction in the
        # disagreement comparison.
        preds[i, j] = r.prediction if r.prediction is not None else f"<unparsed:{j}>"
        parse_ok[i, j] = r.parse_ok
        if r.confidence is not None:
            confidences[i, j] = r.confidence
        if statuses[i] is None:
            statuses[i] = r.anchor_status
        elif statuses[i] != r.anchor_status:
            raise SchemaError(
                f"item {r.item_id} has conflicting anchor_status "
                f"{statuses[i]!r} vs {r.anchor_status!r}"
            )
        families[j] = r.model_family
        lineages[j] = r.model_family
        tasks.add(r.task)

    if any(s is None for s in statuses):
        raise SchemaError("some items have no anchor_status; cannot define the clean subset")

    return ErrorMatrix(
        condition=condition,
        item_ids=item_ids,
        member_ids=member_ids,
        errors=errors,
        predictions=preds,
        statuses=statuses.astype(str),
        confidences=confidences,
        parse_ok=parse_ok,
        families=[f or "unknown" for f in families],
        lineages=[l or "unknown" for l in lineages],
        task=(tasks.pop() if len(tasks) == 1 else None),
    )


def save_matrix(path: str | Path, matrix: ErrorMatrix) -> None:
    """Cache a derived matrix as npz. Convenience only -- JSONL rows are truth."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        condition=np.array(matrix.condition),
        item_ids=np.array(matrix.item_ids, dtype=object),
        member_ids=np.array(matrix.member_ids, dtype=object),
        errors=matrix.errors,
        predictions=matrix.predictions.astype(str),
        statuses=matrix.statuses,
        confidences=matrix.confidences,
        parse_ok=matrix.parse_ok,
        families=np.array(matrix.families, dtype=object),
        lineages=np.array(matrix.lineages, dtype=object),
        task=np.array(matrix.task if matrix.task is not None else ""),
    )


def load_matrix(path: str | Path) -> ErrorMatrix:
    data = np.load(Path(path), allow_pickle=True)
    task = str(data["task"])
    return ErrorMatrix(
        condition=str(data["condition"]),
        item_ids=[str(x) for x in data["item_ids"]],
        member_ids=[str(x) for x in data["member_ids"]],
        errors=data["errors"],
        predictions=data["predictions"].astype(object),
        statuses=data["statuses"].astype(str),
        confidences=data["confidences"],
        parse_ok=data["parse_ok"],
        families=[str(x) for x in data["families"]],
        lineages=[str(x) for x in data["lineages"]],
        task=task or None,
    )
