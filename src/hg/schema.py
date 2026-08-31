"""The item-level result record.

Protocol section 7 lists what every model run must store; section 14 fixes the
minimum column set. This module is the single definition of that record, so a
prediction matrix written by any backend can be re-analyzed later -- with
dependence measures that did not exist when the run happened -- without
touching a GPU.

Design notes
------------
* `ItemResult` is deliberately flat and JSON-serializable. The prediction
  matrix is stored as JSONL, not pickles: the file has to outlive the code.
* Everything needed to reconstruct provenance travels *with each row*, not
  in a sidecar. A row that has lost its adapter hash or split id is not
  analyzable, and losing that link is the most common way an ensemble
  dependence result becomes uninterpretable.
* `logprob` / `probs` / `embedding` are optional because not every backend
  exposes them, but the analysis code must degrade rather than crash: see
  `hg.routing.strategies` for confidence fallbacks.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

# Anchor status values (protocol section 6). Ordered weakest-agreement-last so
# `sorted()` on this vocabulary is meaningful in reports.
ANCHOR_UNANIMOUS = "unanimous"
ANCHOR_MAJORITY = "majority"
ANCHOR_CONTESTED = "contested"
ANCHOR_STATUSES = (ANCHOR_UNANIMOUS, ANCHOR_MAJORITY, ANCHOR_CONTESTED)

# Ensemble construction conditions (protocol section 4).
CONDITIONS = ("HS", "HD", "DS", "DD")

# The minimum column set from protocol section 14, in order. Kept as an
# explicit tuple because it is a contract with downstream analysis, not an
# incidental property of the dataclass field order.
MINIMUM_COLUMNS = (
    "item_id",
    "model_family",
    "model_version",
    "adapter_seed",
    "train_split",
    "prediction",
    "score",
    "confidence",
    "anchor_label",
    "anchor_status",
    "correct",
)


class SchemaError(ValueError):
    """A row or run violates the item-level result contract."""


@dataclass(slots=True)
class ItemResult:
    """One (item, model member) cell of the prediction matrix."""

    # --- identity of the item -------------------------------------------
    item_id: str
    task: str

    # --- identity of the ensemble member --------------------------------
    model_family: str
    model_version: str  # exact checkpoint, not a friendly name
    member_id: str  # unique within a condition; ensemble index handle
    condition: str  # one of CONDITIONS
    adapter_seed: int | None = None
    adapter_hash: str | None = None
    train_split: str | None = None
    prompt_version: str | None = None

    # --- what the model did ---------------------------------------------
    prediction: str | None = None
    raw_output: str | None = None
    parse_ok: bool = True
    score: float | None = None  # task score, e.g. per-item loss or 0/1
    confidence: float | None = None  # calibrated-ish P(prediction)
    probs: dict[str, float] | None = None  # full class distribution if available
    logprob: float | None = None  # token/sequence log-probability
    embedding: list[float] | None = None  # optional representation

    # --- decoding provenance --------------------------------------------
    inference_seed: int | None = None
    decoding: dict[str, Any] = field(default_factory=dict)

    # --- anchoring and correctness --------------------------------------
    anchor_label: str | None = None  # y_i^ref
    anchor_status: str | None = None  # A_i
    correct: bool | None = None  # E_ij = 1 - correct

    def validate(self) -> None:
        if self.condition not in CONDITIONS:
            raise SchemaError(
                f"unknown condition {self.condition!r}; expected one of {CONDITIONS}"
            )
        if self.anchor_status is not None and self.anchor_status not in ANCHOR_STATUSES:
            raise SchemaError(
                f"unknown anchor_status {self.anchor_status!r}; "
                f"expected one of {ANCHOR_STATUSES}"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise SchemaError(f"confidence {self.confidence} outside [0, 1]")
        if self.probs:
            total = sum(self.probs.values())
            if not 0.99 <= total <= 1.01:
                raise SchemaError(f"probs sum to {total:.4f}, expected 1.0")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != {}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ItemResult":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise SchemaError(f"unexpected fields in row: {sorted(unknown)}")
        return cls(**d)  # type: ignore[arg-type]


@dataclass(slots=True)
class RunManifest:
    """Provenance for one condition's worth of prediction rows.

    Written next to every prediction matrix. The point is that a results
    directory answers "what produced this?" on its own, months later.
    """

    run_id: str
    condition: str
    task: str
    dataset_config: str
    experiment_config: str
    members: list[dict[str, Any]] = field(default_factory=list)
    backend: str | None = None
    # Jaccard overlap O_jk between members' training splits (section 13).
    train_overlap: dict[str, float] = field(default_factory=dict)
    git_commit: str | None = None
    created_at: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RunManifest":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})  # type: ignore[arg-type]


def check_minimum_columns(row: Mapping[str, Any]) -> None:
    """Assert the section 14 minimum schema is present (values may be null)."""
    missing = [c for c in MINIMUM_COLUMNS if c not in row]
    if missing:
        raise SchemaError(f"row is missing required columns: {missing}")


def check_rectangular(rows: Sequence[ItemResult]) -> tuple[list[str], list[str]]:
    """Validate that rows form a complete item x member matrix.

    Returns `(item_ids, member_ids)` in stable order. A ragged matrix silently
    biases every consensus metric -- an item scored by four of five members
    cannot be "all wrong" on the same footing -- so this is checked loudly
    rather than filled with defaults.
    """
    items: list[str] = []
    members: list[str] = []
    seen_items: set[str] = set()
    seen_members: set[str] = set()
    cells: set[tuple[str, str]] = set()

    for r in rows:
        if r.item_id not in seen_items:
            seen_items.add(r.item_id)
            items.append(r.item_id)
        if r.member_id not in seen_members:
            seen_members.add(r.member_id)
            members.append(r.member_id)
        key = (r.item_id, r.member_id)
        if key in cells:
            raise SchemaError(f"duplicate cell for item {r.item_id!r} member {r.member_id!r}")
        cells.add(key)

    expected = len(items) * len(members)
    if len(cells) != expected:
        gaps = [
            (i, m) for i in items for m in members if (i, m) not in cells
        ][:5]
        raise SchemaError(
            f"prediction matrix is ragged: {len(cells)} cells for "
            f"{len(items)} items x {len(members)} members; first gaps: {gaps}"
        )
    return items, members


def rows_to_jsonl(rows: Iterable[ItemResult]) -> Iterable[str]:
    for r in rows:
        yield json.dumps(r.to_dict(), sort_keys=True, separators=(",", ":"))
