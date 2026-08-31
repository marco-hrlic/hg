"""The human-anchor data model (protocol section 6).

The rule this module exists to enforce: do not collapse a human panel into a
majority label and throw the disagreement away. Raw judgments are stored, and
an *anchor status* is derived alongside the reference label:

    unanimous   every annotator agreed
    majority    a clear majority, but not unanimous
    contested   no clear majority (or the majority is at/below threshold)

Why this matters for the experiment: `C_E^clean` -- the primary proxy for a
clean systematic residual -- is conditioned on `unanimous` items. If contested
items leak in, a persistent residual can be pure anchor disagreement wearing
the costume of an information-limited floor, which is exactly the confound
section 18 lists under "human-label ambiguity".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..io import read_jsonl
from ..schema import (
    ANCHOR_CONTESTED,
    ANCHOR_MAJORITY,
    ANCHOR_UNANIMOUS,
    SchemaError,
)


@dataclass(slots=True)
class PanelItem:
    """An input plus the raw panel judgments y_i1^H .. y_ir^H."""

    item_id: str
    x: str  # the input shown to annotators and to models
    judgments: list[str]
    # Optional externally supplied reference label. When present it wins over
    # the derived majority: a curated gold label is a different object from a
    # panel majority, and conflating them hides adjudication decisions.
    reference: str | None = None
    annotator_ids: list[str] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.judgments:
            raise SchemaError(f"item {self.item_id}: panel has no judgments")
        if self.annotator_ids is not None and len(self.annotator_ids) != len(self.judgments):
            raise SchemaError(
                f"item {self.item_id}: {len(self.judgments)} judgments but "
                f"{len(self.annotator_ids)} annotator ids"
            )


@dataclass(slots=True)
class AnchoredItem:
    """A panel item resolved into (y_i^ref, A_i) with the raw panel retained."""

    item_id: str
    x: str
    judgments: list[str]
    reference: str
    status: str
    # Share of the panel backing the reference label; 1.0 when unanimous.
    agreement: float
    n_annotators: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        """Clean == unanimous. Used to define the C_E^clean subset."""
        return self.status == ANCHOR_UNANIMOUS


def reference_label(judgments: Sequence[str], tie_break: str = "lexicographic") -> str:
    """Plurality label over a panel, with a deterministic tie-break.

    A tie is real information (it means the item is contested), so the label
    picked here is only a placeholder for scoring; `anchor_status` is what
    downstream analysis conditions on. The tie-break is deterministic so that
    reruns are comparable rather than seed-dependent.
    """
    if not judgments:
        raise SchemaError("cannot derive a reference label from an empty panel")
    counts = Counter(judgments)
    top = max(counts.values())
    tied = sorted(label for label, c in counts.items() if c == top)
    if len(tied) == 1 or tie_break == "lexicographic":
        return tied[0]
    if tie_break == "first":
        for j in judgments:
            if j in tied:
                return j
    raise SchemaError(f"unknown tie_break {tie_break!r}")


def anchor_status(
    judgments: Sequence[str],
    majority_threshold: float = 0.5,
    contested_margin: float = 0.0,
) -> tuple[str, float]:
    """Derive A_i and the agreement share backing the modal label.

    Parameters
    ----------
    majority_threshold:
        Modal share must *exceed* this to count as a majority. The default 0.5
        is a strict majority; raise it (e.g. 0.66) when the task's annotation
        guidelines treat a bare majority as unreliable.
    contested_margin:
        Extra margin the modal label must hold over the runner-up. With three
        annotators and labels {a, a, b}, share is 0.67 and margin 0.33; setting
        `contested_margin=0.4` would push that item to contested. Useful when
        panels are small and a single annotator flips the majority.
    """
    n = len(judgments)
    if n == 0:
        raise SchemaError("cannot derive anchor status from an empty panel")
    counts = Counter(judgments)
    ordered = counts.most_common()
    top_share = ordered[0][1] / n
    runner_up_share = ordered[1][1] / n if len(ordered) > 1 else 0.0
    margin = top_share - runner_up_share

    if len(counts) == 1:
        return ANCHOR_UNANIMOUS, 1.0
    if top_share > majority_threshold and margin >= contested_margin:
        return ANCHOR_MAJORITY, top_share
    return ANCHOR_CONTESTED, top_share


def build_anchored_items(
    panel: Iterable[PanelItem],
    majority_threshold: float = 0.5,
    contested_margin: float = 0.0,
    tie_break: str = "lexicographic",
) -> list[AnchoredItem]:
    """Resolve a panel into anchored items, keeping the raw judgments."""
    out: list[AnchoredItem] = []
    for item in panel:
        status, agreement = anchor_status(
            item.judgments,
            majority_threshold=majority_threshold,
            contested_margin=contested_margin,
        )
        ref = item.reference
        if ref is None:
            ref = reference_label(item.judgments, tie_break=tie_break)
        elif status == ANCHOR_UNANIMOUS and ref != item.judgments[0]:
            # A curated label that disagrees with a unanimous panel is not a
            # clean anchor; it is an adjudication override and must not be
            # counted in the clean subset.
            status, agreement = ANCHOR_CONTESTED, 0.0
        out.append(
            AnchoredItem(
                item_id=item.item_id,
                x=item.x,
                judgments=list(item.judgments),
                reference=ref,
                status=status,
                agreement=agreement,
                n_annotators=len(item.judgments),
                meta=dict(item.meta),
            )
        )
    return out


def load_panel(path: str | Path) -> list[PanelItem]:
    """Read a panel JSONL file.

    Expected fields per line: `item_id`, `x`, `judgments` (list), and
    optionally `reference`, `annotator_ids`, `meta`.
    """
    items: list[PanelItem] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        missing = [k for k in ("item_id", "x", "judgments") if k not in row]
        if missing:
            raise SchemaError(f"{path}: panel row missing {missing}")
        item_id = str(row["item_id"])
        if item_id in seen:
            raise SchemaError(f"{path}: duplicate item_id {item_id!r}")
        seen.add(item_id)
        items.append(
            PanelItem(
                item_id=item_id,
                x=str(row["x"]),
                judgments=[str(j) for j in row["judgments"]],
                reference=(str(row["reference"]) if row.get("reference") is not None else None),
                annotator_ids=(
                    [str(a) for a in row["annotator_ids"]]
                    if row.get("annotator_ids") is not None
                    else None
                ),
                meta=dict(row.get("meta") or {}),
            )
        )
    return items


def save_panel(path: str | Path, panel: Sequence[PanelItem]) -> int:
    from ..io import write_jsonl

    rows: list[dict[str, Any]] = []
    for item in panel:
        row: dict[str, Any] = {
            "item_id": item.item_id,
            "x": item.x,
            "judgments": item.judgments,
        }
        if item.reference is not None:
            row["reference"] = item.reference
        if item.annotator_ids is not None:
            row["annotator_ids"] = item.annotator_ids
        if item.meta:
            row["meta"] = item.meta
        rows.append(row)
    return write_jsonl(path, rows)


def status_counts(items: Iterable[AnchoredItem]) -> dict[str, int]:
    counts = {ANCHOR_UNANIMOUS: 0, ANCHOR_MAJORITY: 0, ANCHOR_CONTESTED: 0}
    for it in items:
        counts[it.status] += 1
    return counts


def clean_subset(items: Sequence[AnchoredItem]) -> list[AnchoredItem]:
    return [it for it in items if it.is_clean]


def anchor_maps(
    items: Iterable[AnchoredItem],
) -> tuple[dict[str, str], dict[str, str]]:
    """`(item_id -> reference label, item_id -> anchor status)`.

    Inference and scoring both need these, and they must come from one place
    so a row's `anchor_status` can never disagree with the analysis subset.
    """
    ref: dict[str, str] = {}
    status: dict[str, str] = {}
    for it in items:
        ref[it.item_id] = it.reference
        status[it.item_id] = it.status
    return ref, status


def validate_labels(items: Iterable[AnchoredItem], labels: Sequence[str]) -> None:
    """Every judgment and reference must live in the task's declared label space."""
    allowed = set(labels)
    for it in items:
        bad = sorted({j for j in it.judgments if j not in allowed})
        if bad:
            raise SchemaError(
                f"item {it.item_id}: judgments {bad} outside label space {sorted(allowed)}"
            )
        if it.reference not in allowed:
            raise SchemaError(
                f"item {it.item_id}: reference {it.reference!r} outside label space"
            )
