"""Proposal envelope (sidecar to the engine's delta format) and its parser.

A proposal is what a model (or a human) *suggests*. It is never applied
directly: it is validated (``validation.py``), reviewed (``review.py``) and
only accepted operations are translated into an ordinary engine delta.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from ..models import canonical_json
from .source import sha256_text

OP_KINDS = ("add_node", "update_node", "remove_node", "add_edge", "remove_edge", "no_change")
SUPPORT_LEVELS = ("supported", "hypothesized", "insufficient")
CHANGE_CLASSES = ("semantic", "editorial", "implementation", "contract", "structure")


class ProposalFormatError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("; ".join(issues))


@dataclass
class Citation:
    source_id: str
    sha256: str | None
    line_start: int | None
    line_end: int | None
    excerpt: str

    @classmethod
    def parse(cls, raw: Any, where: str, issues: list[str]) -> "Citation | None":
        if not isinstance(raw, dict):
            issues.append(f"{where}: citation must be an object")
            return None
        lines = raw.get("lines")
        ls = le = None
        if isinstance(lines, list) and len(lines) == 2 and all(isinstance(x, int) for x in lines):
            ls, le = lines
        elif lines is not None:
            issues.append(f"{where}: lines must be [start, end]")
        if not isinstance(raw.get("source_id"), str) or not raw.get("source_id"):
            issues.append(f"{where}: citation needs source_id")
            return None
        return cls(source_id=raw["source_id"], sha256=raw.get("sha256"), line_start=ls, line_end=le,
                   excerpt=raw.get("excerpt") if isinstance(raw.get("excerpt"), str) else "")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"source_id": self.source_id, "sha256": self.sha256, "excerpt": self.excerpt}
        if self.line_start is not None:
            d["lines"] = [self.line_start, self.line_end]
        return d


@dataclass
class ProposedOp:
    op_id: str
    kind: str
    payload: dict[str, Any]  # node / edge / id+patch etc., exactly as proposed
    reason: str
    support: str
    evidence: list[Citation]
    confidence: float | None = None  # recorded, never used to promote support
    change_class: str | None = None

    def target_ids(self) -> list[str]:
        p = self.payload
        if self.kind == "add_node":
            return [p.get("node", {}).get("id", "")]
        if self.kind == "add_edge":
            e = p.get("edge", {})
            return [e.get("src", ""), e.get("dst", "")]
        if self.kind == "no_change":
            return [p.get("target", "")]
        return [p.get("id", "")]

    def fingerprint(self) -> str:
        """Content identity of the suggestion, ignoring ids/evidence (used to detect resubmission)."""
        p = copy.deepcopy(self.payload)
        if self.kind == "add_edge":
            e = p.get("edge", {})
            p = {"edge": {k: e.get(k) for k in ("src", "dst", "relation")}}
        p.pop("expected_rev", None)
        return sha256_text(canonical_json({"kind": self.kind, "payload": p}))

    def to_dict(self) -> dict[str, Any]:
        return {"op_id": self.op_id, "kind": self.kind, **copy.deepcopy(self.payload), "reason": self.reason,
                "support": self.support, "source_evidence": [c.to_dict() for c in self.evidence],
                "confidence": self.confidence, "change_class": self.change_class}


@dataclass
class Proposal:
    proposal_id: str
    base_graph_revision: int
    intent: str
    operations: list[ProposedOp]
    unresolved_questions: list[dict[str, Any]] = field(default_factory=list)
    possible_contradictions: list[dict[str, Any]] = field(default_factory=list)
    origin: dict[str, Any] = field(default_factory=dict)  # prompt/model/call/source provenance

    def op(self, op_id: str) -> ProposedOp:
        for o in self.operations:
            if o.op_id == op_id:
                return o
        raise KeyError(op_id)

    def to_dict(self) -> dict[str, Any]:
        return {"proposal_id": self.proposal_id, "base_graph_revision": self.base_graph_revision,
                "intent": self.intent, "operations": [o.to_dict() for o in self.operations],
                "unresolved_questions": copy.deepcopy(self.unresolved_questions),
                "possible_contradictions": copy.deepcopy(self.possible_contradictions),
                "origin": copy.deepcopy(self.origin)}


_PAYLOAD_KEYS = {
    "add_node": ("node",),
    "update_node": ("id", "expected_rev", "patch", "aspect", "contract_preserved"),
    "remove_node": ("id", "expected_rev"),
    "add_edge": ("edge",),
    "remove_edge": ("id", "expected_rev"),
    "no_change": ("target",),
}


def extract_json(text: str) -> Any:
    """Parse model output: plain JSON, or the first fenced ```json block."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def parse_proposal(raw: Any, proposal_id: str, base_graph_revision: int,
                   origin: dict[str, Any] | None = None) -> Proposal:
    """Structural parse. Semantic/evidence/engine checks happen in validation.py.

    ``proposal_id`` and ``base_graph_revision`` are assigned by the caller, not
    trusted from model output.
    """
    issues: list[str] = []
    if not isinstance(raw, dict):
        raise ProposalFormatError(["proposal must be a JSON object"])
    ops_raw = raw.get("operations", [])
    if not isinstance(ops_raw, list):
        raise ProposalFormatError(["operations must be a list"])
    ops: list[ProposedOp] = []
    seen_ids: set[str] = set()
    for i, o in enumerate(ops_raw):
        where = f"operations[{i}]"
        if not isinstance(o, dict):
            issues.append(f"{where}: must be an object")
            continue
        kind = o.get("kind") or o.get("op")
        if kind not in OP_KINDS:
            issues.append(f"{where}: unknown kind {kind!r}")
            continue
        op_id = str(o.get("op_id") or f"op-{i + 1}")
        if op_id in seen_ids:
            op_id = f"{op_id}#{i + 1}"
        seen_ids.add(op_id)
        payload = {k: copy.deepcopy(o[k]) for k in _PAYLOAD_KEYS[kind] if k in o}
        support = o.get("support", "insufficient")
        if support not in SUPPORT_LEVELS:
            issues.append(f"{where}: support must be one of {SUPPORT_LEVELS}")
            support = "insufficient"
        cites = []
        for j, c in enumerate(o.get("source_evidence") or []):
            cit = Citation.parse(c, f"{where}.source_evidence[{j}]", issues)
            if cit:
                cites.append(cit)
        conf = o.get("confidence")
        cc = o.get("change_class")
        if cc is not None and cc not in CHANGE_CLASSES:
            issues.append(f"{where}: change_class must be one of {CHANGE_CLASSES}")
            cc = None
        ops.append(ProposedOp(op_id=op_id, kind=kind, payload=payload, reason=str(o.get("reason", "")),
                              support=support, evidence=cites,
                              confidence=conf if isinstance(conf, (int, float)) else None, change_class=cc))
    if issues:
        raise ProposalFormatError(issues)
    uq = raw.get("unresolved_questions") or []
    pc = raw.get("possible_contradictions") or []
    if not isinstance(uq, list) or not isinstance(pc, list):
        raise ProposalFormatError(["unresolved_questions and possible_contradictions must be lists"])
    return Proposal(proposal_id=proposal_id, base_graph_revision=base_graph_revision,
                    intent=str(raw.get("intent", "")), operations=ops, unresolved_questions=uq,
                    possible_contradictions=pc, origin=dict(origin or {}))
