"""Typed records and closed enums for the v0.1 specification graph."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class NodeKind(StrEnum):
    GOAL = "goal"
    CAPABILITY = "capability"
    REQUIREMENT = "requirement"
    CONTRACT = "contract"
    COMPONENT = "component"
    ARTIFACT = "artifact"
    TEST = "test"
    QUESTION = "question"


class Relation(StrEnum):
    REFINES = "refines"  # parent -> child
    DEPENDS_ON = "depends_on"  # consumer -> provider
    CONSTRAINED_BY = "constrained_by"  # subject -> contract
    IMPLEMENTS = "implements"  # component -> requirement
    VERIFIES = "verifies"  # test -> requirement/contract
    REALIZED_BY = "realized_by"  # component -> artifact
    BLOCKS = "blocks"  # question -> target


class EdgeStatus(StrEnum):
    CONFIRMED = "confirmed"
    HYPOTHESIZED = "hypothesized"


class Aspect(StrEnum):
    SEMANTIC = "semantic"
    CONTRACT = "contract"
    IMPLEMENTATION = "implementation"
    STRUCTURE = "structure"
    EVIDENCE = "evidence"


class Obligation(StrEnum):
    REVIEW_SPEC = "review_spec"
    REVIEW_IMPLEMENTATION = "review_implementation"
    RERUN_TEST = "rerun_test"
    CHECK_CONTRACT = "check_contract"
    INVESTIGATE_UNKNOWN = "investigate_unknown"


class Readiness(StrEnum):
    BLOCKED = "blocked"
    CONDITIONAL = "conditional"
    READY = "ready"


class WorkStatus(StrEnum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUBMITTED = "submitted"
    VERIFIED = "verified"
    BLOCKED = "blocked"
    NEEDS_REVALIDATION = "needs_revalidation"
    ABANDONED = "abandoned"


class Freshness(StrEnum):
    """Implementation/evidence freshness, deliberately separate from readiness."""

    UNIMPLEMENTED = "unimplemented"
    IN_PROGRESS = "in_progress"
    CURRENT = "current"
    NEEDS_REVALIDATION = "needs_revalidation"


def canonical_json(value: Any) -> str:
    """Deterministic JSON used for storage, fixtures and diffs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class Node:
    id: str
    kind: str
    title: str
    body: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    rev: int = 1
    deleted: bool = False
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self, include_timestamps: bool = True) -> dict[str, Any]:
        d = asdict(self)
        d["data"] = copy.deepcopy(self.data)
        if not include_timestamps:
            d.pop("created_at")
            d.pop("updated_at")
        if not self.deleted:
            d.pop("deleted")
        return d

    def clone(self) -> "Node":
        return Node(**{**asdict(self), "data": copy.deepcopy(self.data)})


@dataclass
class Edge:
    id: str
    src: str
    dst: str
    relation: str
    status: str = EdgeStatus.CONFIRMED.value
    rev: int = 1
    deleted: bool = False
    evidence_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not self.deleted:
            d.pop("deleted")
        if self.evidence_ref is None:
            d.pop("evidence_ref")
        return d

    def clone(self) -> "Edge":
        return Edge(**asdict(self))

    @property
    def hypothesized(self) -> bool:
        return self.status == EdgeStatus.HYPOTHESIZED


@dataclass
class Graph:
    """In-memory graph view. Tombstoned entities are retained (``deleted=True``)."""

    revision: int
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def clone(self) -> "Graph":
        return Graph(
            revision=self.revision,
            nodes={k: v.clone() for k, v in self.nodes.items()},
            edges={k: v.clone() for k, v in self.edges.items()},
        )

    # -- live views -------------------------------------------------------
    def node(self, node_id: str) -> Node | None:
        n = self.nodes.get(node_id)
        return n if n is not None and not n.deleted else None

    def live_nodes(self) -> list[Node]:
        return [n for _, n in sorted(self.nodes.items()) if not n.deleted]

    def live_edges(self) -> list[Edge]:
        return [e for _, e in sorted(self.edges.items()) if not e.deleted]

    def out_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        return [
            e
            for e in self.live_edges()
            if e.src == node_id and (relation is None or e.relation == relation)
        ]

    def in_edges(self, node_id: str, relation: str | None = None) -> list[Edge]:
        return [
            e
            for e in self.live_edges()
            if e.dst == node_id and (relation is None or e.relation == relation)
        ]

    def parent_of(self, node_id: str) -> str | None:
        parents = self.in_edges(node_id, Relation.REFINES)
        return parents[0].src if parents else None

    def ancestors(self, node_id: str) -> list[str]:
        """Refinement ancestors, nearest first. Cycle-guarded."""
        chain: list[str] = []
        seen = {node_id}
        cur = self.parent_of(node_id)
        while cur is not None and cur not in seen:
            chain.append(cur)
            seen.add(cur)
            cur = self.parent_of(cur)
        return chain

    def children_of(self, node_id: str) -> list[str]:
        return [e.dst for e in self.out_edges(node_id, Relation.REFINES)]

    def is_leaf_requirement(self, node_id: str) -> bool:
        n = self.node(node_id)
        if n is None or n.kind != NodeKind.REQUIREMENT:
            return False
        return not any(
            (c := self.node(child)) is not None and c.kind == NodeKind.REQUIREMENT
            for child in self.children_of(node_id)
        )

    def leaf_requirements(self) -> list[str]:
        return [n.id for n in self.live_nodes() if self.is_leaf_requirement(n.id)]

    def revs(self, node_ids: list[str] | set[str]) -> dict[str, int]:
        return {nid: self.nodes[nid].rev for nid in sorted(node_ids) if nid in self.nodes}

    # -- snapshots ----------------------------------------------------------
    def to_snapshot(self, include_deleted: bool = True) -> dict[str, Any]:
        nodes = [
            n.to_dict(include_timestamps=False)
            for _, n in sorted(self.nodes.items())
            if include_deleted or not n.deleted
        ]
        edges = [
            e.to_dict()
            for _, e in sorted(self.edges.items())
            if include_deleted or not e.deleted
        ]
        return {"graph_revision": self.revision, "nodes": nodes, "edges": edges}


@dataclass
class ReadinessReason:
    code: str
    message: str
    node_ids: list[str] = field(default_factory=list)


@dataclass
class ReadinessResult:
    requirement_id: str
    rev: int
    readiness: str
    blockers: list[ReadinessReason] = field(default_factory=list)
    conditions: list[ReadinessReason] = field(default_factory=list)
    assumptions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def implementable(self) -> bool:
        return self.readiness in (Readiness.READY, Readiness.CONDITIONAL)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ImpactRecord:
    target_id: str
    obligation: str
    origin_id: str
    path: list[str]
    reason: str
    confidence: str  # "high" | "provisional" | "low"
    change_ref: str | None = None  # op index during propose; change id after commit
    provisional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrunedBoundary:
    consumer_id: str
    provider_id: str
    edge_id: str
    evidence_id: str
    contracts: dict[str, int]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuggestedAction:
    rank: int
    tier: int
    action: str
    target_ids: list[str]
    score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
