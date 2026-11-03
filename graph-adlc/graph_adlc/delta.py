"""Staged application of graph deltas to an isolated copy of the graph."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .models import Aspect, Edge, EdgeStatus, Graph, Node
from .validation import classify_update, validate_graph, validate_node_payload

MUTABLE_NODE_FIELDS = {"title", "body", "data"}


@dataclass
class ChangeSeed:
    """A primitive change event the influence engine starts from."""

    op_index: int
    node_id: str
    aspects: set[str]
    edge_id: str | None = None  # set for structure seeds originating at an edge
    edge_change: str | None = None  # "added" | "removed"
    note: str | None = None


@dataclass
class AppliedDelta:
    before: Graph
    after: Graph
    seeds: list[ChangeSeed] = field(default_factory=list)
    diff: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    aspect_overrides: list[dict[str, Any]] = field(default_factory=list)
    changed_node_ids: set[str] = field(default_factory=set)
    removed_edge_ids: set[str] = field(default_factory=set)
    added_edge_ids: set[str] = field(default_factory=set)
    unverified_claims: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.errors


def _merge_data(old: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge: patch keys replace keys; a ``None`` value deletes the key."""
    out = copy.deepcopy(old)
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_delta(before: Graph, delta: dict[str, Any], now: str) -> AppliedDelta:
    after = before.clone()
    res = AppliedDelta(before=before, after=after)

    base = delta.get("base_graph_revision")
    if base is None:
        res.errors.append("delta is missing base_graph_revision")
    elif base != before.revision:
        res.conflicts.append(
            f"base_graph_revision {base} is obsolete; current graph revision is {before.revision}"
        )

    ops = delta.get("operations")
    if not isinstance(ops, list) or not ops:
        res.errors.append("delta must contain a non-empty operations list")
        return res

    for i, op in enumerate(ops):
        kind = op.get("op") if isinstance(op, dict) else None
        handler = _HANDLERS.get(kind)
        if handler is None:
            res.errors.append(f"op[{i}]: unknown operation {kind!r}")
            continue
        handler(res, i, op, now)
        if op.get("contract_preserved") is True:
            res.unverified_claims.append(
                {"op_index": i, "claim": "contract_preserved", "target": op.get("id")}
            )

    if not res.errors:
        res.errors.extend(validate_graph(after))
    after.revision = before.revision + 1
    return res


def _check_expected(res: AppliedDelta, i: int, entity: Node | Edge | None, op: dict[str, Any], label: str) -> bool:
    if entity is None or entity.deleted:
        res.errors.append(f"op[{i}]: {label} {op.get('id')} does not exist")
        return False
    exp = op.get("expected_rev")
    if exp is None:
        res.errors.append(f"op[{i}]: expected_rev is required for {op['op']}")
        return False
    if exp != entity.rev:
        res.conflicts.append(
            f"op[{i}]: {label} {entity.id} expected_rev {exp} but current rev is {entity.rev}"
        )
        return False
    return True


def _add_node(res: AppliedDelta, i: int, op: dict[str, Any], now: str) -> None:
    spec = op.get("node") or {}
    nid = spec.get("id")
    if not nid or not isinstance(nid, str):
        res.errors.append(f"op[{i}]: add_node requires node.id")
        return
    existing = res.after.nodes.get(nid)
    if existing is not None:
        res.errors.append(
            f"op[{i}]: node id {nid} already exists{' (tombstoned; ids are never reused)' if existing.deleted else ''}"
        )
        return
    data = spec.get("data", {})
    issues = validate_node_payload(nid, spec.get("kind", ""), spec.get("title"), data)
    if issues:
        res.errors.extend(f"op[{i}]: {m}" for m in issues)
        return
    node = Node(
        id=nid, kind=spec["kind"], title=spec["title"], body=spec.get("body", ""),
        data=copy.deepcopy(data), rev=1, created_at=now, updated_at=now,
    )
    res.after.nodes[nid] = node
    res.changed_node_ids.add(nid)
    res.seeds.append(ChangeSeed(i, nid, {Aspect.STRUCTURE.value}, note="node added"))
    res.diff.append({"op_index": i, "op": "add_node", "id": nid, "before": None,
                     "after": node.to_dict(include_timestamps=False)})


def _update_node(res: AppliedDelta, i: int, op: dict[str, Any], now: str) -> None:
    nid = op.get("id")
    node = res.after.nodes.get(nid)
    if not _check_expected(res, i, node, op, "node"):
        return
    patch = op.get("patch") or {}
    illegal = set(patch) - MUTABLE_NODE_FIELDS
    if illegal:
        res.errors.append(f"op[{i}]: cannot patch fields {sorted(illegal)} (mutable: {sorted(MUTABLE_NODE_FIELDS)})")
        return
    if not patch:
        res.errors.append(f"op[{i}]: empty patch")
        return
    if "data" in patch and not isinstance(patch["data"], dict):
        res.errors.append(f"op[{i}]: patch.data must be an object")
        return
    before_dict = node.to_dict(include_timestamps=False)
    if "title" in patch:
        node.title = patch["title"]
    if "body" in patch:
        node.body = patch["body"]
    if "data" in patch:
        node.data = _merge_data(node.data, patch["data"])
    node.rev += 1
    node.updated_at = now
    decision = classify_update(node.kind, op.get("aspect"), patch)
    if decision.overridden:
        res.aspect_overrides.append(
            {"op_index": i, "node_id": nid, "declared": decision.declared,
             "effective": sorted(decision.effective), "reason": decision.note}
        )
    res.changed_node_ids.add(nid)
    res.seeds.append(ChangeSeed(i, nid, set(decision.effective)))
    res.diff.append({"op_index": i, "op": "update_node", "id": nid, "aspects": sorted(decision.effective),
                     "before": before_dict, "after": node.to_dict(include_timestamps=False),
                     "fields": _field_diff(before_dict, node.to_dict(include_timestamps=False))})


def _field_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("title", "body"):
        if before.get(k) != after.get(k):
            out[k] = {"from": before.get(k), "to": after.get(k)}
    bd, ad = before.get("data", {}), after.get("data", {})
    for k in sorted(set(bd) | set(ad)):
        if bd.get(k) != ad.get(k):
            out[f"data.{k}"] = {"from": bd.get(k), "to": ad.get(k)}
    return out


def _remove_node(res: AppliedDelta, i: int, op: dict[str, Any], now: str) -> None:
    nid = op.get("id")
    node = res.after.nodes.get(nid)
    if not _check_expected(res, i, node, op, "node"):
        return
    node.deleted = True
    node.rev += 1
    node.updated_at = now
    res.changed_node_ids.add(nid)
    res.seeds.append(ChangeSeed(i, nid, {Aspect.STRUCTURE.value}, note="node removed"))
    cascaded = []
    for e in res.after.live_edges():
        if nid in (e.src, e.dst):
            e.deleted = True
            e.rev += 1
            res.removed_edge_ids.add(e.id)
            cascaded.append(e.id)
            res.seeds.append(ChangeSeed(i, e.src, {Aspect.STRUCTURE.value}, e.id, "removed"))
            res.seeds.append(ChangeSeed(i, e.dst, {Aspect.STRUCTURE.value}, e.id, "removed"))
    res.diff.append({"op_index": i, "op": "remove_node", "id": nid,
                     "before": res.before.nodes[nid].to_dict(include_timestamps=False),
                     "after": None, "cascaded_edge_removals": cascaded})


def _add_edge(res: AppliedDelta, i: int, op: dict[str, Any], now: str) -> None:
    spec = op.get("edge") or {}
    eid = spec.get("id")
    if not eid or not isinstance(eid, str):
        res.errors.append(f"op[{i}]: add_edge requires edge.id")
        return
    if eid in res.after.edges:
        res.errors.append(f"op[{i}]: edge id {eid} already exists (ids are never reused)")
        return
    missing = [k for k in ("src", "dst", "relation") if not spec.get(k)]
    if missing:
        res.errors.append(f"op[{i}]: add_edge missing {missing}")
        return
    edge = Edge(id=eid, src=spec["src"], dst=spec["dst"], relation=spec["relation"],
                status=spec.get("status", EdgeStatus.CONFIRMED.value), rev=1,
                evidence_ref=spec.get("evidence_ref"))
    res.after.edges[eid] = edge
    res.added_edge_ids.add(eid)
    res.seeds.append(ChangeSeed(i, edge.src, {Aspect.STRUCTURE.value}, eid, "added"))
    res.seeds.append(ChangeSeed(i, edge.dst, {Aspect.STRUCTURE.value}, eid, "added"))
    res.diff.append({"op_index": i, "op": "add_edge", "id": eid, "before": None, "after": edge.to_dict()})


def _remove_edge(res: AppliedDelta, i: int, op: dict[str, Any], now: str) -> None:
    eid = op.get("id")
    edge = res.after.edges.get(eid)
    if not _check_expected(res, i, edge, op, "edge"):
        return
    before_dict = edge.to_dict()
    edge.deleted = True
    edge.rev += 1
    res.removed_edge_ids.add(eid)
    res.seeds.append(ChangeSeed(i, edge.src, {Aspect.STRUCTURE.value}, eid, "removed"))
    res.seeds.append(ChangeSeed(i, edge.dst, {Aspect.STRUCTURE.value}, eid, "removed"))
    res.diff.append({"op_index": i, "op": "remove_edge", "id": eid, "before": before_dict, "after": None})


_HANDLERS = {
    "add_node": _add_node,
    "update_node": _update_node,
    "remove_node": _remove_node,
    "add_edge": _add_edge,
    "remove_edge": _remove_edge,
}
