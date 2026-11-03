"""Local, derived readiness of leaf requirements (SPEC §1.3).

Readiness is a *specification* property. Implementation/evidence freshness is a
separate dimension handled in :mod:`graph_adlc.lifecycle`.
"""

from __future__ import annotations

from .models import Graph, NodeKind, Readiness, ReadinessReason, ReadinessResult, Relation
from .validation import normalize_assumptions

BLOCKING_CONTRACT_STATUSES = {"incompatible", "unresolved", "rejected"}
CONDITIONAL_CONTRACT_STATUSES = {"draft", "proposed", "deprecated"}


def readiness(g: Graph, requirement_id: str) -> ReadinessResult:
    node = g.node(requirement_id)
    if node is None:
        raise KeyError(requirement_id)
    if node.kind != NodeKind.REQUIREMENT:
        raise ValueError(f"{requirement_id} is a {node.kind}, not a requirement")

    blockers: list[ReadinessReason] = []
    conditions: list[ReadinessReason] = []

    if not g.is_leaf_requirement(requirement_id):
        children = [c for c in g.children_of(requirement_id) if g.node(c)]
        blockers.append(ReadinessReason(
            "not_leaf", "only leaf requirements are implementation candidates in v0.1", children))

    # 1. Blocking, unresolved questions directly linked to the requirement.
    for e in g.in_edges(requirement_id, Relation.BLOCKS):
        q = g.node(e.src)
        if q is None:
            continue
        open_q = q.data.get("state") == "open"
        if open_q and q.data.get("blocking"):
            blockers.append(ReadinessReason("open_blocking_question", f"{q.id}: {q.title}", [q.id, e.id]))
        elif open_q:
            conditions.append(ReadinessReason(
                "open_non_blocking_question", f"{q.id} is open but non-blocking: {q.title}", [q.id, e.id]))
        if e.hypothesized and open_q:
            conditions.append(ReadinessReason(
                "hypothesized_edge", f"blocking relation {e.id} is hypothesized", [e.id]))

    # 2. Behavioural acceptance criteria present.
    if not node.data.get("acceptance_criteria"):
        blockers.append(ReadinessReason("missing_acceptance_criteria", "acceptance criteria are empty", [node.id]))

    # 3. Required contracts and providers.
    for e in g.out_edges(requirement_id, Relation.CONSTRAINED_BY) + g.out_edges(requirement_id, Relation.DEPENDS_ON):
        target = g.node(e.dst)
        if target is None:
            continue
        if target.kind == NodeKind.CONTRACT:
            status = target.data.get("status")
            if status in BLOCKING_CONTRACT_STATUSES:
                blockers.append(ReadinessReason(
                    "contract_not_resolved", f"required contract {target.id} is {status}", [target.id, e.id]))
            elif status in CONDITIONAL_CONTRACT_STATUSES:
                conditions.append(ReadinessReason(
                    "contract_not_accepted", f"relies on contract {target.id} with status {status}",
                    [target.id, e.id]))
        if e.hypothesized:
            conditions.append(ReadinessReason(
                "hypothesized_edge", f"{e.relation} {e.id} -> {e.dst} is hypothesized, not confirmed",
                [e.dst, e.id]))

    # 4. Explicit, tracked assumptions.
    assumptions = normalize_assumptions(node.data.get("assumptions"), node.id)
    for a in assumptions:
        status = a.get("status", "accepted")
        ids = [a["id"]] + ([a["about"]] if a.get("about") else [])
        if status == "accepted":
            conditions.append(ReadinessReason("tracked_assumption", f"relies on assumption: {a['statement']}", ids))
        elif status == "proposed":
            blockers.append(ReadinessReason("unaccepted_assumption", f"assumption not accepted: {a['statement']}", ids))
        else:
            blockers.append(ReadinessReason("invalid_assumption", f"assumption {status}: {a['statement']}", ids))

    if blockers:
        state = Readiness.BLOCKED
    elif conditions and any(c.code != "open_non_blocking_question" for c in conditions):
        state = Readiness.CONDITIONAL
    else:
        state = Readiness.READY
    return ReadinessResult(requirement_id, node.rev, state.value, blockers, conditions,
                           [a for a in assumptions if a.get("status") == "accepted"])


def frontier(g: Graph) -> list[ReadinessResult]:
    return [readiness(g, rid) for rid in g.leaf_requirements()]


def readiness_diff(before: Graph, after: Graph) -> list[dict]:
    b = {r.requirement_id: r for r in frontier(before)}
    a = {r.requirement_id: r for r in frontier(after)}
    out = []
    for rid in sorted(set(a) | set(b)):
        old, new = b.get(rid), a.get(rid)
        old_s = old.readiness if old else None
        new_s = new.readiness if new else None
        if old_s != new_s:
            out.append({"requirement_id": rid, "from": old_s, "to": new_s})
    return out
