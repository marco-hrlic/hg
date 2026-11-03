"""Graph invariants, node payload schemas and aspect classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Aspect, EdgeStatus, Graph, NodeKind, Relation


class GraphError(Exception):
    """Base error for graph operations."""


class ValidationError(GraphError):
    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        super().__init__("; ".join(self.issues) if self.issues else "validation failed")


class RevisionConflict(GraphError):
    def __init__(self, issues: list[str]):
        self.issues = list(issues)
        super().__init__("; ".join(self.issues))


# --------------------------------------------------------------------------
# Edge endpoint rules (fixed directions from SPEC §1.2)
# --------------------------------------------------------------------------

K = NodeKind
ENDPOINTS: dict[str, tuple[set[str], set[str]]] = {
    Relation.REFINES: ({K.GOAL, K.CAPABILITY, K.REQUIREMENT}, {K.CAPABILITY, K.REQUIREMENT}),
    Relation.DEPENDS_ON: (
        {K.REQUIREMENT, K.COMPONENT, K.CAPABILITY},
        {K.REQUIREMENT, K.COMPONENT, K.CONTRACT, K.CAPABILITY},
    ),
    Relation.CONSTRAINED_BY: ({K.REQUIREMENT, K.COMPONENT, K.CAPABILITY}, {K.CONTRACT}),
    Relation.IMPLEMENTS: ({K.COMPONENT}, {K.REQUIREMENT}),
    Relation.VERIFIES: ({K.TEST}, {K.REQUIREMENT, K.CONTRACT}),
    Relation.REALIZED_BY: ({K.COMPONENT}, {K.ARTIFACT}),
    Relation.BLOCKS: (
        {K.QUESTION},
        {K.GOAL, K.CAPABILITY, K.REQUIREMENT, K.CONTRACT, K.COMPONENT},
    ),
}

CONTRACT_STATUSES = {"draft", "proposed", "accepted", "incompatible", "unresolved", "rejected", "deprecated"}
QUESTION_STATES = {"open", "resolved", "withdrawn"}
ASSUMPTION_STATUSES = {"proposed", "accepted", "rejected", "invalidated"}

# Data keys that may legitimately be updated under the ``evidence`` aspect
# without any propagation (bookkeeping only).
EVIDENCE_DATA_KEYS = {"last_result", "last_run_at", "evidence_ref", "notes"}


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def normalize_assumptions(raw: Any, owner_id: str) -> list[dict[str, Any]]:
    """Assumptions may be given as plain strings or dicts; normalize to dicts."""
    out: list[dict[str, Any]] = []
    for i, a in enumerate(raw or []):
        if isinstance(a, str):
            out.append({"id": f"{owner_id}#A{i + 1}", "statement": a, "status": "accepted"})
        elif isinstance(a, dict):
            out.append(
                {
                    "id": a.get("id", f"{owner_id}#A{i + 1}"),
                    "statement": a.get("statement", ""),
                    "status": a.get("status", "accepted"),
                    **({"about": a["about"]} if "about" in a else {}),
                    **({k: v for k, v in a.items() if k not in {"id", "statement", "status", "about"}}),
                }
            )
    return out


def validate_node_payload(node_id: str, kind: str, title: Any, data: Any) -> list[str]:
    issues: list[str] = []
    if kind not in set(NodeKind):
        return [f"{node_id}: unknown node kind {kind!r}"]
    if not isinstance(title, str) or not title.strip():
        issues.append(f"{node_id}: title must be a non-empty string")
    if not isinstance(data, dict):
        return issues + [f"{node_id}: data must be an object"]

    if kind == K.REQUIREMENT:
        if "acceptance_criteria" in data and not _is_str_list(data["acceptance_criteria"]):
            issues.append(f"{node_id}: acceptance_criteria must be a list of strings")
        if "open_questions" in data and not _is_str_list(data["open_questions"]):
            issues.append(f"{node_id}: open_questions must be a list of strings")
        if "assumptions" in data:
            if not isinstance(data["assumptions"], list):
                issues.append(f"{node_id}: assumptions must be a list")
            else:
                for a in data["assumptions"]:
                    if isinstance(a, dict):
                        if not isinstance(a.get("statement", ""), str) or not a.get("statement"):
                            issues.append(f"{node_id}: assumption needs a statement")
                        if a.get("status", "accepted") not in ASSUMPTION_STATUSES:
                            issues.append(f"{node_id}: invalid assumption status {a.get('status')!r}")
                    elif not isinstance(a, str):
                        issues.append(f"{node_id}: assumption must be a string or object")
    elif kind == K.CONTRACT:
        status = data.get("status")
        if status is None:
            issues.append(f"{node_id}: contract requires data.status")
        elif status not in CONTRACT_STATUSES:
            issues.append(f"{node_id}: invalid contract status {status!r}")
    elif kind == K.ARTIFACT:
        if not isinstance(data.get("path"), str) or not data.get("path"):
            issues.append(f"{node_id}: artifact requires data.path")
    elif kind == K.TEST:
        if "command" in data and not isinstance(data["command"], str):
            issues.append(f"{node_id}: test command must be a string")
    elif kind == K.QUESTION:
        if not isinstance(data.get("blocking"), bool):
            issues.append(f"{node_id}: question requires boolean data.blocking")
        if data.get("state") not in QUESTION_STATES:
            issues.append(f"{node_id}: question requires data.state in {sorted(QUESTION_STATES)}")
    elif kind == K.GOAL:
        if "success_criteria" in data and not _is_str_list(data["success_criteria"]):
            issues.append(f"{node_id}: success_criteria must be a list of strings")
    return issues


def validate_graph(g: Graph) -> list[str]:
    """Hard invariants. Returns a sorted list of issues (empty = valid)."""
    issues: list[str] = []
    for n in g.live_nodes():
        issues += validate_node_payload(n.id, n.kind, n.title, n.data)
        if not isinstance(n.rev, int) or n.rev < 1:
            issues.append(f"{n.id}: rev must be a positive integer")

    refine_parents: dict[str, list[str]] = {}
    for e in g.live_edges():
        if e.relation not in set(Relation):
            issues.append(f"{e.id}: unknown relation {e.relation!r}")
            continue
        if e.status not in set(EdgeStatus):
            issues.append(f"{e.id}: invalid edge status {e.status!r}")
        src, dst = g.node(e.src), g.node(e.dst)
        if src is None:
            issues.append(f"{e.id}: source {e.src} does not exist (or is deleted)")
        if dst is None:
            issues.append(f"{e.id}: destination {e.dst} does not exist (or is deleted)")
        if src is None or dst is None:
            continue
        if e.src == e.dst:
            issues.append(f"{e.id}: self-loop is not allowed")
        allowed_src, allowed_dst = ENDPOINTS[e.relation]
        if src.kind not in allowed_src or dst.kind not in allowed_dst:
            issues.append(
                f"{e.id}: {e.relation} not allowed from {src.kind} ({e.src}) to {dst.kind} ({e.dst})"
            )
        if e.relation == Relation.REFINES:
            refine_parents.setdefault(e.dst, []).append(e.src)

    # Duplicate (src, dst, relation) among live edges.
    seen: dict[tuple[str, str, str], str] = {}
    for e in g.live_edges():
        key = (e.src, e.dst, e.relation)
        if key in seen:
            issues.append(f"{e.id}: duplicates edge {seen[key]} ({e.src} -{e.relation}-> {e.dst})")
        else:
            seen[key] = e.id

    for child, parents in sorted(refine_parents.items()):
        if len(parents) > 1:
            issues.append(f"{child}: has multiple refinement parents {sorted(parents)}")

    # Refinement acyclicity.
    for start in sorted(refine_parents):
        cur, seen_nodes = start, {start}
        while cur in refine_parents:
            cur = refine_parents[cur][0]
            if cur in seen_nodes:
                issues.append(f"refinement cycle detected through {start}")
                break
            seen_nodes.add(cur)
    return sorted(set(issues))


# --------------------------------------------------------------------------
# Aspect classification (SPEC §2): caller may declare; engine validates.
# --------------------------------------------------------------------------


@dataclass
class AspectDecision:
    declared: str
    effective: set[str]
    note: str | None = None
    overridden: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def classify_update(kind: str, declared: str | None, patch: dict[str, Any]) -> AspectDecision:
    """Validate a declared aspect against the node kind and patched fields.

    Unknown/unsupported declarations are *conservatively upgraded*, never
    accepted as-is. In particular ``evidence`` is only honoured for pure
    bookkeeping keys on tests/artifacts, so it cannot be used to suppress impact.
    """
    decl = declared or Aspect.SEMANTIC.value
    if decl not in set(Aspect):
        return AspectDecision(decl, {Aspect.SEMANTIC.value}, f"unknown aspect {decl!r}; treated as semantic", True)

    data_keys = set((patch.get("data") or {}).keys())
    only_bookkeeping = set(patch) <= {"data"} and data_keys and data_keys <= EVIDENCE_DATA_KEYS

    def keep() -> AspectDecision:
        return AspectDecision(decl, {decl})

    def upgrade(to: set[str], why: str) -> AspectDecision:
        return AspectDecision(decl, to, why, True)

    if decl == Aspect.EVIDENCE:
        if kind in (K.TEST, K.ARTIFACT) and only_bookkeeping:
            return keep()
        fallback = Aspect.CONTRACT.value if kind == K.CONTRACT else (
            Aspect.IMPLEMENTATION.value if kind == K.ARTIFACT else Aspect.SEMANTIC.value
        )
        return upgrade(
            {fallback},
            f"'evidence' aspect only applies to bookkeeping keys {sorted(EVIDENCE_DATA_KEYS)} on tests/artifacts; "
            f"conservatively treated as {fallback}",
        )

    if decl == Aspect.STRUCTURE:
        fallback = Aspect.CONTRACT.value if kind == K.CONTRACT else Aspect.SEMANTIC.value
        return upgrade({fallback}, f"'structure' applies to node/edge add/remove; update treated as {fallback}")

    if kind == K.CONTRACT:
        if decl == Aspect.CONTRACT:
            return keep()
        if decl == Aspect.SEMANTIC:
            return upgrade(
                {Aspect.CONTRACT.value, Aspect.SEMANTIC.value},
                "semantic change to a contract is also a contract change",
            )
        return upgrade({Aspect.CONTRACT.value}, f"{decl!r} change to a contract treated as contract change")

    if kind in (K.COMPONENT, K.ARTIFACT):
        if decl in (Aspect.SEMANTIC, Aspect.IMPLEMENTATION):
            return keep()
        return upgrade({Aspect.SEMANTIC.value}, f"{decl!r} is not valid for a {kind}; treated as semantic")

    # goal / capability / requirement / test / question
    if decl == Aspect.SEMANTIC:
        return keep()
    return upgrade({Aspect.SEMANTIC.value}, f"{decl!r} is not valid for a {kind}; treated as semantic")
