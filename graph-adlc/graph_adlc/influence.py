"""Directional, aspect-sensitive influence propagation with explanation paths.

The output is a set of *candidate verification obligations* (SPEC §3.1), not a
proof of complete semantic impact: the domain is complete only relative to the
registered edges and rules below. Hypothesized edges are traversed (never used
to prune) and additionally raise ``investigate_unknown``.

Rule table (event at node N with aspect A; only "origin" events — real changes —
and explicitly non-terminal events propagate):

=====================  ===============================  ===================================  =========
aspect at N            edge (direction from N)          obligation on other endpoint         terminal
=====================  ===============================  ===================================  =========
semantic               refines  N -> child              review_spec (recursive downward)     no
semantic (origin)      refines  parent -> N             review_spec on parent                yes
semantic (origin)      blocks   N(question) -> target   review_spec                          yes
semantic (origin)      implements component -> N(req)   review_implementation (impl aspect)  no
semantic (origin)      verifies test -> N               rerun_test                           yes
semantic (origin)      depends_on consumer -> N         check_contract                       yes
semantic (origin)      implements N(component) -> req   check_contract                       yes
contract (origin)      constrained_by subject -> N      check_contract (contract aspect)     no
contract (origin)      verifies test -> N               rerun_test                           yes
contract               depends_on consumer -> N         check_contract                       yes
contract (subject)     verifies test -> N(req)          rerun_test                           yes
contract (subject)     implements component -> N(req)   review_implementation (impl aspect)  no
contract (subject)     implements N(component) -> req   check_contract                       yes
implementation         realized_by component -> N(art)  review_implementation (impl aspect)  no
implementation         implements N(component) -> req   check_contract                       yes
implementation         depends_on consumer -> N         check_contract, provisional unless   yes
                                                        pruned by contract-preservation
                                                        evidence (SPEC §3.2 pruning policy)
structure (edge op)    both endpoints of added/removed edge, per relation (see STRUCTURE_RULES)  yes
=====================  ===============================  ===================================  =========

Rule sets. ``v1`` (the default) is exactly the table above. ``v1.1`` adds one rule:

=====================  ===============================  ===================================  =========
any, terminal          implements C(component) -> req   check_contract (contract aspect)     yes
check_contract on C,   (C reached as the consumer of a
via depends_on         changed provider)
=====================  ===============================  ===================================  =========

Under v1 a consumer's ``check_contract`` is terminal, so a broken consumer never reaches the requirement it
implements (v0.2 "premature terminal propagation"; v0.3 ledgerlite REQ-CSV-EXPORT). v1.1 takes exactly one more
hop and stops. It inherits the consumer's provisional and hypothesized flags.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .delta import AppliedDelta, ChangeSeed
from .models import (
    Aspect,
    Edge,
    Graph,
    ImpactRecord,
    Node,
    NodeKind,
    Obligation,
    PrunedBoundary,
    Relation,
)

R, O, A = Relation, Obligation, Aspect
RULESETS = ("v1", "v1.1")
DEFAULT_RULES = "v1"


def check_rules(rules: str) -> str:
    if rules not in RULESETS:
        raise ValueError(f"unknown influence rule set {rules!r} (one of {', '.join(RULESETS)})")
    return rules

# relation -> (obligation for src endpoint, obligation for dst endpoint)
STRUCTURE_RULES: dict[str, tuple[str, str]] = {
    R.REFINES: (O.REVIEW_SPEC, O.REVIEW_SPEC),
    R.DEPENDS_ON: (O.CHECK_CONTRACT, O.REVIEW_SPEC),
    R.CONSTRAINED_BY: (O.CHECK_CONTRACT, O.REVIEW_SPEC),
    R.IMPLEMENTS: (O.REVIEW_IMPLEMENTATION, O.REVIEW_IMPLEMENTATION),
    R.VERIFIES: (O.RERUN_TEST, O.REVIEW_SPEC),
    R.REALIZED_BY: (O.REVIEW_IMPLEMENTATION, O.REVIEW_IMPLEMENTATION),
    R.BLOCKS: (O.REVIEW_SPEC, O.REVIEW_SPEC),
}


class UnionGraph:
    """Union of old and proposed edges so that removals stay visible (SPEC §2.4)."""

    def __init__(self, before: Graph, after: Graph):
        self.before, self.after = before, after
        self.edges: dict[str, tuple[Edge, str]] = {}  # id -> (edge, "kept"|"added"|"removed")
        before_live = {e.id: e for e in before.live_edges()}
        after_live = {e.id: e for e in after.live_edges()}
        for eid in sorted(set(before_live) | set(after_live)):
            if eid in after_live:
                self.edges[eid] = (after_live[eid], "kept" if eid in before_live else "added")
            else:
                self.edges[eid] = (before_live[eid], "removed")
        self._incident: dict[str, list[str]] = {}
        for eid, (e, _) in self.edges.items():
            self._incident.setdefault(e.src, []).append(eid)
            self._incident.setdefault(e.dst, []).append(eid)

    def node(self, nid: str) -> Node | None:
        return self.after.nodes.get(nid) or self.before.nodes.get(nid)

    def kind(self, nid: str) -> str | None:
        n = self.node(nid)
        return n.kind if n else None

    def live_after(self, nid: str) -> bool:
        return self.after.node(nid) is not None

    def incident(self, nid: str) -> list[tuple[Edge, str]]:
        return [self.edges[eid] for eid in sorted(self._incident.get(nid, []))]


@dataclass
class Event:
    node: str
    aspect: str
    obligation: str | None  # None for origin (a real change)
    origin: str
    op_index: int | None
    path: tuple[str, ...] = ()
    terminal: bool = False
    hypothesized: bool = False
    provisional: bool = False
    reason: str = ""

    @property
    def is_origin(self) -> bool:
        return self.obligation is None


@dataclass
class ImpactReport:
    records: list[ImpactRecord] = field(default_factory=list)
    pruned: list[PrunedBoundary] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligations": [r.to_dict() for r in self.records],
            "pruned": [p.to_dict() for p in self.pruned],
            "notes": list(self.notes),
        }

    def targets(self, obligation: str | None = None) -> set[str]:
        return {r.target_id for r in self.records if obligation is None or r.obligation == obligation}

    def by_target(self) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for r in self.records:
            out.setdefault(r.target_id, set()).add(r.obligation)
        return out


EvidenceLookup = Callable[[Iterable[str]], list[dict[str, Any]]]


class InfluenceEngine:
    def __init__(
        self,
        before: Graph,
        after: Graph,
        *,
        changed_node_ids: set[str] | None = None,
        evidence_refs: list[str] | None = None,
        evidence_lookup: EvidenceLookup | None = None,
        unverified_claims: list[dict[str, Any]] | None = None,
        rules: str = DEFAULT_RULES,
    ):
        self.rules = check_rules(rules)
        self.g = UnionGraph(before, after)
        self.changed = set(changed_node_ids or ())
        self.evidence = (evidence_lookup(evidence_refs) if evidence_lookup and evidence_refs else [])
        self.claims = unverified_claims or []
        self.report = ImpactReport()
        self._records: dict[tuple[str, str], ImpactRecord] = {}
        self._pruned_keys: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------
    def run(self, seeds: list[ChangeSeed]) -> ImpactReport:
        queue: deque[Event] = deque()
        for s in seeds:
            if s.edge_id is not None:
                queue.extend(self._structure_events(s))
            else:
                for aspect in sorted(s.aspects):
                    if aspect in (A.STRUCTURE, A.EVIDENCE):
                        continue  # node add/remove alone has no edge-borne influence
                    queue.append(Event(s.node_id, aspect, None, s.node_id, s.op_index))

        visited: set[tuple[str, str, str | None, bool]] = set()
        while queue:
            ev = queue.popleft()
            key = (ev.node, ev.aspect, ev.obligation, ev.terminal)
            if key in visited:
                continue
            visited.add(key)
            if not ev.is_origin:
                self._record(ev)
            if ev.terminal:
                if self.rules == "v1.1":
                    queue.extend(self._consumer_requirement_reach(ev))
                continue
            for nxt in self._propagate(ev):
                queue.append(nxt)

        self.report.records = sorted(
            self._records.values(), key=lambda r: (r.target_id, r.obligation, r.origin_id)
        )
        if self.claims and not self.report.pruned:
            for c in self.claims:
                self.report.notes.append(
                    f"op[{c['op_index']}] claims contract preservation for {c['target']} without "
                    "recorded validation evidence; claim ignored for pruning"
                )
        self.report.notes.append(
            "Influence is complete only relative to registered edges and rules; "
            "absence of an edge is not evidence of independence."
        )
        if self.rules != DEFAULT_RULES:
            self.report.notes.append(f"influence rule set {self.rules}")
        return self.report

    def _consumer_requirement_reach(self, ev: Event) -> list[Event]:
        """v1.1: a consumer flagged by a changed provider -> the requirements it implements (one hop, terminal)."""
        if ev.obligation != O.CHECK_CONTRACT or self.g.kind(ev.node) != NodeKind.COMPONENT or not ev.path:
            return []
        arrived = self.g.edges.get(ev.path[-1])
        if arrived is None or arrived[0].relation != R.DEPENDS_ON or arrived[0].src != ev.node:
            return []  # only a consumer reached across depends_on, not e.g. a component checking its own requirement
        out: list[Event] = []
        for edge, state in self.g.incident(ev.node):
            if edge.relation == R.IMPLEMENTS and edge.src == ev.node:
                out += self._emit(ev, edge, state, edge.dst, A.CONTRACT, O.CHECK_CONTRACT, True,
                                  f"consumer {ev.node} must re-check its reliance on a changed provider; the "
                                  "requirement it implements may no longer hold (rules v1.1)")
        return out

    # ------------------------------------------------------------------
    def _record(self, ev: Event) -> None:
        if not self.g.live_after(ev.node):
            return  # tombstoned target: visible in diff, no obligation
        key = (ev.node, ev.obligation)
        if key in self._records:
            return  # BFS => first path is a shortest explanation
        confidence = "low" if ev.hypothesized else ("provisional" if ev.provisional else "high")
        self._records[key] = ImpactRecord(
            target_id=ev.node,
            obligation=ev.obligation,
            origin_id=ev.origin,
            path=list(ev.path),
            reason=ev.reason,
            confidence=confidence,
            change_ref=f"op[{ev.op_index}]" if ev.op_index is not None else None,
            provisional=ev.provisional,
        )

    def _structure_events(self, s: ChangeSeed) -> list[Event]:
        entry = self.g.edges.get(s.edge_id)
        if entry is None:
            return []
        e, _state = entry
        src_ob, dst_ob = STRUCTURE_RULES[e.relation]
        ob = src_ob if s.node_id == e.src else dst_ob
        role = "source" if s.node_id == e.src else "target"
        verb = s.edge_change or "changed"
        reason = (
            f"{e.relation} edge {e.id} ({e.src} -> {e.dst}) {verb}; {role} endpoint must be re-evaluated"
        )
        events = [Event(s.node_id, A.STRUCTURE, ob, e.id, s.op_index, (e.id,), True,
                        e.hypothesized, False, reason)]
        if e.hypothesized:
            events.append(Event(s.node_id, A.STRUCTURE, O.INVESTIGATE_UNKNOWN, e.id, s.op_index, (e.id,),
                                True, True, False, f"hypothesized edge {e.id} {verb}; dependency unconfirmed"))
        return events

    def _emit(self, ev: Event, edge: Edge, state: str, target: str, aspect: str, ob: str,
              terminal: bool, reason: str, provisional: bool = False) -> list[Event]:
        hyp = ev.hypothesized or edge.hypothesized
        suffix = " (via edge removed in this change)" if state == "removed" else (
            " (via edge added in this change)" if state == "added" else "")
        path = ev.path + (edge.id,)
        out = [Event(target, aspect, ob, ev.origin, ev.op_index, path, terminal, hyp,
                     provisional or ev.provisional, reason + suffix)]
        if edge.hypothesized:
            out.append(Event(target, aspect, O.INVESTIGATE_UNKNOWN, ev.origin, ev.op_index, path, True, True,
                             False, f"path crosses hypothesized edge {edge.id}; confirm or refute the dependency"))
        return out

    def _propagate(self, ev: Event) -> list[Event]:
        out: list[Event] = []
        n, kind = ev.node, self.g.kind(ev.node)
        last = ev.path[-1] if ev.path else None
        for edge, state in self.g.incident(n):
            if edge.id == last:
                continue  # never bounce straight back along the arriving edge
            outgoing = edge.src == n
            other = edge.dst if outgoing else edge.src
            rel = edge.relation

            if ev.aspect == A.SEMANTIC:
                if rel == R.REFINES and outgoing:
                    out += self._emit(ev, edge, state, other, A.SEMANTIC, O.REVIEW_SPEC, False,
                                      f"refinement parent {n} changed semantically; review child for consistency")
                elif not ev.is_origin:
                    continue  # downward review obligations only continue downward
                elif rel == R.REFINES and not outgoing:
                    out += self._emit(ev, edge, state, other, A.SEMANTIC, O.REVIEW_SPEC, True,
                                      f"child {n} changed; parent-consistency check (terminal, no sibling fan-out)")
                elif rel == R.BLOCKS and outgoing:
                    out += self._emit(ev, edge, state, other, A.SEMANTIC, O.REVIEW_SPEC, True,
                                      f"question {n} changed; re-evaluate the target it blocks/qualifies")
                elif rel == R.IMPLEMENTS and not outgoing:
                    out += self._emit(ev, edge, state, other, A.IMPLEMENTATION, O.REVIEW_IMPLEMENTATION, False,
                                      f"requirement {n} changed semantically; implementing component must be reviewed")
                elif rel == R.VERIFIES and not outgoing:
                    out += self._emit(ev, edge, state, other, A.SEMANTIC, O.RERUN_TEST, True,
                                      f"{n} changed; verifying test must be re-run")
                elif rel == R.DEPENDS_ON and not outgoing:
                    out += self._emit(ev, edge, state, other, A.CONTRACT, O.CHECK_CONTRACT, True,
                                      f"provider {n} changed observable behaviour; consumer must check its reliance")
                elif rel == R.IMPLEMENTS and outgoing:
                    out += self._emit(ev, edge, state, other, A.CONTRACT, O.CHECK_CONTRACT, True,
                                      f"component {n} changed; implemented requirement must still be satisfied")

            elif ev.aspect == A.CONTRACT:
                if kind == NodeKind.CONTRACT and ev.is_origin:
                    if rel == R.CONSTRAINED_BY and not outgoing:
                        out += self._emit(ev, edge, state, other, A.CONTRACT, O.CHECK_CONTRACT, False,
                                          f"contract {n} changed; constrained subject must re-check compliance")
                    elif rel == R.VERIFIES and not outgoing:
                        out += self._emit(ev, edge, state, other, A.CONTRACT, O.RERUN_TEST, True,
                                          f"contract {n} changed; test verifying it must be re-run")
                    elif rel == R.DEPENDS_ON and not outgoing:
                        out += self._emit(ev, edge, state, other, A.CONTRACT, O.CHECK_CONTRACT, True,
                                          f"contract {n} changed; dependent consumer must re-check")
                elif not ev.is_origin:  # a subject whose contract changed
                    if rel == R.DEPENDS_ON and not outgoing:
                        out += self._emit(ev, edge, state, other, A.CONTRACT, O.CHECK_CONTRACT, True,
                                          f"provider {n} is subject to a changed contract; consumer must re-check")
                    elif kind == NodeKind.REQUIREMENT and rel == R.VERIFIES and not outgoing:
                        out += self._emit(ev, edge, state, other, A.CONTRACT, O.RERUN_TEST, True,
                                          f"requirement {n} relies on a changed contract; verification needs re-evaluation")
                    elif kind == NodeKind.REQUIREMENT and rel == R.IMPLEMENTS and not outgoing:
                        out += self._emit(ev, edge, state, other, A.IMPLEMENTATION, O.REVIEW_IMPLEMENTATION, False,
                                          f"requirement {n} relies on a changed contract; review its implementation")
                    elif kind == NodeKind.COMPONENT and rel == R.IMPLEMENTS and outgoing:
                        out += self._emit(ev, edge, state, other, A.CONTRACT, O.CHECK_CONTRACT, True,
                                          f"component {n} is subject to a changed contract; implemented requirement must re-check")

            elif ev.aspect == A.IMPLEMENTATION:
                if kind == NodeKind.ARTIFACT and rel == R.REALIZED_BY and not outgoing:
                    out += self._emit(ev, edge, state, other, A.IMPLEMENTATION, O.REVIEW_IMPLEMENTATION, False,
                                      f"artifact {n} changed; component realized by it must be reviewed")
                elif kind == NodeKind.COMPONENT and rel == R.IMPLEMENTS and outgoing:
                    out += self._emit(ev, edge, state, other, A.CONTRACT, O.CHECK_CONTRACT, True,
                                      f"implementation of {n} changed; requirement must still hold "
                                      "(terminal until a violation is established)")
                elif kind == NodeKind.COMPONENT and rel == R.DEPENDS_ON and not outgoing:
                    out += self._cross_provider_boundary(ev, edge, state, other)
        return out

    # ------------------------------------------------------------------
    def _cross_provider_boundary(self, ev: Event, edge: Edge, state: str, consumer: str) -> list[Event]:
        provider = ev.node
        if not edge.hypothesized and state != "removed":
            proof = self._preservation_evidence(provider)
            if proof is not None:
                ev_id, contracts = proof
                key = (consumer, provider)
                if key not in self._pruned_keys:
                    self._pruned_keys.add(key)
                    self.report.pruned.append(PrunedBoundary(
                        consumer_id=consumer, provider_id=provider, edge_id=edge.id, evidence_id=ev_id,
                        contracts=contracts,
                        reason=(f"{ev_id} records contracts {sorted(contracts)} of {provider} as preserved at the "
                                "current revisions; implementation-only change does not cross this boundary"),
                    ))
                return []
        claimed = any(c.get("target") == provider for c in self.claims)
        why = "an unverified preservation claim cannot prune" if claimed else "no validated contract-preservation evidence"
        return self._emit(ev, edge, state, consumer, A.CONTRACT, O.CHECK_CONTRACT, True,
                          f"implementation of provider {provider} changed and {why}; consumer provisionally flagged",
                          provisional=True)

    def _preservation_evidence(self, provider: str) -> tuple[str, dict[str, int]] | None:
        after = self.g.after
        contracts = {e.dst for e in after.out_edges(provider, R.CONSTRAINED_BY)}
        if not contracts:
            return None  # nothing explicit to preserve => cannot prune
        current = after.revs(contracts)
        for ev in self.evidence:
            if ev.get("kind") != "contract_preservation" or ev.get("node_id") != provider:
                continue
            if ev.get("result") != "passed" or ev.get("invalidated_by"):
                continue
            spec_revs = ev.get("spec_revs") or {}
            if any(spec_revs.get(c) != current[c] for c in contracts):
                continue
            if contracts & self.changed:
                continue
            return ev["id"], current
        return None


def analyze_delta(
    applied: AppliedDelta,
    *,
    evidence_refs: list[str] | None = None,
    evidence_lookup: EvidenceLookup | None = None,
    rules: str = DEFAULT_RULES,
) -> ImpactReport:
    engine = InfluenceEngine(
        applied.before,
        applied.after,
        changed_node_ids=applied.changed_node_ids,
        evidence_refs=evidence_refs,
        evidence_lookup=evidence_lookup,
        unverified_claims=applied.unverified_claims,
        rules=rules,
    )
    return engine.run(applied.seeds)


def analyze_node(
    graph: Graph,
    node_id: str,
    aspect: str,
    *,
    evidence_refs: list[str] | None = None,
    evidence_lookup: EvidenceLookup | None = None,
    rules: str = DEFAULT_RULES,
) -> ImpactReport:
    """Hypothetical impact of changing ``node_id`` under ``aspect`` (no mutation)."""
    if graph.node(node_id) is None:
        raise KeyError(node_id)
    if aspect not in set(Aspect):
        raise ValueError(f"unknown aspect {aspect!r}")
    seeds: list[ChangeSeed] = []
    if aspect == A.STRUCTURE:
        for e in graph.live_edges():
            if node_id in (e.src, e.dst):
                seeds.append(ChangeSeed(0, e.src, {aspect}, e.id, "changed"))
                seeds.append(ChangeSeed(0, e.dst, {aspect}, e.id, "changed"))
    else:
        aspects = {aspect}
        if graph.nodes[node_id].kind == NodeKind.CONTRACT and aspect == A.SEMANTIC:
            aspects.add(A.CONTRACT.value)
        seeds.append(ChangeSeed(0, node_id, aspects))
    engine = InfluenceEngine(graph, graph, changed_node_ids={node_id}, evidence_refs=evidence_refs,
                             evidence_lookup=evidence_lookup, rules=rules)
    report = engine.run(seeds)
    for r in report.records:
        r.change_ref = f"hypothetical:{node_id}:{aspect}"
    return report
