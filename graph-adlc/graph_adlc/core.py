"""Programmatic core (SPEC §5). No HTTP server; the CLI is a thin wrapper."""

from __future__ import annotations

import json
from typing import Any

from . import lifecycle
from .context import context_packet
from .delta import apply_delta
from .influence import DEFAULT_RULES, ImpactReport, analyze_delta, analyze_node, check_rules
from .models import Edge, Graph, Node, ReadinessResult, Relation, canonical_json
from .readiness import frontier, readiness, readiness_diff
from .scheduler import suggest_actions
from .store import Store
from .validation import RevisionConflict, ValidationError, validate_graph


class GraphADLC:
    def __init__(self, db_path: str, store: Store | None = None, *, influence_rules: str = DEFAULT_RULES):
        self.store = store or Store(db_path)
        self.influence_rules = check_rules(influence_rules)  # "v1" (default) or "v1.1"; see influence.py

    def close(self) -> None:
        self.store.close()

    # -- graph kernel ---------------------------------------------------------
    def seed_graph(self, snapshot: dict[str, Any]) -> int:
        if self.store.initialized():
            raise ValidationError(["database already initialized"])
        now = self.store.clock()
        g = Graph(revision=int(snapshot.get("graph_revision", 1)))
        for n in snapshot.get("nodes", []):
            if n["id"] in g.nodes:
                raise ValidationError([f"duplicate node id {n['id']}"])
            g.nodes[n["id"]] = Node(id=n["id"], kind=n["kind"], title=n["title"], body=n.get("body", ""),
                                    data=n.get("data", {}), rev=int(n.get("rev", 1)),
                                    deleted=bool(n.get("deleted", False)), created_at=now, updated_at=now)
        for e in snapshot.get("edges", []):
            if e["id"] in g.edges:
                raise ValidationError([f"duplicate edge id {e['id']}"])
            g.edges[e["id"]] = Edge(id=e["id"], src=e["src"], dst=e["dst"], relation=e["relation"],
                                    status=e.get("status", "confirmed"), rev=int(e.get("rev", 1)),
                                    deleted=bool(e.get("deleted", False)), evidence_ref=e.get("evidence_ref"))
        issues = validate_graph(g)
        if issues:
            raise ValidationError(issues)
        with self.store.transaction():
            for n in g.nodes.values():
                self.store.upsert_node(n)
            for e in g.edges.values():
                self.store.upsert_edge(e)
            self.store.set_meta("graph_revision", g.revision)
            self.store.log_event("seeded", None, {"graph_revision": g.revision, "nodes": len(g.nodes),
                                                  "edges": len(g.edges)})
        return g.revision

    def graph(self) -> Graph:
        return self.store.load_graph()

    def get_node(self, node_id: str) -> Node:
        n = self.graph().nodes.get(node_id)
        if n is None:
            raise KeyError(node_id)
        return n

    def get_subgraph(self, root_id: str, depth: int | None = None) -> dict[str, Any]:
        g = self.graph()
        if g.node(root_id) is None:
            raise KeyError(root_id)
        ids, frontier_ids, level = {root_id}, [root_id], 0
        while frontier_ids and (depth is None or level < depth):
            nxt = [c for nid in frontier_ids for c in g.children_of(nid) if c not in ids]
            ids.update(nxt)
            frontier_ids, level = nxt, level + 1
        return {"graph_revision": g.revision,
                "nodes": [g.nodes[i].to_dict(include_timestamps=False) for i in sorted(ids)],
                "edges": [e.to_dict() for e in g.live_edges() if e.src in ids and e.dst in ids]}

    def export(self) -> dict[str, Any]:
        return self.graph().to_snapshot()

    def export_json(self) -> str:
        return json.dumps(self.export(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"

    # -- readiness / context / impact ------------------------------------------
    def readiness(self, requirement_id: str) -> ReadinessResult:
        return readiness(self.graph(), requirement_id)

    def frontier(self) -> list[ReadinessResult]:
        return frontier(self.graph())

    def context(self, requirement_id: str) -> dict[str, Any]:
        return context_packet(self.graph(), requirement_id)

    def impact(self, node_id: str, aspect: str, evidence_refs: list[str] | None = None) -> ImpactReport:
        return analyze_node(self.graph(), node_id, aspect, evidence_refs=evidence_refs,
                            evidence_lookup=self.store.list_evidence, rules=self.influence_rules)

    # -- transactions -----------------------------------------------------------
    def _analyze(self, delta: dict[str, Any]) -> tuple[Any, ImpactReport | None]:
        g = self.graph()
        applied = apply_delta(g, delta, self.store.clock())
        if not applied.ok:
            return applied, None
        report = analyze_delta(applied, evidence_refs=delta.get("evidence_refs"),
                               evidence_lookup=self.store.list_evidence, rules=self.influence_rules)
        return applied, report

    def propose(self, delta: dict[str, Any]) -> dict[str, Any]:
        """Read-only preview: nothing in the database changes."""
        applied, impact = self._analyze(delta)
        out: dict[str, Any] = {
            "current_graph_revision": applied.before.revision,
            "base_graph_revision": delta.get("base_graph_revision"),
            "proposed_graph_revision": applied.before.revision + 1,
            "actor": delta.get("actor"), "rationale": delta.get("rationale"),
            "committable": applied.ok, "conflicts": applied.conflicts, "errors": applied.errors,
            "aspect_overrides": applied.aspect_overrides, "diff": applied.diff,
        }
        if impact is None:
            return out
        stale = lifecycle.mark_stale_after_change(self.store, applied.after, "(proposed)", impact.records,
                                                  dry_run=True)
        changes = readiness_diff(applied.before, applied.after)
        out.update({
            "impact": impact.to_dict(),
            "readiness_changes": changes,
            "newly_ready": [c["requirement_id"] for c in changes if c["to"] in ("ready", "conditional")],
            "frontier_after": [r.to_dict() for r in frontier(applied.after)],
            "would_invalidate": stale,
            "required_actions": _group_actions(impact),
            "approvals_required": self._approvals_required(applied),
        })
        return out

    def _approvals_required(self, applied: Any) -> list[str]:
        policy = {a for a in (self.store.get_meta("approval_required_aspects", "") or "").split(",") if a}
        hit = sorted({a for s in applied.seeds for a in s.aspects} & policy)
        return hit

    def commit(self, delta: dict[str, Any], approvals: list[str] | None = None) -> dict[str, Any]:
        """Atomic, revision-guarded commit. Raises on conflict/invalid delta; no partial writes."""
        with self.store.transaction():
            applied, impact = self._analyze(delta)
            if applied.conflicts:
                raise RevisionConflict(applied.conflicts)
            if applied.errors:
                raise ValidationError(applied.errors)
            needed = self._approvals_required(applied)
            if needed and not approvals:
                raise ValidationError([f"approval required for aspects {needed}"])
            new_rev = applied.before.revision + 1
            change_id = f"CHG-{new_rev}"
            for r in impact.records:
                r.change_ref = f"{change_id}/{r.change_ref}"
            for nid, n in applied.after.nodes.items():
                old = applied.before.nodes.get(nid)
                if old is None or old.to_dict() != n.to_dict():
                    self.store.upsert_node(n)
            for eid, e in applied.after.edges.items():
                old = applied.before.edges.get(eid)
                if old is None or old.to_dict() != e.to_dict():
                    self.store.upsert_edge(e)
            applied.after.revision = new_rev
            self.store.set_meta("graph_revision", new_rev)
            stale = lifecycle.mark_stale_after_change(self.store, applied.after, change_id, impact.records)
            changes = readiness_diff(applied.before, applied.after)
            impact_doc = {**impact.to_dict(), "readiness_changes": changes, "invalidated": stale,
                          "aspect_overrides": applied.aspect_overrides, "approvals": approvals or []}
            self.store.insert_change({
                "id": change_id, "graph_rev": new_rev, "base_graph_rev": applied.before.revision,
                "actor": delta.get("actor", "unknown"), "rationale": delta.get("rationale", ""),
                "operations": delta["operations"], "impact": impact_doc, "diff": applied.diff,
                "created_at": self.store.clock(),
            })
            self.store.log_event("committed", change_id, {"graph_revision": new_rev,
                                                          "obligations": len(impact.records),
                                                          "invalidated": stale})
        return {"change_id": change_id, "graph_revision": new_rev, "impact": impact.to_dict(),
                "readiness_changes": changes, "invalidated": stale, "diff": applied.diff,
                "required_actions": _group_actions(impact)}

    # -- lifecycle ------------------------------------------------------------------
    def create_work_unit(self, requirement_id: str, kind: str = "implement") -> dict[str, Any]:
        return lifecycle.public_view(lifecycle.create_work_unit(self.store, self.graph(), requirement_id, kind))

    def start_work_unit(self, work_id: str) -> dict[str, Any]:
        return lifecycle.public_view(lifecycle.start_work_unit(self.store, work_id))

    def submit_work_unit(self, work_id: str, code_revision: str,
                         evidence: list[dict[str, Any]]) -> dict[str, Any]:
        w, ev = lifecycle.submit_work_unit(self.store, self.graph(), work_id, code_revision, evidence)
        return {**lifecycle.public_view(w), "evaluation": ev}

    def record_evidence(self, node_id: str, kind: str, result: str = "passed", *,
                        code_rev: str | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
        return lifecycle.record_evidence(self.store, self.graph(), node_id=node_id, kind=kind, result=result,
                                         code_rev=code_rev, details=details)

    def reconcile(self, work_id: str, rebase: bool = False) -> dict[str, Any]:
        return lifecycle.reconcile(self.store, self.graph(), work_id, rebase=rebase)

    def work_unit(self, work_id: str) -> dict[str, Any]:
        w = self.store.get_work_unit(work_id)
        return {**lifecycle.public_view(w),
                "evaluation": lifecycle.evaluate(w, self.graph(), self.store.list_evidence())}

    def work_units(self) -> list[dict[str, Any]]:
        return [lifecycle.public_view(w) for w in self.store.list_work_units()]

    def evidence(self) -> list[dict[str, Any]]:
        g = self.graph()
        return [{**e, "current": not lifecycle.evidence_staleness(e, g),
                 "stale_reasons": lifecycle.evidence_staleness(e, g)} for e in self.store.list_evidence()]

    def suggest_actions(self, limit: int | None = None) -> list[Any]:
        weights = json.loads(self.store.get_meta("scheduler_weights", "{}") or "{}")
        return suggest_actions(self.graph(), self.store.list_work_units(), self.store.list_evidence(),
                               weights, limit)

    def freshness(self, requirement_id: str) -> str:
        return lifecycle.freshness(self.graph(), self.store.list_work_units(), self.store.list_evidence(),
                                   requirement_id)

    def status(self) -> dict[str, Any]:
        g = self.graph()
        wus, evs = self.store.list_work_units(), self.store.list_evidence()
        leaves = []
        for r in frontier(g):
            leaves.append({"requirement_id": r.requirement_id, "rev": r.rev, "readiness": r.readiness,
                           "freshness": lifecycle.freshness(g, wus, evs, r.requirement_id),
                           "blockers": [b.code for b in r.blockers], "conditions": [c.code for c in r.conditions]})
        open_q = [n.id for n in g.live_nodes() if n.kind == "question" and n.data.get("state") == "open"]
        return {
            "graph_revision": g.revision,
            "nodes": len(g.live_nodes()), "edges": len(g.live_edges()),
            "tombstones": {"nodes": sum(n.deleted for n in g.nodes.values()),
                           "edges": sum(e.deleted for e in g.edges.values())},
            "leaf_requirements": leaves,
            "open_questions": open_q,
            "work_units": [{"id": w["id"], "targets": w["target_ids"], "status": w["status"],
                            "base_graph_revision": w["baseline"]["base_graph_revision"]} for w in wus],
            "changes": [{"id": c["id"], "graph_rev": c["graph_rev"], "actor": c["actor"]}
                        for c in self.store.list_changes()],
        }

    def log(self) -> list[dict[str, Any]]:
        return self.store.list_changes()


def _group_actions(impact: ImpactReport) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for r in impact.records:
        out.setdefault(r.obligation, [])
        if r.target_id not in out[r.obligation]:
            out[r.obligation].append(r.target_id)
    return {k: sorted(v) for k, v in sorted(out.items())}


__all__ = ["GraphADLC", "RevisionConflict", "ValidationError", "canonical_json", "Relation"]
