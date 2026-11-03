"""Work units, evidence freshness and reconciliation (SPEC §4, R6/R7)."""

from __future__ import annotations

from typing import Any

from .models import Freshness, Graph, ImpactRecord, NodeKind, Obligation, Readiness, Relation, WorkStatus
from .readiness import readiness
from .store import Store

R, W = Relation, WorkStatus

OPEN_STATUSES = {W.PROPOSED, W.ACTIVE, W.SUBMITTED, W.VERIFIED, W.BLOCKED, W.NEEDS_REVALIDATION}
SUBMITTABLE = {W.PROPOSED, W.ACTIVE, W.SUBMITTED}
INVALIDATING_OBLIGATIONS = {Obligation.CHECK_CONTRACT, Obligation.RERUN_TEST, Obligation.INVESTIGATE_UNKNOWN}

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    W.PROPOSED: {W.ACTIVE, W.SUBMITTED, W.BLOCKED, W.NEEDS_REVALIDATION, W.ABANDONED},
    W.ACTIVE: {W.SUBMITTED, W.BLOCKED, W.NEEDS_REVALIDATION, W.ABANDONED},
    W.SUBMITTED: {W.SUBMITTED, W.VERIFIED, W.BLOCKED, W.NEEDS_REVALIDATION, W.ABANDONED},
    W.VERIFIED: {W.NEEDS_REVALIDATION, W.ABANDONED},
    W.BLOCKED: {W.ACTIVE, W.NEEDS_REVALIDATION, W.ABANDONED},
    W.NEEDS_REVALIDATION: {W.ACTIVE, W.BLOCKED, W.ABANDONED},
    W.ABANDONED: set(),
}


class LifecycleError(Exception):
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.details = details or {}
        super().__init__(message)


# ---------------------------------------------------------------------------
# Baselines and required checks
# ---------------------------------------------------------------------------

def baseline_node_ids(g: Graph, targets: list[str]) -> set[str]:
    """The spec/contract revisions a piece of work is produced against."""
    ids: set[str] = set()
    for t in targets:
        ids.add(t)
        for e in g.out_edges(t, R.CONSTRAINED_BY):
            ids.add(e.dst)
        for e in g.out_edges(t, R.DEPENDS_ON):
            ids.add(e.dst)
            ids.update(c.dst for c in g.out_edges(e.dst, R.CONSTRAINED_BY))
        for e in g.in_edges(t, R.VERIFIES):
            ids.add(e.src)
    return {i for i in ids if g.node(i) is not None}


def test_checks(g: Graph, targets: list[str]) -> list[dict[str, str]]:
    tests = sorted({e.src for t in targets for e in g.in_edges(t, R.VERIFIES) if g.node(e.src)})
    return [{"kind": "test_result", "node_id": t, "reason": "test verifies target"} for t in tests]


def transition(w: dict[str, Any], new_status: str, now: str, note: str, ref: str | None = None) -> None:
    old = w["status"]
    if new_status not in ALLOWED_TRANSITIONS[old]:
        raise LifecycleError(f"{w['id']}: transition {old} -> {new_status} is not allowed")
    w["status"] = new_status
    w["updated_at"] = now
    w["history"].append({"at": now, "from": old, "to": new_status, "note": note, **({"ref": ref} if ref else {})})


def public_view(w: dict[str, Any]) -> dict[str, Any]:
    b = w["baseline"]
    return {
        "id": w["id"], "kind": w["kind"], "target_ids": w["target_ids"], "status": w["status"],
        "base_graph_revision": b["base_graph_revision"], "node_revisions": b["node_revisions"],
        "assumptions": b["assumptions"], "acceptance_criteria": b["acceptance_criteria"],
        "readiness_at_baseline": b["readiness"], "required_checks": b["required_checks"],
        "context_node_ids": w["context_node_ids"], "code_rev": w.get("code_rev"),
        "history": w["history"],
    }


# ---------------------------------------------------------------------------
# Evidence freshness
# ---------------------------------------------------------------------------

def evidence_staleness(ev: dict[str, Any], g: Graph) -> list[str]:
    """Reasons evidence is not current (empty list = current)."""
    reasons = []
    if ev.get("invalidated_by"):
        reasons.append(f"invalidated by {ev['invalidated_by']}")
    for nid, rev in sorted(ev["spec_revs"].items()):
        node = g.nodes.get(nid)
        if node is None or node.deleted:
            reasons.append(f"{nid} no longer exists")
        elif node.rev != rev:
            reasons.append(f"{nid} recorded at r{rev}, now r{node.rev}")
    return reasons


def evaluate(w: dict[str, Any], g: Graph, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    b = w["baseline"]
    mismatches = []
    for nid, rev in sorted(b["node_revisions"].items()):
        node = g.nodes.get(nid)
        cur = None if node is None or node.deleted else node.rev
        if cur != rev:
            mismatches.append({"node_id": nid, "baseline_rev": rev, "current_rev": cur})
    mine = [e for e in evidence if e.get("work_unit_id") == w["id"]]
    stale = [{"evidence_id": e["id"], "node_id": e["node_id"], "kind": e["kind"], "reasons": r}
             for e in mine if (r := evidence_staleness(e, g))]
    fresh = [e for e in mine if e["seq"] > b["evidence_after_seq"] and not evidence_staleness(e, g)]
    missing, satisfied = [], []
    for chk in b["required_checks"]:
        hit = [e for e in fresh if e["kind"] == chk["kind"] and e["node_id"] == chk["node_id"]]
        passed = [e for e in hit if e["result"] == "passed"]
        if passed:
            satisfied.append({**chk, "evidence_id": passed[-1]["id"]})
        else:
            missing.append({**chk, "status": "failed" if hit else "missing"})
    return {"baseline_mismatches": mismatches, "stale_evidence": stale, "missing_checks": missing,
            "satisfied_checks": satisfied, "verifiable": not mismatches and not missing}


def freshness(g: Graph, work_units: list[dict[str, Any]], evidence: list[dict[str, Any]], req_id: str) -> str:
    mine = [w for w in work_units if req_id in w["target_ids"] and w["status"] != W.ABANDONED]
    if not mine:
        return Freshness.UNIMPLEMENTED.value
    w = mine[-1]
    if w["status"] == W.NEEDS_REVALIDATION:
        return Freshness.NEEDS_REVALIDATION.value
    if w["status"] == W.VERIFIED:
        # Derived check as well: never trust a stored "verified" if its evidence has gone stale.
        return Freshness.CURRENT.value if evaluate(w, g, evidence)["verifiable"] else Freshness.NEEDS_REVALIDATION.value
    return Freshness.IN_PROGRESS.value


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------

def create_work_unit(store: Store, g: Graph, requirement_id: str, kind: str = "implement") -> dict[str, Any]:
    r = readiness(g, requirement_id)
    if r.readiness == Readiness.BLOCKED:
        raise LifecycleError(f"{requirement_id} is blocked", {"readiness": r.to_dict()})
    for w in store.list_work_units():
        if requirement_id in w["target_ids"] and w["status"] not in (W.ABANDONED, W.VERIFIED):
            raise LifecycleError(f"{requirement_id} already has open work unit {w['id']} ({w['status']})")
    from .context import context_packet  # local import to avoid cycle at module import

    now = store.clock()
    seq = store.next_seq("work")
    node = g.nodes[requirement_id]
    ctx = context_packet(g, requirement_id)
    assumptions = [
        {**a, "origin_node": requirement_id, "origin_rev": node.rev, "origin_graph_revision": g.revision}
        for a in r.assumptions
    ]
    w = {
        "id": f"WORK-{seq}", "seq": seq, "kind": kind, "target_ids": [requirement_id],
        "baseline": {
            "base_graph_revision": g.revision,
            "node_revisions": g.revs(baseline_node_ids(g, [requirement_id])),
            "readiness": r.readiness,
            "conditions": [c.__dict__ for c in r.conditions],
            "assumptions": assumptions,
            "acceptance_criteria": list(node.data.get("acceptance_criteria", [])),
            "required_checks": test_checks(g, [requirement_id]),
            "evidence_after_seq": int(store.get_meta("seq:evidence", "0")),
        },
        "context_node_ids": sorted(ctx["node_revisions"]),
        "status": W.PROPOSED.value, "code_rev": None,
        "history": [{"at": now, "from": None, "to": W.PROPOSED.value,
                     "note": f"created from {requirement_id}@r{node.rev} ({r.readiness}) at graph rev {g.revision}"}],
        "created_at": now, "updated_at": now,
    }
    with store.transaction():
        store.save_work_unit(w)
        store.log_event("work_unit_created", w["id"], public_view(w))
    return w


def start_work_unit(store: Store, work_id: str) -> dict[str, Any]:
    with store.transaction():
        w = store.get_work_unit(work_id)
        transition(w, W.ACTIVE.value, store.clock(), "work started")
        store.save_work_unit(w)
        store.log_event("work_unit_started", work_id, {"status": w["status"]})
    return w


def record_evidence(store: Store, g: Graph, *, node_id: str, kind: str, result: str,
                    work_unit_id: str | None = None, code_rev: str | None = None,
                    spec_node_ids: set[str] | None = None, details: dict[str, Any] | None = None) -> dict[str, Any]:
    if g.node(node_id) is None:
        raise LifecycleError(f"evidence target {node_id} does not exist")
    if result not in ("passed", "failed"):
        raise LifecycleError("evidence result must be 'passed' or 'failed'")
    ids = set(spec_node_ids or ()) | {node_id}
    if kind == "contract_preservation":
        ids |= {e.dst for e in g.out_edges(node_id, R.CONSTRAINED_BY)}
    if g.nodes[node_id].kind == NodeKind.TEST:
        ids |= {e.dst for e in g.out_edges(node_id, R.VERIFIES)}
    seq = store.next_seq("evidence")
    ev = {"id": f"EV-{seq}", "seq": seq, "work_unit_id": work_unit_id, "node_id": node_id, "kind": kind,
          "result": result, "spec_revs": g.revs(ids), "code_rev": code_rev, "details": details or {},
          "invalidated_by": None, "created_at": store.clock()}
    with store.transaction():
        store.insert_evidence(ev)
        store.log_event("evidence_recorded", ev["id"], {k: ev[k] for k in ("node_id", "kind", "result", "spec_revs", "code_rev", "work_unit_id")})
    return ev


def submit_work_unit(store: Store, g: Graph, work_id: str, code_revision: str,
                     evidence: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    with store.transaction():
        w = store.get_work_unit(work_id)
        if w["status"] == W.NEEDS_REVALIDATION:
            raise LifecycleError(
                f"{work_id} needs revalidation against the current graph; run reconcile (and rebase) first",
                {"evaluation": evaluate(w, g, store.list_evidence())})
        if w["status"] not in SUBMITTABLE:
            raise LifecycleError(f"{work_id} cannot be submitted from status {w['status']}")
        pre = evaluate(w, g, store.list_evidence())
        if pre["baseline_mismatches"]:
            transition(w, W.NEEDS_REVALIDATION.value, store.clock(), "baseline no longer current at submission")
            store.save_work_unit(w)
            raise LifecycleError(f"{work_id} baseline is stale; reconcile first", {"evaluation": pre})
        spec_ids = set(w["baseline"]["node_revisions"])
        for item in evidence:
            record_evidence(store, g, node_id=item["node_id"], kind=item.get("kind", "test_result"),
                            result=item.get("result", "passed"), work_unit_id=work_id, code_rev=code_revision,
                            spec_node_ids=spec_ids, details=item.get("details"))
        w["code_rev"] = code_revision
        transition(w, W.SUBMITTED.value, store.clock(), f"submitted at code rev {code_revision}")
        ev = evaluate(w, g, store.list_evidence())
        if ev["verifiable"]:
            transition(w, W.VERIFIED.value, store.clock(),
                       "all required checks have current passing evidence",
                       ",".join(c["evidence_id"] for c in ev["satisfied_checks"]))
        store.save_work_unit(w)
        store.log_event("work_unit_submitted", work_id, {"status": w["status"], "code_rev": code_revision,
                                                         "missing_checks": ev["missing_checks"]})
    return w, ev


def reconcile(store: Store, g: Graph, work_id: str, *, rebase: bool = False) -> dict[str, Any]:
    with store.transaction():
        w = store.get_work_unit(work_id)
        evidence = store.list_evidence()
        ev = evaluate(w, g, evidence)
        base_rev = w["baseline"]["base_graph_revision"]
        relevant = set(w["context_node_ids"]) | set(w["baseline"]["node_revisions"])
        obligations = []
        for ch in store.list_changes():
            if ch["graph_rev"] <= base_rev:
                continue
            for o in ch["impact"]["obligations"]:
                if o["target_id"] in relevant:
                    obligations.append({**o, "change_id": ch["id"]})
        r = readiness(g, w["target_ids"][0])
        report: dict[str, Any] = {
            "work_unit_id": work_id, "status_before": w["status"], "graph_revision": g.revision,
            "baseline_graph_revision": base_rev, **ev, "obligations_since_baseline": obligations,
            "target_readiness": r.readiness, "rebased": False,
        }
        actions = []
        if ev["baseline_mismatches"]:
            actions.append("review spec changes and rebase the work unit onto current revisions")
        changed_contracts = sorted(
            m["node_id"] for m in ev["baseline_mismatches"]
            if (n := g.nodes.get(m["node_id"])) is not None and n.kind == NodeKind.CONTRACT
        )
        for c in changed_contracts:
            actions.append(f"record contract_check evidence for {c} at r{g.nodes[c].rev}")
        for chk in ev["missing_checks"]:
            actions.append(f"record fresh {chk['kind']} for {chk['node_id']}")

        if rebase:
            now = store.clock()
            if r.readiness == Readiness.BLOCKED:
                if w["status"] != W.BLOCKED:
                    transition(w, W.BLOCKED.value, now, "target is blocked at current revision")
                report["rebased"] = False
                actions.insert(0, "resolve blockers on target before rebasing")
            else:
                if w["status"] in (W.VERIFIED, W.SUBMITTED, W.PROPOSED):
                    transition(w, W.NEEDS_REVALIDATION.value, now, "explicit reconciliation requested")
                old = w["baseline"]
                checks = test_checks(g, w["target_ids"]) + [
                    {"kind": "contract_check", "node_id": c,
                     "reason": f"contract changed r{old['node_revisions'][c]} -> r{g.nodes[c].rev} since baseline"}
                    for c in changed_contracts
                ]
                for o in obligations:  # unresolved contract obligations on the target itself
                    if o["obligation"] == Obligation.CHECK_CONTRACT and o["target_id"] in w["target_ids"]:
                        for e in g.out_edges(o["target_id"], R.CONSTRAINED_BY):
                            if not any(c["kind"] == "contract_check" and c["node_id"] == e.dst for c in checks):
                                checks.append({"kind": "contract_check", "node_id": e.dst,
                                               "reason": f"check_contract obligation from {o['change_id']}"})
                tid = w["target_ids"][0]
                node = g.nodes[tid]
                w["baseline"] = {
                    **old,
                    "base_graph_revision": g.revision,
                    "node_revisions": g.revs(baseline_node_ids(g, w["target_ids"])),
                    "readiness": r.readiness,
                    "conditions": [c.__dict__ for c in r.conditions],
                    "assumptions": [{**a, "origin_node": tid, "origin_rev": node.rev,
                                     "origin_graph_revision": g.revision} for a in r.assumptions],
                    "acceptance_criteria": list(node.data.get("acceptance_criteria", [])),
                    "required_checks": checks,
                    "evidence_after_seq": int(store.get_meta("seq:evidence", "0")),
                    "previous_baselines": old.get("previous_baselines", []) + [
                        {"base_graph_revision": old["base_graph_revision"], "node_revisions": old["node_revisions"]}],
                }
                transition(w, W.ACTIVE.value, now,
                           f"rebased from graph rev {base_rev} to {g.revision}; fresh evidence required")
                report["rebased"] = True
                report["new_baseline"] = {"base_graph_revision": g.revision,
                                          "node_revisions": w["baseline"]["node_revisions"],
                                          "required_checks": checks}
                actions = [f"record fresh {c['kind']} for {c['node_id']}" for c in checks] + [
                    "submit the work unit with the updated code revision"]
            store.save_work_unit(w)
        report["status_after"] = w["status"]
        report["actions"] = actions
        report["can_verify"] = ev["verifiable"] and not report["rebased"] and w["status"] in SUBMITTABLE
        store.log_event("reconciled", work_id, {k: report[k] for k in
                                                ("status_before", "status_after", "rebased", "baseline_mismatches",
                                                 "missing_checks", "actions")})
    return report


def mark_stale_after_change(store: Store, g_after: Graph, change_id: str, records: list[ImpactRecord],
                            *, dry_run: bool = False) -> dict[str, list[dict[str, Any]]]:
    """Make work/evidence produced against superseded revisions explicitly stale (R6)."""
    hot = {r.target_id for r in records if r.obligation in INVALIDATING_OBLIGATIONS}
    stale_work, stale_ev = [], []
    for ev in store.list_evidence():
        if ev["invalidated_by"]:
            continue
        reasons = evidence_staleness(ev, g_after)
        if ev["node_id"] in hot:
            reasons.append(f"{ev['node_id']} received a verification obligation")
        if reasons:
            stale_ev.append({"evidence_id": ev["id"], "node_id": ev["node_id"], "reasons": reasons})
            if not dry_run:
                store.invalidate_evidence(ev["id"], change_id)
    now = store.clock()
    for w in store.list_work_units():
        if w["status"] in (W.ABANDONED, W.NEEDS_REVALIDATION):
            continue
        mism = evaluate(w, g_after, [])["baseline_mismatches"]
        hits = sorted(set(w["target_ids"]) & hot)
        if not mism and not hits:
            continue
        reasons = [f"{m['node_id']} r{m['baseline_rev']} -> r{m['current_rev']}" for m in mism]
        reasons += [f"{t} received a verification obligation" for t in hits]
        stale_work.append({"work_unit_id": w["id"], "status": w["status"], "reasons": reasons})
        if not dry_run:
            transition(w, W.NEEDS_REVALIDATION.value, now, "; ".join(reasons), change_id)
            store.save_work_unit(w)
    return {"work_units": stale_work, "evidence": stale_ev}
