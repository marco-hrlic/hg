"""Human review gate and commit through the existing GraphADLC transaction path.

Nothing is committed without an explicit per-operation decision. Decisions:
``accept`` / ``reject`` / ``defer``. ``invalid`` operations cannot be accepted;
``quarantined`` ones need ``acknowledge: true`` on the decision. Accepted
operations are translated into one ordinary engine delta and committed with
``GraphADLC.commit`` (strict base revision, per-entity ``expected_rev``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from ..core import GraphADLC
from ..validation import RevisionConflict, ValidationError
from .ledger import Ledger
from .proposal import Proposal
from .source import SourceBundle, SourceCatalog
from .validation import build_delta, evidence_key, validate_proposal

DECISIONS = ("accept", "reject", "defer")


class ReviewError(Exception):
    pass


class NeedsRegeneration(Exception):
    """The graph moved under the proposal in a way that cannot be rebased safely."""


@dataclass
class Decision:
    op_id: str
    decision: str
    reviewer: str
    rationale: str = ""
    acknowledge: bool = False


def render_review(proposal: Proposal, report: dict[str, Any]) -> str:
    """Itemized, human-readable diff with evidence and an influence preview."""
    L = [f"Proposal {proposal.proposal_id} against graph r{report['base_graph_revision']} "
         f"(current r{report['graph_revision']}{', STALE BASE' if report['stale_base'] else ''})",
         f"Intent: {proposal.intent}",
         f"Origin: {proposal.origin.get('client', {}).get('provider', 'unknown')}"
         f"{' (FAKE — not a model output)' if proposal.origin.get('is_fake') else ''}, "
         f"prompt {proposal.origin.get('prompt', {}).get('prompt_id', '?')}."
         f"{proposal.origin.get('prompt', {}).get('version', '?')}", ""]
    for o in proposal.operations:
        v = report["ops"][o.op_id]
        L.append(f"[{v['verdict'].upper():11}] {o.op_id} {o.kind} {_describe(o.kind, o.payload)}")
        L.append(f"    support: claimed={o.support} effective={v['effective_support']}"
                 f"{'' if o.confidence is None else f' (model confidence {o.confidence}, not used)'}")
        if o.reason:
            L.append(f"    reason: {o.reason}")
        for c in v["citations"]:
            lines = c.get("resolved_lines") or c.get("lines")
            L.append(f"    evidence [{c['status']}] {c['source_id']} L{lines}: \"{c['excerpt'][:160]}\""
                     + (f" — {c['detail']}" if c.get("detail") else ""))
        for f in v["findings"]:
            L.append(f"    {f['severity']}: {f['code']} — {f['message']}")
    if proposal.unresolved_questions:
        L.append("\nUnresolved questions:")
        L += [f"  - {q.get('question', q)} (about {q.get('about', [])})" for q in proposal.unresolved_questions]
    if proposal.possible_contradictions:
        L.append("\nPossible contradictions:")
        L += [f"  - {c.get('description', c)} (between {c.get('between', [])})" for c in proposal.possible_contradictions]
    ep = report["engine_preview"]
    L.append("\nInfluence preview if every admissible/quarantined op were accepted:")
    if not ep.get("ran"):
        L.append(f"  (not run: {ep.get('reason')})")
    elif not ep.get("committable"):
        L.append(f"  engine rejects the combined delta: {ep.get('errors')}")
    else:
        for ob, targets in sorted(ep["required_actions"].items()):
            L.append(f"  {ob}: {', '.join(targets)}")
        for rc in ep["readiness_changes"]:
            L.append(f"  readiness {rc['requirement_id']}: {rc['from']} -> {rc['to']}")
    L.append(f"\nNote: {report['disclaimer']}")
    return "\n".join(L)


def _describe(kind: str, p: dict[str, Any]) -> str:
    if kind == "add_node":
        n = p.get("node", {})
        return f"{n.get('id')} ({n.get('kind')}): {n.get('title')!r}"
    if kind == "add_edge":
        e = p.get("edge", {})
        return f"{e.get('src')} -{e.get('relation')}-> {e.get('dst')}"
    if kind == "update_node":
        return f"{p.get('id')}@r{p.get('expected_rev')} fields {sorted((p.get('patch') or {}).keys())}"
    if kind == "no_change":
        return f"{p.get('target')}"
    return f"{p.get('id')}@r{p.get('expected_rev')}"


def check_decisions(proposal: Proposal, report: dict[str, Any], decisions: list[Decision]) -> list[str]:
    errs = []
    by = {d.op_id: d for d in decisions}
    for d in decisions:
        if d.decision not in DECISIONS:
            errs.append(f"{d.op_id}: unknown decision {d.decision!r}")
        if d.op_id not in report["ops"]:
            errs.append(f"{d.op_id}: not an operation of {proposal.proposal_id}")
            continue
        v = report["ops"][d.op_id]
        if d.decision == "accept" and v["verdict"] == "invalid":
            errs.append(f"{d.op_id}: invalid operations cannot be accepted ({[f['code'] for f in v['findings'] if f['severity'] == 'error']})")
        if d.decision == "accept" and v["verdict"] == "quarantined" and not d.acknowledge:
            errs.append(f"{d.op_id}: quarantined; accepting requires acknowledge=true "
                        f"({[f['code'] for f in v['findings'] if f['severity'] == 'review']})")
        if not d.reviewer:
            errs.append(f"{d.op_id}: reviewer identity is required")
    for o in proposal.operations:
        if o.op_id not in by:
            errs.append(f"{o.op_id}: no decision recorded (every operation needs accept/reject/defer)")
    # An accepted edge whose new endpoint node is not accepted cannot be committed.
    accepted = {d.op_id for d in decisions if d.decision == "accept"}
    new_nodes = {o.payload["node"]["id"]: o.op_id for o in proposal.operations if o.kind == "add_node"}
    for o in proposal.operations:
        if o.kind == "add_edge" and o.op_id in accepted:
            for end in (o.payload["edge"]["src"], o.payload["edge"]["dst"]):
                if end in new_nodes and new_nodes[end] not in accepted and \
                        report["ops"][new_nodes[end]]["verdict"] != "duplicate":
                    errs.append(f"{o.op_id}: endpoint {end} is added by {new_nodes[end]}, which is not accepted")
    return errs


def apply_review(app: GraphADLC, ledger: Ledger, proposal: Proposal, report: dict[str, Any],
                 decisions: list[Decision], bundle: SourceBundle, catalog: SourceCatalog | None = None,
                 approvals: list[str] | None = None) -> dict[str, Any]:
    """Record decisions; commit accepted graph ops via GraphADLC.commit. Graph untouched on any failure."""
    errs = check_decisions(proposal, report, decisions)
    if errs:
        raise ReviewError("; ".join(errs))
    accepted = [d.op_id for d in decisions if d.decision == "accept"
                and report["ops"][d.op_id]["verdict"] != "duplicate"]
    graph_ops = [oid for oid in accepted if proposal.op(oid).kind != "no_change"]
    reviewers = sorted({d.reviewer for d in decisions if d.decision == "accept"})
    actor = f"llm-graph:{proposal.proposal_id} reviewed-by:{','.join(reviewers) or 'none'}"
    commit_result, rebased = None, False
    if graph_ops:
        g = app.graph()
        delta = build_delta(proposal, graph_ops, report, report["base_graph_revision"], actor, g)
        try:
            commit_result = app.commit(delta, approvals=approvals)
        except RevisionConflict:
            # Re-read and re-validate against the current graph; rebase only if every accepted op is still admissible.
            moved = copy.deepcopy(proposal)
            moved.base_graph_revision = app.graph().revision
            _, fresh = validate_proposal(app, moved, bundle, catalog, None)
            bad = [oid for oid in graph_ops if fresh["ops"].get(oid, {}).get("verdict") in ("invalid", None)
                   or (fresh["ops"][oid]["verdict"] == "quarantined" and report["ops"][oid]["verdict"] == "admissible")]
            if bad:
                ledger.save_proposal(proposal.to_dict(), report, "stale")
                raise NeedsRegeneration(f"graph moved to r{fresh['graph_revision']}; ops {bad} are no longer "
                                        "admissible — regenerate the proposal from the current graph")
            g = app.graph()
            delta = build_delta(moved, graph_ops, fresh, g.revision, actor + " (rebased)", g)
            commit_result = app.commit(delta, approvals=approvals)
            rebased = True
        except ValidationError:
            ledger.save_proposal(proposal.to_dict(), report, "invalid")
            raise
    change_id = commit_result["change_id"] if commit_result else None
    rows = []
    for d in decisions:
        v = report["ops"][d.op_id]
        rows.append({"op_id": d.op_id, "decision": d.decision, "reviewer": d.reviewer,
                     "rationale": d.rationale + (" [acknowledged findings]" if d.acknowledge else ""),
                     "fingerprint": proposal.op(d.op_id).fingerprint(), "evidence_key": evidence_key(v["citations"])})
    ledger.save_proposal(proposal.to_dict(), report, "committed" if change_id else "reviewed")
    ledger.record_decisions(proposal.proposal_id, rows, change_id)
    if change_id:
        ledger.set_status(proposal.proposal_id, "committed", change_id)
        prov = []
        g = app.graph()
        for oid in graph_ops:
            o = proposal.op(oid)
            eo = report["ops"][oid]["engine_op"] or {}
            ent = (eo.get("node") or eo.get("edge") or {}).get("id") or eo.get("id")
            if o.kind == "add_edge":  # resolve the committed edge id
                e = o.payload["edge"]
                live = [x for x in g.live_edges() if (x.src, x.dst, x.relation) == (e["src"], e["dst"], e["relation"])]
                ent = live[0].id if live else ent
            prov.append({"entity_id": ent, "proposal_id": proposal.proposal_id, "op_id": oid, "change_id": change_id,
                         "support": report["ops"][oid]["effective_support"],
                         "citations": [c for c in report["ops"][oid]["citations"] if c["status"] == "verified"]})
        ledger.record_provenance(prov)
    return {"proposal_id": proposal.proposal_id, "change_id": change_id, "rebased": rebased,
            "accepted": accepted, "rejected": [d.op_id for d in decisions if d.decision == "reject"],
            "deferred": [d.op_id for d in decisions if d.decision == "defer"],
            "no_change_records": [oid for oid in accepted if proposal.op(oid).kind == "no_change"],
            "commit": commit_result}


# ---------------------------------------------------------------------------
# Reviewer policies (for replay). A policy is *not* a human; reports say so.
# ---------------------------------------------------------------------------

def policy_reviewer(name: str = "policy:conservative-v1") -> Callable[[Proposal, dict[str, Any]], list[Decision]]:
    """Deterministic stand-in reviewer: accepts admissible ops, accepts duplicates as no-ops, accepts editorial
    no_change records, rejects invalid ops, defers everything quarantined."""

    def decide(proposal: Proposal, report: dict[str, Any]) -> list[Decision]:
        out = []
        for o in proposal.operations:
            v = report["ops"][o.op_id]["verdict"]
            if v in ("admissible", "duplicate"):
                out.append(Decision(o.op_id, "accept", name, f"policy: {v}"))
            elif v == "invalid":
                out.append(Decision(o.op_id, "reject", name, "policy: failed validation"))
            else:
                out.append(Decision(o.op_id, "defer", name, "policy: quarantined; needs a human decision"))
        return out

    return decide


def decisions_from_json(items: list[dict[str, Any]], default_reviewer: str) -> list[Decision]:
    return [Decision(op_id=i["op_id"], decision=i["decision"], reviewer=i.get("reviewer", default_reviewer),
                     rationale=i.get("rationale", ""), acknowledge=bool(i.get("acknowledge", False))) for i in items]
