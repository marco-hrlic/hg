"""Evidence, provenance and admissibility validation for proposals.

Validation is layered and conservative:

1. **Citations** are resolved against the supplied source snapshot (and the
   catalog of all known versions, to recognise stale citations). An excerpt
   must occur verbatim (whitespace-normalised) in the cited lines.
2. **Support** is recomputed: a model saying ``supported`` without a verified
   citation is downgraded. Numeric confidence never promotes anything.
3. **Policy checks**: duplicates, removals, authority (code cannot rewrite
   intent), contradictions, suspected editorial changes, resubmission of a
   previously rejected suggestion.
4. **Engine admissibility** via the existing ``GraphADLC.propose`` dry run.

Schema/engine validity does not imply semantic correctness; that is the
reviewer's job and is reported as such.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from ..core import GraphADLC
from ..models import EdgeStatus, Graph, NodeKind, Relation
from ..delta import MUTABLE_NODE_FIELDS
from ..validation import ENDPOINTS, validate_node_payload
from .ledger import Ledger
from .proposal import Citation, Proposal, ProposedOp
from .source import INTENT_AUTHORITY, SourceBundle, SourceCatalog, normalize_ws

DISCLAIMER = ("Schema, citation and engine validity do not imply semantic correctness. A verified excerpt shows "
              "the text exists, not that the proposed relationship follows from it. A missing edge is not proof "
              "of independence.")

# Verdicts, most severe first.
VERDICTS = ("invalid", "quarantined", "duplicate", "admissible")
INTENT_KINDS = {NodeKind.GOAL, NodeKind.CAPABILITY, NodeKind.REQUIREMENT, NodeKind.CONTRACT}


@dataclass
class Finding:
    code: str
    severity: str  # "error" -> invalid, "review" -> quarantined, "info"
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "severity": self.severity, "message": self.message}


@dataclass
class OpValidation:
    op_id: str
    verdict: str = "admissible"
    effective_support: str = "insufficient"
    findings: list[Finding] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    engine_op: dict[str, Any] | None = None

    def add(self, code: str, severity: str, message: str) -> None:
        self.findings.append(Finding(code, severity, message))

    def finalize(self) -> None:
        sev = {f.severity for f in self.findings}
        if "error" in sev:
            self.verdict = "invalid"
        elif "review" in sev:
            self.verdict = "quarantined"
        elif any(f.code == "duplicate" for f in self.findings):
            self.verdict = "duplicate"
        else:
            self.verdict = "admissible"

    def to_dict(self) -> dict[str, Any]:
        return {"op_id": self.op_id, "verdict": self.verdict, "effective_support": self.effective_support,
                "findings": [f.to_dict() for f in self.findings], "citations": self.citations,
                "engine_op": copy.deepcopy(self.engine_op)}


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

def check_citation(c: Citation, bundle: SourceBundle, catalog: SourceCatalog | None) -> dict[str, Any]:
    out = {**c.to_dict(), "status": "verified", "detail": ""}
    excerpt = normalize_ws(c.excerpt)
    diff = next((d for d in bundle.diffs if d["source_id"] == c.source_id), None)
    if diff is not None:
        if not excerpt:
            out.update(status="unlocatable", detail="empty excerpt")
        elif excerpt not in normalize_ws(_strip_diff_markers(diff["text"])) and excerpt not in normalize_ws(diff["text"]):
            out.update(status="fabricated", detail="excerpt does not occur in the cited diff")
        return out
    doc = bundle.get(c.source_id)
    if doc is None:
        known = catalog.versions(c.source_id) if catalog else {}
        out.update(status="stale" if known else "fabricated",
                   detail="source not in the supplied snapshot" + (" (older version known)" if known else ""))
        return out
    if c.sha256 and c.sha256 != doc.sha256:
        if catalog and catalog.known(c.source_id, c.sha256):
            out.update(status="stale", detail=f"cites superseded version {c.sha256[:12]}; current is {doc.sha256[:12]}")
        else:
            out.update(status="fabricated", detail="sha256 does not match any known version of this source")
        return out
    if not excerpt:
        out.update(status="unlocatable", detail="empty excerpt")
        return out
    if c.line_start is not None:
        span = doc.span(c.line_start, c.line_end or c.line_start)
        if span is None:
            out.update(status="fabricated", detail=f"lines {c.line_start}-{c.line_end} out of range")
            return out
        if excerpt not in normalize_ws(span):
            loc = _locate(doc.lines, excerpt)
            if loc:
                out.update(status="verified", detail=f"excerpt found at lines {loc[0]}-{loc[1]}, not as cited",
                           resolved_lines=list(loc))
            else:
                out.update(status="fabricated", detail="excerpt does not occur in the source")
            return out
        out["resolved_lines"] = [c.line_start, c.line_end or c.line_start]
        return out
    loc = _locate(doc.lines, excerpt)
    if loc is None:
        out.update(status="fabricated", detail="excerpt does not occur in the source")
    else:
        out["resolved_lines"] = list(loc)
    return out


def _strip_diff_markers(text: str) -> str:
    return "\n".join(l[1:] if l[:1] in "+- " and not l.startswith(("+++", "---")) else l
                     for l in text.splitlines())


def _locate(lines: list[str], excerpt: str, max_span: int = 40) -> tuple[int, int] | None:
    """Smallest 1-based line span whose whitespace-normalised text contains ``excerpt``."""
    norm = [normalize_ws(l) for l in lines]
    for j in range(len(lines)):
        acc = ""
        for i in range(j, max(-1, j - max_span), -1):
            acc = normalize_ws(norm[i] + " " + acc)
            if excerpt in acc:
                return (i + 1, j + 1)
    return None


def _norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


# ---------------------------------------------------------------------------
# Translation to the engine's delta format
# ---------------------------------------------------------------------------

def edge_id_for(src: str, relation: str, dst: str, taken: set[str]) -> str:
    base = f"LE-{src}--{relation}--{dst}"
    eid, n = base, 1
    while eid in taken:
        n += 1
        eid = f"{base}-{n}"
    return eid


def translate_op(op: ProposedOp, proposal_id: str, effective_support: str, taken_edge_ids: set[str]) -> dict[str, Any] | None:
    p = op.payload
    if op.kind == "no_change":
        return None
    if op.kind == "add_node":
        n = p["node"]
        return {"op": "add_node", "node": {"id": n["id"], "kind": n["kind"], "title": n.get("title", ""),
                                           "body": n.get("body", ""), "data": copy.deepcopy(n.get("data", {}))}}
    if op.kind == "add_edge":
        e = p["edge"]
        eid = edge_id_for(e["src"], e["relation"], e["dst"], taken_edge_ids)
        taken_edge_ids.add(eid)
        status = EdgeStatus.CONFIRMED.value if effective_support == "supported" and e.get("status", "confirmed") == "confirmed" \
            else EdgeStatus.HYPOTHESIZED.value
        return {"op": "add_edge", "edge": {"id": eid, "src": e["src"], "dst": e["dst"], "relation": e["relation"],
                                           "status": status, "evidence_ref": f"{proposal_id}/{op.op_id}"}}
    if op.kind == "update_node":
        out = {"op": "update_node", "id": p["id"], "expected_rev": p.get("expected_rev"),
               "patch": copy.deepcopy(p.get("patch", {}))}
        if p.get("aspect"):
            out["aspect"] = p["aspect"]
        return out
    return {"op": op.kind, "id": p.get("id"), "expected_rev": p.get("expected_rev")}


def build_delta(proposal: Proposal, op_ids: list[str], validation: dict[str, Any], base_graph_revision: int,
                actor: str, graph: Graph) -> dict[str, Any]:
    taken = set(graph.edges)
    ops = []
    for o in proposal.operations:
        if o.op_id not in op_ids:
            continue
        v = validation["ops"][o.op_id]
        eo = translate_op(o, proposal.proposal_id, v["effective_support"], taken)
        if eo is not None:
            ops.append(eo)
    return {"base_graph_revision": base_graph_revision, "actor": actor,
            "rationale": f"{proposal.proposal_id}: {proposal.intent} [ops {', '.join(op_ids)}]",
            "operations": ops}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_proposal(app: GraphADLC, proposal: Proposal, bundle: SourceBundle,
                      catalog: SourceCatalog | None = None, ledger: Ledger | None = None) -> tuple[Proposal, dict[str, Any]]:
    """Return the (possibly augmented) proposal and a validation report.

    Augmentation: a declared contradiction about an existing intent node yields a
    validator-generated *question* suggestion (never an approved replacement).
    """
    g = app.graph()
    proposal = copy.deepcopy(proposal)
    proposal.operations.extend(_contradiction_questions(proposal, g, bundle, catalog))
    results: dict[str, OpValidation] = {}
    added_nodes = {o.payload.get("node", {}).get("id"): o for o in proposal.operations if o.kind == "add_node"}
    contradicted = {nid for c in proposal.possible_contradictions for nid in (c.get("between") or [])}
    seen_edges: dict[tuple[str, str, str], str] = {}

    for o in proposal.operations:
        v = OpValidation(o.op_id)
        results[o.op_id] = v
        _check_evidence(o, v, bundle, catalog)
        _check_op(o, v, g, added_nodes, contradicted, seen_edges, bundle)
        if ledger is not None:
            prior = ledger.rejected(o.fingerprint())
            if prior:
                key = evidence_key(v.citations)
                same = [r for r in prior if r["evidence_key"] == key]
                if same:
                    v.add("previously_rejected", "review",
                          f"identical suggestion rejected in {same[0]['proposal_id']}/{same[0]['op_id']} "
                          f"({same[0]['rationale']!r}) and resubmitted without new evidence")
                else:
                    v.add("previously_rejected_new_evidence", "info",
                          f"similar suggestion rejected in {prior[0]['proposal_id']}; new evidence supplied")
        if any(k in o.payload.get("patch", {}) for k in ("contract_preserved",)) or o.payload.get("contract_preserved"):
            v.add("unverified_preservation_claim", "review",
                  "a model's claim that a contract is preserved is not evidence and cannot justify pruning")
        v.finalize()

    # Dependent-op propagation: an edge whose endpoint node op is invalid is itself invalid.
    _propagate_dependencies(proposal, results, g)

    # Engine admissibility dry run over everything that could be accepted.
    engine = _engine_dry_run(app, proposal, results, g)

    report = {
        "proposal_id": proposal.proposal_id,
        "graph_revision": g.revision,
        "base_graph_revision": proposal.base_graph_revision,
        "stale_base": proposal.base_graph_revision != g.revision,
        "ops": {k: v.to_dict() for k, v in results.items()},
        "summary": {verdict: sorted(k for k, v in results.items() if v.verdict == verdict) for verdict in VERDICTS},
        "unresolved_questions": proposal.unresolved_questions,
        "possible_contradictions": proposal.possible_contradictions,
        "engine_preview": engine,
        "disclaimer": DISCLAIMER,
    }
    return proposal, report


def evidence_key(citations: list[dict[str, Any]]) -> str:
    return "|".join(sorted(f"{c['source_id']}@{(c.get('sha256') or '')[:12]}:{c.get('resolved_lines')}"
                           for c in citations if c.get("status") == "verified"))


def _check_evidence(o: ProposedOp, v: OpValidation, bundle: SourceBundle, catalog: SourceCatalog | None) -> None:
    for c in o.evidence:
        v.citations.append(check_citation(c, bundle, catalog))
    bad = [c for c in v.citations if c["status"] in ("fabricated", "stale")]
    for c in bad:
        v.add(f"{c['status']}_citation", "error", f"{c['source_id']}: {c['detail']}")
    verified = [c for c in v.citations if c["status"] == "verified"]
    if o.support == "supported" and verified and not bad:
        v.effective_support = "supported"
    elif o.support == "supported" and not verified:
        v.effective_support = "insufficient"
        v.add("unsupported_claim", "review", "marked 'supported' but no citation could be verified")
    elif o.support == "hypothesized":
        v.effective_support = "hypothesized"
    else:
        v.effective_support = "insufficient" if not verified else "hypothesized"
    if o.kind != "no_change" and v.effective_support == "insufficient" and not bad:
        v.add("insufficient_support", "review", "no verified evidence; surface as a question rather than a fact")
    for c in v.citations:
        if c["status"] == "unlocatable":
            v.add("unlocatable_citation", "review", f"{c['source_id']}: {c['detail']}")


def _check_op(o: ProposedOp, v: OpValidation, g: Graph, added_nodes: dict[str, ProposedOp],
              contradicted: set[str], seen_edges: dict[tuple[str, str, str], str], bundle: SourceBundle) -> None:
    p = o.payload
    if o.kind == "add_node":
        n = p.get("node") or {}
        nid = n.get("id")
        if not nid:
            v.add("schema", "error", "add_node requires node.id")
            return
        for m in validate_node_payload(nid, n.get("kind", ""), n.get("title"), n.get("data", {})):
            v.add("schema", "error", m)
        ex = g.nodes.get(nid)
        if ex is not None:
            if ex.deleted:
                v.add("id_tombstoned", "error", f"{nid} was removed earlier; ids are never reused")
            elif (ex.kind, ex.title, ex.body, ex.data) == (n.get("kind"), n.get("title"), n.get("body", ""), n.get("data", {})) \
                    or (ex.kind == n.get("kind") and _norm_text(ex.title) == _norm_text(n.get("title", ""))):
                v.add("duplicate", "info", f"{nid} already exists with equivalent content; nothing to add")
            else:
                v.add("id_exists", "review",
                      f"{nid} already exists (r{ex.rev}) with different content; an update_node is required instead")
        else:
            for other in g.live_nodes():
                if other.kind == n.get("kind") and _norm_text(other.title) == _norm_text(n.get("title", "")):
                    v.add("duplicate_suspected", "review",
                          f"same kind and title as existing {other.id}; likely a paraphrased duplicate")
        return
    if o.kind == "add_edge":
        e = p.get("edge") or {}
        src, dst, rel = e.get("src"), e.get("dst"), e.get("relation")
        if rel not in set(Relation):
            v.add("unsupported_relation", "error", f"relation {rel!r} is not in the ontology {sorted(Relation)}")
            return
        kinds = {}
        for end in (src, dst):
            node = g.node(end) if end else None
            if node is not None:
                kinds[end] = node.kind
            elif end in added_nodes:
                kinds[end] = added_nodes[end].payload.get("node", {}).get("kind")
            else:
                v.add("unknown_endpoint", "error", f"endpoint {end!r} does not exist and is not added by this proposal")
        if len(kinds) == 2 or (src == dst and src in kinds):
            allowed_src, allowed_dst = ENDPOINTS[rel]
            if kinds.get(src) not in allowed_src or kinds.get(dst) not in allowed_dst:
                v.add("invalid_endpoints", "error", f"{rel} not allowed from {kinds.get(src)} to {kinds.get(dst)}")
        if e.get("status") not in (None, "confirmed", "hypothesized"):
            v.add("schema", "error", f"invalid edge status {e.get('status')!r}")
        key = (src, dst, rel)
        live = [x for x in g.live_edges() if (x.src, x.dst, x.relation) == key]
        if live:
            v.add("duplicate", "info", f"edge already registered as {live[0].id}; nothing to add")
        elif key in seen_edges:
            v.add("duplicate", "info", f"same edge already proposed as {seen_edges[key]}")
        else:
            seen_edges[key] = o.op_id
        if v.effective_support == "hypothesized":
            v.add("stored_as_hypothesized", "info", "will be stored with status 'hypothesized' (raises investigate_unknown)")
        return
    if o.kind == "update_node":
        nid = p.get("id")
        node = g.node(nid) if nid else None
        if node is None:
            v.add("unknown_target", "error", f"update target {nid!r} does not exist")
            return
        patch = p.get("patch") or {}
        illegal = set(patch) - MUTABLE_NODE_FIELDS
        if not patch or illegal:
            v.add("schema", "error", f"patch must be non-empty and only touch {sorted(MUTABLE_NODE_FIELDS)}")
        if p.get("expected_rev") is None:
            v.add("schema", "error", "update_node requires expected_rev from the supplied graph snapshot")
        elif p["expected_rev"] != node.rev:
            v.add("revision_conflict", "error", f"{nid} expected r{p['expected_rev']} but current is r{node.rev}")
        title_same = "title" not in patch or _norm_text(patch["title"]) == _norm_text(node.title)
        body_same = "body" not in patch or _norm_text(patch["body"]) == _norm_text(node.body)
        if title_same and body_same and "data" not in patch:
            v.add("editorial_suspected", "review",
                  "text differs only in case/punctuation/whitespace; the engine has no editorial aspect, so "
                  "committing this would propagate as a semantic change")
        if o.change_class == "editorial":
            v.add("editorial_declared", "review",
                  "declared editorial, but update_node always propagates as semantic; prefer a no_change record")
        if node.kind in INTENT_KINDS:
            types = {(bundle.get(c["source_id"]).source_type if bundle.get(c["source_id"]) else "code")
                     for c in v.citations if c["status"] == "verified"}
            if types and max(INTENT_AUTHORITY.get(t, 0) for t in types) < 2:
                v.add("authority_conflict", "review",
                      f"{nid} is a {node.kind}; code/test evidence cannot change business intent. Raise a question "
                      "or decision instead")
        if nid in contradicted:
            v.add("contradiction", "review", f"{nid} is named in a possible contradiction; needs a human decision")
        return
    if o.kind in ("remove_node", "remove_edge"):
        eid = p.get("id")
        ent = (g.node(eid) if o.kind == "remove_node" else
               (g.edges.get(eid) if eid in g.edges and not g.edges[eid].deleted else None))
        if ent is None:
            v.add("unknown_target", "error", f"{o.kind} target {eid!r} does not exist")
        if p.get("expected_rev") is None:
            v.add("schema", "error", f"{o.kind} requires expected_rev")
        if v.effective_support != "supported":
            v.add("removal_without_evidence", "error",
                  "a source being silent is not evidence of removal; an explicit cited statement is required")
        v.add("requires_authorization", "review", "removals always need explicit human authorization")
        return
    if o.kind == "no_change":
        if o.change_class == "editorial":
            v.add("editorial_no_op", "info", "recorded as wording-only; no graph change and no propagation")


def _propagate_dependencies(proposal: Proposal, results: dict[str, OpValidation], g: Graph) -> None:
    invalid_nodes = {o.payload["node"]["id"] for o in proposal.operations
                     if o.kind == "add_node" and results[o.op_id].verdict == "invalid" and o.payload.get("node", {}).get("id")}
    for o in proposal.operations:
        if o.kind != "add_edge":
            continue
        e = o.payload.get("edge", {})
        hit = {e.get("src"), e.get("dst")} & invalid_nodes
        if hit and g.node(next(iter(hit))) is None:
            results[o.op_id].add("depends_on_invalid_op", "error", f"endpoint node op for {sorted(hit)} is invalid")
            results[o.op_id].finalize()


def _engine_dry_run(app: GraphADLC, proposal: Proposal, results: dict[str, OpValidation], g: Graph) -> dict[str, Any]:
    """Dry-run all non-invalid, non-duplicate ops through GraphADLC.propose. Engine errors invalidate ops."""
    for _ in range(len(proposal.operations) + 1):
        candidate = [o.op_id for o in proposal.operations if results[o.op_id].verdict in ("admissible", "quarantined")]
        report = {k: v.to_dict() for k, v in results.items()}
        delta = build_delta(proposal, candidate, {"ops": report}, g.revision, "llm-graph:dry-run", g)
        engine_ops = [o for o in proposal.operations if o.op_id in candidate and o.kind != "no_change"]
        for o, eo in zip(engine_ops, delta["operations"]):
            results[o.op_id].engine_op = eo
        if not delta["operations"]:
            return {"ran": False, "reason": "no graph operations to preview"}
        prev = app.propose(delta)
        if prev["committable"]:
            imp = prev.get("impact", {})
            return {"ran": True, "committable": True, "op_ids": candidate,
                    "impact_records": imp.get("obligations", []), "required_actions": prev.get("required_actions", {}),
                    "readiness_changes": prev.get("readiness_changes", []),
                    "would_invalidate": prev.get("would_invalidate", {}), "diff": prev.get("diff", [])}
        blamed = False
        for msg in prev["errors"]:
            m = re.match(r"op\[(\d+)\]", msg)
            if m and int(m.group(1)) < len(engine_ops):
                o = engine_ops[int(m.group(1))]
                results[o.op_id].add("engine_rejected", "error", msg)
                results[o.op_id].finalize()
                blamed = True
        if not blamed:
            # Graph-level invariant failure: cannot attribute to one op; quarantine all candidates.
            return {"ran": True, "committable": False, "errors": prev["errors"], "conflicts": prev["conflicts"],
                    "op_ids": candidate}
    return {"ran": True, "committable": False, "errors": ["dry run did not converge"]}


def _contradiction_questions(proposal: Proposal, g: Graph, bundle: SourceBundle,
                             catalog: SourceCatalog | None) -> list[ProposedOp]:
    """A contradiction about an existing intent node becomes a *question* suggestion (L07)."""
    out = []
    existing_q = {o.payload.get("node", {}).get("id") for o in proposal.operations if o.kind == "add_node"}
    for i, c in enumerate(proposal.possible_contradictions, 1):
        about = [nid for nid in (c.get("between") or []) if g.node(nid) is not None]
        if not about:
            continue
        qid = f"Q-{proposal.proposal_id}-C{i}"
        if qid in existing_q or g.nodes.get(qid) is not None:
            continue
        cites = []
        for raw in c.get("source_evidence") or []:
            cit = Citation.parse(raw, "contradiction", [])
            if cit:
                cites.append(cit)
        desc = str(c.get("description", "Possible contradiction"))
        out.append(ProposedOp(
            op_id=f"auto-q{i}", kind="add_node",
            payload={"node": {"id": qid, "kind": "question", "title": "Resolve: " + (desc if len(desc) <= 90 else desc[:90].rsplit(" ", 1)[0] + " …"), "body": desc,
                              "data": {"blocking": True, "state": "open", "about": about}}},
            reason="validator: contradiction requires a human decision, not a silent replacement",
            support="supported" if cites else "hypothesized", evidence=cites, change_class="semantic"))
        targets = [nid for nid in about if g.node(nid).kind in ENDPOINTS[Relation.BLOCKS][1]]
        for j, t in enumerate(targets, 1):
            out.append(ProposedOp(
                op_id=f"auto-q{i}-b{j}", kind="add_edge",
                payload={"edge": {"src": qid, "dst": t, "relation": "blocks"}},
                reason="validator: open contradiction blocks the affected node until decided",
                support="supported" if cites else "hypothesized", evidence=cites, change_class="structure"))
    return out
