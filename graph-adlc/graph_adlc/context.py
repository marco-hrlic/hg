"""Scoped work-context packets (SPEC §4, "Context materialization").

The packet is the source of truth; :func:`render_text` is merely a view.
Inclusion is rule-based and every included node carries the rule that pulled it
in, so the packet can be audited. Descendants, siblings and the rest of the
graph are not included unless an explicit relevant edge connects them.
"""

from __future__ import annotations

from typing import Any

from .models import Graph, NodeKind, Relation
from .readiness import readiness
from .validation import normalize_assumptions

R = Relation


def context_packet(g: Graph, requirement_id: str) -> dict[str, Any]:
    target = g.node(requirement_id)
    if target is None or target.kind != NodeKind.REQUIREMENT:
        raise KeyError(f"{requirement_id} is not a live requirement")

    included: dict[str, str] = {requirement_id: "target"}

    def add(nid: str, why: str) -> None:
        if g.node(nid) is not None and nid not in included:
            included[nid] = why

    for anc in g.ancestors(requirement_id):
        add(anc, "refinement ancestor")
    for e in g.out_edges(requirement_id, R.CONSTRAINED_BY):
        add(e.dst, f"contract constraining target ({e.id})")
    for e in g.out_edges(requirement_id, R.DEPENDS_ON):
        add(e.dst, f"provider of target ({e.id})")
        for c in g.out_edges(e.dst, R.CONSTRAINED_BY):
            add(c.dst, f"contract of provider {e.dst} ({c.id})")
    for e in g.in_edges(requirement_id, R.IMPLEMENTS):
        add(e.src, f"component implementing target ({e.id})")
        for p in g.out_edges(e.src, R.DEPENDS_ON):
            add(p.dst, f"provider of implementing component {e.src} ({p.id})")
            for c in g.out_edges(p.dst, R.CONSTRAINED_BY):
                add(c.dst, f"contract of provider {p.dst} ({c.id})")
    for e in g.in_edges(requirement_id, R.VERIFIES):
        add(e.src, f"test verifying target ({e.id})")
    for e in g.in_edges(requirement_id, R.BLOCKS):
        add(e.src, f"question blocking/qualifying target ({e.id})")

    ids = set(included)
    nodes = []
    for nid in sorted(ids):
        n = g.nodes[nid]
        nodes.append({"id": n.id, "kind": n.kind, "rev": n.rev, "title": n.title, "body": n.body,
                      "data": n.data, "included_because": included[nid]})
    edges = [e.to_dict() for e in g.live_edges() if e.src in ids and e.dst in ids]
    r = readiness(g, requirement_id)
    return {
        "requirement_id": requirement_id,
        "graph_revision": g.revision,
        "node_revisions": g.revs(ids),
        "nodes": nodes,
        "edges": edges,
        "acceptance_criteria": list(target.data.get("acceptance_criteria", [])),
        "assumptions": normalize_assumptions(target.data.get("assumptions"), target.id),
        "readiness": r.to_dict(),
        "scope_note": ("Bounded context: target, refinement ancestors, linked contracts, providers and their "
                       "contracts, implementing components, verifying tests and blocking questions only."),
    }


def render_text(packet: dict[str, Any]) -> str:
    lines = [f"# Work context for {packet['requirement_id']} (graph rev {packet['graph_revision']})", ""]
    lines.append(f"Readiness: {packet['readiness']['readiness']}")
    if packet["acceptance_criteria"]:
        lines.append("Acceptance criteria:")
        lines += [f"  - {c}" for c in packet["acceptance_criteria"]]
    if packet["assumptions"]:
        lines.append("Assumptions (tracked):")
        lines += [f"  - [{a['id']}] {a['statement']} ({a.get('status')})" for a in packet["assumptions"]]
    lines.append("Nodes:")
    for n in packet["nodes"]:
        lines.append(f"  - {n['id']}@r{n['rev']} [{n['kind']}] {n['title']} -- {n['included_because']}")
    lines.append("Edges:")
    for e in packet["edges"]:
        lines.append(f"  - {e['id']}: {e['src']} -{e['relation']}-> {e['dst']} ({e['status']})")
    return "\n".join(lines)
