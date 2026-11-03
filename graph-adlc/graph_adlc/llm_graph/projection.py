"""Spec-only Markdown projection of the committed graph.

Generated and read-only: it is readable without graph tooling, carries
traceable IDs, and distinguishes committed (human-approved) content from
pending proposals. Edits to the projection must be re-submitted as proposals.
Code is not exported: it stays as ordinary files in Git.
"""

from __future__ import annotations

from typing import Any

from ..models import Graph, NodeKind, Relation
from .ledger import Ledger


def _prov(ledger: Ledger | None, entity_id: str) -> str:
    if ledger is None:
        return ""
    rows = ledger.provenance(entity_id)
    if not rows:
        return ""
    refs = []
    for r in rows:
        for c in r["citations"][:2]:
            lines = c.get("resolved_lines") or c.get("lines")
            refs.append(f"{c['source_id'].split('@')[0]}{'' if not lines else f' L{lines[0]}-{lines[1]}'}")
    return f" _(source: {'; '.join(dict.fromkeys(refs))}; via {rows[-1]['proposal_id']}/{rows[-1]['change_id']})_" if refs else ""


def export_spec_markdown(graph: Graph, ledger: Ledger | None = None, title: str = "Specification") -> str:
    g = graph
    L: list[str] = [f"# {title}", "",
                    f"> Generated from the specification graph at revision {g.revision}. This file is read-only: "
                    "propose changes through the graph review process instead of editing it. Implementation code "
                    "is not included; it lives in the project's Git repository.", ""]

    def node_line(nid: str, depth: int = 0) -> list[str]:
        n = g.node(nid)
        if n is None:
            return []
        pad = "  " * depth
        out = [f"{pad}- **{n.id}** — {n.title}{_prov(ledger, n.id)}"]
        if n.body:
            out.append(f"{pad}  {n.body}")
        if n.kind == NodeKind.REQUIREMENT:
            ac = n.data.get("acceptance_criteria") or []
            if ac:
                out.append(f"{pad}  Acceptance criteria:")
                out += [f"{pad}  - {c}" for c in ac]
            else:
                out.append(f"{pad}  Acceptance criteria: _none recorded yet_")
            for a in n.data.get("assumptions") or []:
                st = a if isinstance(a, str) else f"{a.get('statement')} ({a.get('status', 'accepted')})"
                out.append(f"{pad}  Assumption: {st}")
            cons = [e.dst for e in g.out_edges(n.id, Relation.CONSTRAINED_BY)]
            deps = [e for e in g.out_edges(n.id, Relation.DEPENDS_ON)]
            if cons:
                out.append(f"{pad}  Constrained by: {', '.join(cons)}")
            if deps:
                out.append(f"{pad}  Depends on: {', '.join(e.dst + (' (unconfirmed)' if e.hypothesized else '') for e in deps)}")
            blockers = [e.src for e in g.in_edges(n.id, Relation.BLOCKS)
                        if (q := g.node(e.src)) is not None and q.data.get("state") == "open"]
            if blockers:
                out.append(f"{pad}  Blocked by open question(s): {', '.join(blockers)}")
        for child in sorted(g.children_of(nid)):
            out += node_line(child, depth + 1)
        return out

    goals = [n for n in g.live_nodes() if n.kind == NodeKind.GOAL]
    L.append("## Goals, capabilities and requirements")
    L.append("")
    if not goals:
        L.append("_No goals recorded._")
    for goal in goals:
        L += node_line(goal.id)
        crit = goal.data.get("success_criteria") or []
        if crit:
            L.append(f"  Success criteria: {'; '.join(crit)}")
    orphans = [n for n in g.live_nodes() if n.kind in (NodeKind.CAPABILITY, NodeKind.REQUIREMENT)
               and g.parent_of(n.id) is None]
    if orphans:
        L += ["", "### Not yet placed under a goal", ""]
        for n in orphans:
            L += node_line(n.id)

    contracts = [n for n in g.live_nodes() if n.kind == NodeKind.CONTRACT]
    L += ["", "## Contracts", ""]
    if not contracts:
        L.append("_None recorded._")
    for c in contracts:
        L.append(f"- **{c.id}** — {c.title} [status: {c.data.get('status')}]{_prov(ledger, c.id)}")
        if c.body:
            L.append(f"  {c.body}")
        subjects = sorted(e.src for e in g.in_edges(c.id, Relation.CONSTRAINED_BY))
        if subjects:
            L.append(f"  Applies to: {', '.join(subjects)}")

    comps = [n for n in g.live_nodes() if n.kind == NodeKind.COMPONENT]
    if comps:
        L += ["", "## Components (responsibilities only)", ""]
        for c in comps:
            impl = sorted(e.dst for e in g.out_edges(c.id, Relation.IMPLEMENTS))
            deps = g.out_edges(c.id, Relation.DEPENDS_ON)
            L.append(f"- **{c.id}** — {c.title}{_prov(ledger, c.id)}")
            if c.body:
                L.append(f"  {c.body}")
            if impl:
                L.append(f"  Implements: {', '.join(impl)}")
            if deps:
                L.append(f"  Depends on: {', '.join(e.dst + (' (unconfirmed)' if e.hypothesized else '') for e in deps)}")

    qs = [n for n in g.live_nodes() if n.kind == NodeKind.QUESTION]
    decided = [q for q in qs if q.data.get("state") == "resolved"]
    open_q = [q for q in qs if q.data.get("state") == "open"]
    L += ["", "## Decisions", ""]
    L += [f"- **{q.id}** — {q.title}: {q.body}{_prov(ledger, q.id)}" for q in decided] or ["_None recorded._"]
    L += ["", "## Open questions", ""]
    for q in open_q:
        tg = sorted(e.dst for e in g.out_edges(q.id, Relation.BLOCKS))
        L.append(f"- **{q.id}** {'(blocking) ' if q.data.get('blocking') else ''}— {q.title}"
                 f"{f' (blocks {', '.join(tg)})' if tg else ''}")
        if q.body:
            L.append(f"  {q.body}")
    if not open_q:
        L.append("_None._")

    hyp = [e for e in g.live_edges() if e.hypothesized]
    pending = ledger.deferred() if ledger else []
    L += ["", "## Proposed — not approved", ""]
    if not hyp and not pending:
        L.append("_Nothing pending._")
    for e in hyp:
        L.append(f"- Unconfirmed relationship: {e.src} {e.relation.replace('_', ' ')} {e.dst} ({e.id})")
    if pending and ledger is not None:
        for d in pending:
            p = ledger.proposal(d["proposal_id"])["proposal"]
            op = next((o for o in p["operations"] if o["op_id"] == d["op_id"]), None)
            if op is None:
                continue
            if op["kind"] == "add_node":
                what = op.get("node", {}).get("id")
            elif op["kind"] == "add_edge":
                e = op.get("edge", {})
                what = f"{e.get('src')} {e.get('relation')} {e.get('dst')}"
            else:
                what = op.get("id") or op.get("target")
            L.append(f"- Deferred {op['kind']} of {what} ({d['proposal_id']}/{d['op_id']}): {op.get('reason', '')}")
    L += ["", "---", "_Completeness note: this projection lists registered relationships only. An absent "
                    "relationship is not evidence that two items are independent._", ""]
    return "\n".join(L)
