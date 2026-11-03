"""Deterministic readable-spec renderer and the frozen document baseline (method A).

The renderer turns the *pre-change* graph (after scenario setup) into Markdown that a
person could review: one section per node, its text, data, and every relationship
written out at both endpoints. Method A then works purely on that text: it parses the
sections back out and selects candidates by ID cross-references. It never touches the
graph object or the influence engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from graph_adlc.models import Graph

# (relation) -> (phrase at src, phrase at dst)
PHRASES = {
    "refines": ("Refined into", "Refines parent"),
    "depends_on": ("Depends on", "Depended on by"),
    "constrained_by": ("Constrained by", "Constrains"),
    "implements": ("Implements", "Implemented by"),
    "verifies": ("Verifies", "Verified by"),
    "realized_by": ("Realized by", "Realizes"),
    "blocks": ("Blocks", "Blocked by"),
}
HEADER = re.compile(r"^## (?P<id>[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+) — (?P<title>.*)$")
KIND = re.compile(r"^Kind: (?P<kind>[a-z]+)")


def _fmt(v: Any) -> str:
    if isinstance(v, list):
        return "; ".join(_fmt(x) for x in v) if v else "(none)"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in sorted(v.items()))
    return str(v)


def render_spec(g: Graph, evidence: list[dict[str, Any]], title: str) -> str:
    lines = [f"# {title}", "", f"Specification snapshot at graph revision {g.revision}. "
             "Relationships are listed at both ends; '(unconfirmed)' marks a suspected relationship.", ""]
    for n in g.live_nodes():
        lines += [f"## {n.id} — {n.title}", f"Kind: {n.kind} · Revision: r{n.rev}", ""]
        if n.body:
            lines += [n.body, ""]
        for k, v in sorted(n.data.items()):
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                lines.append(f"- {k.replace('_', ' ')}:")
                lines += [f"  - {x}" for x in v]
            else:
                lines.append(f"- {k.replace('_', ' ')}: {_fmt(v)}")
        rels = []
        for e in g.live_edges():
            if n.id not in (e.src, e.dst):
                continue
            at_src, at_dst = PHRASES[e.relation]
            phrase, other = (at_src, e.dst) if e.src == n.id else (at_dst, e.src)
            mark = " (unconfirmed)" if e.hypothesized else ""
            rels.append(f"- {phrase} {other}{mark} [{e.id}]")
        if rels:
            lines += ["", "Relationships:"] + rels
        lines.append("")
    lines += ["## Evidence log", ""]
    if not evidence:
        lines.append("(no evidence records)")
    for ev in evidence:
        revs = ", ".join(f"{k} r{v}" for k, v in sorted(ev["spec_revs"].items()))
        stale = f"; invalidated by {ev['invalidated_by']}" if ev.get("invalidated_by") else ""
        lines.append(f"- {ev['id']}: {ev['kind']} on {ev['node_id']} = {ev['result']} (recorded against {revs}{stale})")
    return "\n".join(lines) + "\n"


def render_change_request(delta: dict[str, Any]) -> str:
    lines = ["# Change request", "", f"Rationale: {delta.get('rationale', '')}", ""]
    for i, op in enumerate(delta["operations"]):
        kind = op["op"]
        if kind == "update_node":
            lines.append(f"- op[{i}] update {op['id']} (declared aspect: {op.get('aspect', 'semantic')}); "
                         f"new values: {_fmt(op.get('patch', {}))}")
        elif kind in ("add_node", "remove_node"):
            nid = op.get("id") or op.get("node", {}).get("id")
            lines.append(f"- op[{i}] {kind.replace('_', ' ')} {nid}")
        elif kind == "add_edge":
            e = op["edge"]
            lines.append(f"- op[{i}] add relationship {e['src']} {e['relation']} {e['dst']} ({e.get('status', 'confirmed')})")
        elif kind == "remove_edge":
            lines.append(f"- op[{i}] remove relationship {op['id']}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Method A: parse the rendered document and select by cross-references
# ---------------------------------------------------------------------------

@dataclass
class Section:
    id: str
    kind: str
    text: str
    refs: dict[str, bool] = field(default_factory=dict)  # referenced id -> via an '(unconfirmed)' line


def parse_sections(doc: str) -> dict[str, Section]:
    sections: dict[str, Section] = {}
    cur: list[str] = []
    cur_id = None

    def flush() -> None:
        if cur_id is None:
            return
        text = "\n".join(cur)
        kind = next((m.group("kind") for ln in cur if (m := KIND.match(ln))), "unknown")
        sections[cur_id] = Section(cur_id, kind, text)

    for ln in doc.splitlines():
        m = HEADER.match(ln)
        if m or ln.startswith("## "):
            flush()
            cur, cur_id = [ln], (m.group("id") if m else None)
        else:
            cur.append(ln)
    flush()
    ids = sorted(sections, key=len, reverse=True)
    pattern = re.compile(r"(?<![A-Z0-9-])(" + "|".join(re.escape(i) for i in ids) + r")(?![A-Z0-9-])") if ids else None
    for s in sections.values():
        for ln in s.text.splitlines()[1:]:  # skip own header
            for m in (pattern.finditer(ln) if pattern else []):
                rid = m.group(1)
                if rid != s.id:
                    s.refs[rid] = s.refs.get(rid, False) or "(unconfirmed)" in ln
    return sections


def _obligation(target_kind: str, seed_kind: str, table: dict[str, Any]) -> str:
    rule = table.get(target_kind, "review_spec")
    if isinstance(rule, str):
        return rule
    return rule["then"] if seed_kind in rule["if_seed_kind_in"] else rule["else"]


def document_baseline(doc: str, delta: dict[str, Any], cfg: dict[str, Any], table: dict[str, Any],
                      method: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Frozen procedure from config.json (methods.A / A2). Returns (predictions, trace)."""
    sections = parse_sections(doc)
    hops, budget = int(cfg["hops"]), int(cfg["retrieval_budget_sections"])
    preds: list[dict[str, Any]] = []
    node_seeds: list[tuple[str, str]] = []  # (id, kind)
    for i, op in enumerate(delta["operations"]):
        if op["op"] in ("update_node", "remove_node"):
            sid = op["id"]
            node_seeds.append((sid, sections[sid].kind if sid in sections else "unknown"))
        elif op["op"] == "add_node":
            node_seeds.append((op["node"]["id"], op["node"]["kind"]))
        elif op["op"] in ("add_edge", "remove_edge"):
            if op["op"] == "add_edge":
                src, dst, status = op["edge"]["src"], op["edge"]["dst"], op["edge"].get("status", "confirmed")
            else:
                src, dst, status = _edge_endpoints(sections, op["id"])
            for a, b in ((src, dst), (dst, src)):
                if a in sections:
                    k_a, k_b = sections[a].kind, sections[b].kind if b in sections else "unknown"
                    expl = f"op[{i}] changes the relationship between {a} and {b} (document change request)"
                    preds.append(_pred(method, a, _obligation(k_a, k_b, table), expl, 0))
                    if status == "hypothesized":
                        preds.append(_pred(method, a, "investigate_unknown", expl + "; relationship unconfirmed", 0))
    seed_ids = {s for s, _ in node_seeds}
    found: dict[str, tuple[int, str, str, bool]] = {}  # id -> (hop, seed_id, via, unconfirmed)
    frontier = [(s, s) for s, _ in node_seeds]
    for hop in range(1, hops + 1):
        nxt = []
        for cur, seed in frontier:
            neighbours: dict[str, bool] = {}
            if cur in sections:
                for rid, unc in sections[cur].refs.items():
                    neighbours[rid] = neighbours.get(rid, False) or unc
            for s in sections.values():
                if cur in s.refs:
                    neighbours[s.id] = neighbours.get(s.id, False) or s.refs[cur]
            for nid, unc in sorted(neighbours.items()):
                if nid in seed_ids or nid in found:
                    continue
                found[nid] = (hop, seed, cur, unc)
                nxt.append((nid, seed))
        frontier = nxt
    ranked = sorted(found.items(), key=lambda kv: (kv[1][0], kv[0]))
    kept, dropped = ranked[:budget], ranked[budget:]
    seed_kind = dict(node_seeds)
    for nid, (hop, seed, via, unc) in kept:
        expl = (f"cross-reference between section {nid} and section {via}" +
                (f" (hop {hop} from changed {seed})" if hop > 1 else " (changed in the change request)"))
        preds.append(_pred(method, nid, _obligation(sections[nid].kind, seed_kind[seed], table), expl, hop))
        if unc:
            preds.append(_pred(method, nid, "investigate_unknown", expl + "; reference marked unconfirmed", hop))
    context_ids = sorted(seed_ids | {n for n, _ in kept} | {p["target_id"] for p in preds})
    trace = {"seeds": sorted(seed_ids), "retrieved": [n for n, _ in kept], "over_budget": [n for n, _ in dropped],
             "context_sections": context_ids,
             "context_chars": sum(len(sections[i].text) for i in context_ids if i in sections),
             "document_chars": len(doc)}
    return preds, trace


def _edge_endpoints(sections: dict[str, Section], edge_id: str) -> tuple[str, str, str]:
    """Recover a removed relationship's endpoints from its '[Exx]' tags in the document text."""
    tag = f"[{edge_id}]"
    src = dst = None
    status = "confirmed"
    for s in sections.values():
        for ln in s.text.splitlines():
            if ln.endswith(tag):
                if "(unconfirmed)" in ln:
                    status = "hypothesized"
                phrase_src = any(ln.startswith(f"- {p[0]} ") for p in PHRASES.values())
                if phrase_src:
                    src = s.id
                else:
                    dst = s.id
    if src is None or dst is None:
        raise KeyError(f"relationship {edge_id} not found in document")
    return src, dst, status


def _pred(method: str, target: str, obligation: str, explanation: str, hop: int) -> dict[str, Any]:
    return {"method": method, "target_id": target, "obligation": obligation, "explanation": explanation,
            "path_edge_ids": [], "pruned": False, "provisional": False, "confidence": None, "hop": hop}
