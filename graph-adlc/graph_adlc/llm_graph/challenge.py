"""Graph-change challenge: look for plausible relationships the graph is missing.

Given an intended change (node + aspect, or a delta), compute the engine's
candidate influence domain, then search the supplied code for dependencies the
graph does not register. Two sources of candidates:

* a deterministic, explainable **import scan** (Python ``import``/``from``
  statements between files that registered artifacts realise), and
* optionally an **LLM challenger** (``LLMProposer.challenge``).

Both return *candidate proposals*: nothing is inserted until reviewed.
"""

from __future__ import annotations

import ast
from typing import Any

from ..core import GraphADLC
from ..models import Graph, Relation
from .model_adapter import LLMProposer
from .proposal import Citation, Proposal, ProposedOp
from .source import SourceBundle, SourceDocument


def influence_domain(app: GraphADLC, change: dict[str, Any]) -> dict[str, Any]:
    """``change``: {"node_id", "aspect"} (hypothetical) or {"delta": {...}} (dry-run proposal)."""
    if "delta" in change:
        prev = app.propose(change["delta"])
        recs = prev.get("impact", {}).get("obligations", [])
        origin = sorted({op.get("id") or (op.get("node") or op.get("edge") or {}).get("id")
                         for op in change["delta"]["operations"]})
    else:
        g = app.graph()
        if g.node(change["node_id"]) is None:
            return {"origin": [change["node_id"]], "targets": [], "records": [],
                    "note": "origin node is not in the graph; the engine cannot compute a domain"}
        recs = [r.to_dict() for r in app.impact(change["node_id"], change["aspect"]).records]
        origin = [change["node_id"]]
    targets = sorted({r["target_id"] for r in recs})
    return {"origin": origin, "targets": targets,
            "records": [{k: r[k] for k in ("target_id", "obligation", "path", "confidence")} for r in recs],
            "note": "complete only relative to registered edges; a missing edge is not proof of independence"}


def _module_paths(module: str) -> list[str]:
    p = module.replace(".", "/")
    return [f"{p}.py", f"{p}/__init__.py"]


def _imports(doc: SourceDocument) -> list[tuple[str, int, str]]:
    """(module, line, line_text) for each import in a Python file; empty for non-Python or unparsable files."""
    if not doc.path.endswith(".py"):
        return []
    try:
        tree = ast.parse(doc.content)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append((node.module, node.lineno))
            for a in node.names:
                out.append((f"{node.module}.{a.name}", node.lineno))
    lines = doc.lines
    return [(m, ln, lines[ln - 1]) for m, ln in sorted(set(out), key=lambda x: (x[1], x[0]))]


def _artifact_index(g: Graph) -> dict[str, tuple[str, str]]:
    """repo path -> (artifact id, component id)."""
    idx = {}
    for e in g.live_edges():
        if e.relation == Relation.REALIZED_BY:
            art = g.node(e.dst)
            if art is not None and art.data.get("path"):
                idx[art.data["path"]] = (art.id, e.src)
    return idx


def _match(path_index: dict[str, Any], candidates: list[str]) -> str | None:
    for p in path_index:
        for c in candidates:
            if p == c or p.endswith("/" + c):
                return p
    return None


def scan_code_dependencies(graph: Graph, bundle: SourceBundle, domain: dict[str, Any] | None = None,
                           proposal_id: str = "SCAN") -> Proposal:
    idx = _artifact_index(graph)
    code_paths = {d.path: d for d in bundle.documents if d.source_type in ("code", "test")}
    registered = {(e.src, e.dst) for e in graph.live_edges() if e.relation == Relation.DEPENDS_ON}
    in_domain = set((domain or {}).get("targets", [])) | set((domain or {}).get("origin", []))
    ops: list[ProposedOp] = []
    questions: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path, doc in sorted(code_paths.items()):
        src_path = _match(idx, [path])
        if src_path is None:
            continue
        _, consumer = idx[src_path]
        for module, line, text in _imports(doc):
            target_path = _match(idx, _module_paths(module))
            cite = Citation(source_id=doc.source_id, sha256=doc.sha256, line_start=line, line_end=line,
                            excerpt=text.strip())
            if target_path is None:
                local = _match({p: 1 for p in code_paths}, _module_paths(module))
                if local and local != path and (consumer, local) not in seen:
                    seen.add((consumer, local))
                    questions.append({"question": f"{consumer} imports {local}, which no registered component "
                                                  "realizes. Which component owns it, and is it a dependency?",
                                      "about": [consumer], "blocking": False,
                                      "source_evidence": [cite.to_dict()]})
                continue
            _, provider = idx[target_path]
            if provider == consumer or (consumer, provider) in registered or (consumer, provider) in seen:
                continue
            seen.add((consumer, provider))
            near = consumer in in_domain or provider in in_domain
            ops.append(ProposedOp(
                op_id=f"scan-{len(ops) + 1}", kind="add_edge",
                payload={"edge": {"src": consumer, "dst": provider, "relation": "depends_on"}},
                reason=(f"{doc.path} (realizes {consumer}) imports {module} from {target_path} (realizes {provider}); "
                        f"no depends_on edge is registered{' — touches the candidate influence domain' if near else ''}"),
                support="supported", evidence=[cite], change_class="structure"))
    return Proposal(proposal_id=proposal_id, base_graph_revision=graph.revision,
                    intent="Missing-dependency challenge (deterministic import scan)", operations=ops,
                    unresolved_questions=questions,
                    origin={"client": {"provider": "deterministic-import-scan", "is_fake": False},
                            "prompt": {"prompt_id": "none", "version": "scan-v1"}, "is_fake": False})


def challenge_impact(app: GraphADLC, change: dict[str, Any], bundle: SourceBundle, *,
                     scan_proposal_id: str, llm: LLMProposer | None = None,
                     llm_proposal_id: str | None = None, call_key: str | None = None) -> dict[str, Any]:
    """Return the domain and candidate proposals (unreviewed, uncommitted)."""
    g = app.graph()
    domain = influence_domain(app, change)
    scan = scan_code_dependencies(g, bundle, domain, scan_proposal_id)
    out: dict[str, Any] = {"domain": domain, "scan": scan, "llm": None, "llm_call": None}
    if llm is not None:
        prop, record = llm.challenge(g, bundle, change, domain, proposal_id=llm_proposal_id or "LLM-CHALLENGE",
                                     call_key=call_key or "challenge")
        out["llm"], out["llm_call"] = prop, record
    return out
