"""Small fixtures for the v0.3 acceptance tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graph_adlc.core import GraphADLC
from graph_adlc.llm_graph import (LLMProposer, MaintenanceLoop, ScriptedClient, SourceBundle, SourceDocument,
                                  load_file_source)

HERE = Path(__file__).resolve().parents[1]
TS = HERE / "sources" / "toolshare"

REQS_V1 = load_file_source(TS, "toolshare", "spec/requirements.v1.md", "requirements", "draft-1", "spec/requirements.md")
REQS_V2 = load_file_source(TS, "toolshare", "spec/requirements.v2.md", "requirements", "draft-2", "spec/requirements.md")
ARCH = load_file_source(TS, "toolshare", "spec/architecture.md", "architecture", "2026-09-02")


def cite(doc: SourceDocument, excerpt: str, sha: str | None = None, lines: list[int] | None = None) -> dict[str, Any]:
    if lines is None:
        for i, l in enumerate(doc.lines, 1):
            if excerpt in l:
                lines = [i, i]
                break
    return {"source_id": doc.source_id, "sha256": sha or doc.sha256, "lines": lines, "excerpt": excerpt}


def new_app(tmp_path: Path, name: str = "g.sqlite") -> GraphADLC:
    app = GraphADLC(str(tmp_path / name))
    app.seed_graph({"graph_revision": 1, "nodes": [], "edges": []})
    return app


def loop_with(app: GraphADLC, responses: dict[str, Any]) -> MaintenanceLoop:
    return MaintenanceLoop(app, LLMProposer(ScriptedClient(responses, label="unit-test")))


def op_node(op_id: str, nid: str, kind: str, title: str, data: dict, ev: list, support: str = "supported", body: str = "") -> dict:
    return {"op_id": op_id, "kind": "add_node", "node": {"id": nid, "kind": kind, "title": title, "body": body, "data": data},
            "reason": "unit", "support": support, "source_evidence": ev}


def op_edge(op_id: str, src: str, rel: str, dst: str, ev: list, support: str = "supported") -> dict:
    return {"op_id": op_id, "kind": "add_edge", "edge": {"src": src, "dst": dst, "relation": rel}, "reason": "unit",
            "support": support, "source_evidence": ev}


def seed_response(doc: SourceDocument = REQS_V1) -> dict:
    return {"intent": "seed", "operations": [
        op_node("op-1", "CAP-HOLDS", "capability", "Holds", {},
                [cite(doc, "CAP-HOLDS — Members can queue for tools that are not currently on the shelf.")]),
        op_node("op-2", "REQ-PLACE-HOLD", "requirement", "Place a hold",
                {"acceptance_criteria": ["Holds on the same tool are served first come, first served."]},
                [cite(doc, "REQ-PLACE-HOLD — A member can place a hold on a catalogued tool that is on loan.")]),
        op_node("op-3", "REQ-HOLD-EXPIRY", "requirement", "Hold expiry", {"acceptance_criteria": []},
                [cite(doc, "REQ-HOLD-EXPIRY — When a held tool becomes available, the member has a pickup window to collect it. If they do not")]),
        op_edge("op-4", "CAP-HOLDS", "refines", "REQ-PLACE-HOLD", [cite(doc, "REQ-PLACE-HOLD — A member can place a hold on a catalogued tool that is on loan.")]),
        op_edge("op-5", "CAP-HOLDS", "refines", "REQ-HOLD-EXPIRY", [cite(doc, "REQ-HOLD-EXPIRY — When a held tool becomes available, the member has a pickup window to collect it. If they do not")]),
    ]}


def bundle(*docs: SourceDocument, diffs: list | None = None) -> SourceBundle:
    return SourceBundle(project="toolshare", documents=list(docs), diffs=diffs or [])


def accept_all(step, reviewer: str = "alice", ack: bool = False):
    from graph_adlc.llm_graph import Decision
    return [Decision(o.op_id, "accept", reviewer, "test", ack) for o in step.proposal.operations]


def snapshot(app: GraphADLC) -> str:
    return json.dumps(app.export(), sort_keys=True)
