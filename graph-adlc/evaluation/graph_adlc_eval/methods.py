"""Method adapters A/A2 (document), B (graph) and C (graph + evidence-gated pruning).

Every adapter runs in its own freshly seeded temporary database (``dataset.environment``)
and must leave it unchanged: ``propose`` is read-only by contract, and we verify that by
comparing the canonical export and the change/evidence/event tables before and after.
"""

from __future__ import annotations

import copy
import json
import time
from typing import Any

from graph_adlc import GraphADLC

from .dataset import Environment, Scenario, environment
from .document import document_baseline, render_change_request, render_spec


class MutationError(RuntimeError):
    pass


def _db_fingerprint(app: GraphADLC) -> str:
    st = app.store
    return json.dumps({"export": app.export(), "changes": st.list_changes(), "evidence": st.list_evidence(),
                       "events": len(st.list_events()), "work": st.list_work_units()}, sort_keys=True)


def render_inputs(s: Scenario, env: Environment) -> tuple[str, str]:
    g = env.app.graph()
    doc = render_spec(g, env.app.store.list_evidence(), f"{s.id} specification snapshot")
    return doc, render_change_request(s.proposal)


def run_document(s: Scenario, method: str, cfg: dict[str, Any], table: dict[str, Any]) -> dict[str, Any]:
    with environment(s) as env:
        before = _db_fingerprint(env.app)
        doc, cr = render_inputs(s, env)
        t0 = time.perf_counter()
        preds, trace = document_baseline(doc, s.proposal, cfg, table, method)
        elapsed = time.perf_counter() - t0
        if _db_fingerprint(env.app) != before:
            raise MutationError(f"{s.id}/{method}: database changed")
        graph = env.app.graph()
    trace["change_request_chars"] = len(cr)
    return {"method": method, "predictions": preds, "raw": {"trace": trace, "setup": env.setup_log},
            "elapsed_s": elapsed, "graph": graph, "document": doc, "change_request": cr,
            "available_evidence": env.evidence_ids}


def run_graph(s: Scenario, method: str) -> dict[str, Any]:
    with environment(s) as env:
        before = _db_fingerprint(env.app)
        refs = list(env.evidence_ids) if method == "C" else []
        delta = copy.deepcopy(s.proposal)
        if refs:
            delta["evidence_refs"] = refs
        else:
            delta.pop("evidence_refs", None)
        t0 = time.perf_counter()
        out = env.app.propose(delta)
        elapsed = time.perf_counter() - t0
        if _db_fingerprint(env.app) != before:
            raise MutationError(f"{s.id}/{method}: database changed")
        graph = env.app.graph()
    if not out["committable"]:
        raise RuntimeError(f"{s.id}/{method}: proposal not committable: {out['errors'] or out['conflicts']}")
    preds = [{"method": method, "target_id": o["target_id"], "obligation": o["obligation"],
              "explanation": o["reason"], "path_edge_ids": list(o["path"]), "pruned": False,
              "provisional": bool(o["provisional"]), "confidence": o["confidence"], "origin_id": o["origin_id"]}
             for o in out["impact"]["obligations"]]
    raw = {"evidence_refs_passed": refs, "impact": out["impact"], "readiness_changes": out["readiness_changes"],
           "would_invalidate": out["would_invalidate"], "aspect_overrides": out["aspect_overrides"],
           "setup": env.setup_log}
    return {"method": method, "predictions": preds, "raw": raw, "elapsed_s": elapsed, "graph": graph,
            "available_evidence": env.evidence_ids}
