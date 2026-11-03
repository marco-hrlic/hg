"""Longitudinal replay of project histories through one persistent graph per arm.

Arms (KICKOFF §M4):
  D   model reads sources directly and lists impacts (live provider only)
  G0  graph after the initial construction only; no later maintenance
  G1  model proposes/maintains the graph; deterministic influence
  G2  G1 + missing-edge challenge (deterministic import scan, + model challenger when live)
  H   hand-curated reference graph (diagnostic upper bound, not a fair baseline)

Prediction never reads ``oracle/``. Reviewer decisions come from a deterministic
policy (``policy_reviewer``), which is *not* a human; reports say so.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from graph_adlc.core import GraphADLC
from graph_adlc.lifecycle import LifecycleError
from graph_adlc.llm_graph import (DirectImpactAnalyst, LLMProposer, MaintenanceLoop, ScriptedClient, SourceBundle,
                                  SourceCatalog, SourceDocument, assess_preservation_evidence, export_spec_markdown,
                                  git_diff, influence_domain, load_file_source, load_git_sources, policy_reviewer)
from graph_adlc.llm_graph.model_adapter import redact_record
from graph_adlc.llm_graph.source import normalize_ws  # noqa: F401
from graph_adlc.llm_graph.validation import _locate

HERE = Path(__file__).resolve().parent
FIXED_TIME = "2026-11-03T00:00:00Z"
EMPTY_GRAPH = {"graph_revision": 1, "nodes": [], "edges": []}


# ---------------------------------------------------------------------------
# Fixture repositories (deterministic SHAs)
# ---------------------------------------------------------------------------

def build_repo(project_dir: Path, hist: dict[str, Any], workdir: Path) -> tuple[Path, dict[str, str]]:
    repo = workdir / hist["repo"]["name"]
    repo.mkdir(parents=True)
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "Fixture Author", "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
           "GIT_COMMITTER_NAME": "Fixture Author", "GIT_COMMITTER_EMAIL": "fixture@example.invalid"}

    def git(*args: str, **kw: Any) -> str:
        return subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args], check=True,
                              capture_output=True, text=True, env={**env, **kw}).stdout

    subprocess.run(["git", "-c", "init.defaultBranch=main", "init", "-q", str(repo)], check=True, env=env,
                   capture_output=True)
    shas = {}
    for c in hist["repo"]["commits"]:
        for p in repo.iterdir():
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
        src = project_dir / c["dir"]
        for f in sorted(src.rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                dst = repo / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
        git("add", "-A")
        git("commit", "-q", "-m", c["message"], GIT_AUTHOR_DATE=c["date"], GIT_COMMITTER_DATE=c["date"])
        shas[c["id"]] = git("rev-parse", "HEAD").strip()
    return repo, shas


# ---------------------------------------------------------------------------
# Source state across events
# ---------------------------------------------------------------------------

class SourceState:
    def __init__(self, project: str, project_dir: Path, repo: Path, repo_name: str, shas: dict[str, str]):
        self.project, self.dir, self.repo, self.repo_name, self.shas = project, project_dir, repo, repo_name, shas
        self.docs: dict[str, SourceDocument] = {}
        self.commit: str | None = None

    def advance(self, ev: dict[str, Any]) -> SourceBundle:
        for s in ev.get("sources", []):
            d = load_file_source(self.dir, self.project, s["path"], s["type"], s["version"], s.get("logical_path"))
            self.docs[d.path] = d
        if ev.get("code"):
            self.commit = ev["code"]
        docs = [self.docs[k] for k in sorted(self.docs)]
        if self.commit:
            docs += load_git_sources(self.repo, self.repo_name, self.project, self.shas[self.commit])
        diffs = []
        if ev.get("diff"):
            a, b = ev["diff"]
            diffs.append(git_diff(self.repo, self.repo_name, self.shas[a], self.shas[b]))
        return SourceBundle(project=self.project, documents=docs, diffs=diffs)


def resolve_placeholders(obj: Any, bundle: SourceBundle) -> Any:
    """Scripted-fake convenience: turn '@path' / '@git:path' / '@diff' / '@current' / '@auto' into concrete values."""
    if isinstance(obj, list):
        return [resolve_placeholders(x, bundle) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {k: resolve_placeholders(v, bundle) for k, v in obj.items()}
    sid = out.get("source_id")
    if isinstance(sid, str) and sid.startswith("@"):
        key = sid[1:]
        doc = None
        if key == "diff":
            out["source_id"] = bundle.diffs[0]["source_id"] if bundle.diffs else "missing-diff"
            out.pop("sha256", None)
            out.pop("lines", None)
            return out
        if key.startswith("git:"):
            doc = next((d for d in bundle.documents if d.origin == "git" and d.path == key[4:]), None)
        else:
            doc = next((d for d in bundle.documents if d.origin == "file" and d.path == key), None)
        if doc is None:
            out["source_id"] = f"{bundle.project}:{key}"
            return out
        out["source_id"] = doc.source_id
        if out.get("sha256") == "@current":
            out["sha256"] = doc.sha256
        if out.get("lines") == "@auto":
            ex = normalize_ws(out.get("excerpt", ""))
            span = _locate(doc.lines, ex) if ex else None
            loc = list(span) if span else None
            if loc:
                out["lines"] = loc
            else:
                out.pop("lines")
    return out


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def probe(app: GraphADLC, node_id: str, aspect: str, evidence_refs: list[str] | None = None) -> dict[str, Any]:
    g = app.graph()
    if g.node(node_id) is None:
        return {"node_id": node_id, "aspect": aspect, "present": False, "targets": [], "pairs": [], "pruned": []}
    rep = app.impact(node_id, aspect, evidence_refs=evidence_refs)
    pairs = sorted({(r.target_id, r.obligation) for r in rep.records})
    return {"node_id": node_id, "aspect": aspect, "present": True, "graph_revision": g.revision,
            "targets": sorted({t for t, _ in pairs}), "pairs": [list(p) for p in pairs],
            "provisional": sorted({r.target_id for r in rep.records if r.provisional}),
            "pruned": [p.to_dict() for p in rep.pruned]}


def changed_components(app: GraphADLC, bundle: SourceBundle) -> list[str]:
    g = app.graph()
    paths = set()
    for d in bundle.diffs:
        paths |= set(re.findall(r"^\+\+\+ b/(\S+)", d["text"], flags=re.M))
    comps = set()
    for e in g.live_edges():
        if e.relation == "realized_by" and (a := g.node(e.dst)) is not None and a.data.get("path") in paths:
            comps.add(e.src)
    return sorted(comps)


# ---------------------------------------------------------------------------
# One arm over one project
# ---------------------------------------------------------------------------

def _summarize_validation(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {"graph_revision": report["graph_revision"], "base_graph_revision": report["base_graph_revision"],
            "stale_base": report["stale_base"], "summary": report["summary"],
            "ops": {k: {"verdict": v["verdict"], "effective_support": v["effective_support"],
                        "findings": [f"{f['severity']}:{f['code']}" for f in v["findings"]],
                        "citations": [f"{c['status']}:{c['source_id']}:{c.get('resolved_lines')}" for c in v["citations"]]}
                    for k, v in report["ops"].items()},
            "engine_preview": {k: report["engine_preview"].get(k) for k in ("ran", "committable", "required_actions",
                                                                            "readiness_changes", "errors")}}


def _run_step(loop: MaintenanceLoop, step: Any, bundle: SourceBundle, reviewer: Callable) -> dict[str, Any]:
    out: dict[str, Any] = {"errors": step.errors}
    if step.call is not None:
        out["call"] = redact_record(step.call)
    if step.proposal is None:
        return out
    out["proposal"] = step.proposal.to_dict()
    out["validation"] = _summarize_validation(step.report)
    out["review_text"] = step.review_text
    decisions = reviewer(step.proposal, step.report)
    out["decisions"] = [{"op_id": d.op_id, "decision": d.decision, "reviewer": d.reviewer, "rationale": d.rationale}
                        for d in decisions]
    res = loop.review(step, decisions, bundle)
    out["outcome"] = {k: res[k] for k in ("change_id", "rebased", "accepted", "rejected", "deferred", "no_change_records")}
    if res["commit"]:
        out["outcome"]["required_actions"] = res["commit"]["required_actions"]
        out["outcome"]["readiness_changes"] = res["commit"]["readiness_changes"]
        out["outcome"]["invalidated"] = res["commit"]["invalidated"]
        out["outcome"]["obligations"] = len(res["commit"]["impact"]["obligations"])
    return out


def _lifecycle(app: GraphADLC, actions: list[dict[str, Any]], labels: dict[str, str], shas: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    for a in actions:
        try:
            if a["action"] == "work_create":
                w = app.create_work_unit(a["requirement"])
                app.start_work_unit(w["id"])
                labels[a["label"]] = w["id"]
                out.append({**a, "work_id": w["id"], "status": "active"})
            elif a["action"] == "work_submit":
                wid = labels.get(a["work"])
                if wid is None:
                    out.append({**a, "skipped": "work unit was not created in this arm"})
                    continue
                ev = [{"node_id": n, "kind": k, "result": r} for n, k, r in a["evidence"]]
                res = app.submit_work_unit(wid, f"git:{shas[a['code']][:12]}", ev)
                out.append({**a, "work_id": wid, "status": res["status"],
                            "missing_checks": res["evaluation"]["missing_checks"]})
            elif a["action"] == "work_rebase":
                wid = labels.get(a["work"])
                if wid is None:
                    out.append({**a, "skipped": "work unit was not created in this arm"})
                    continue
                if app.work_unit(wid)["status"] != "needs_revalidation":
                    out.append({**a, "work_id": wid, "skipped": "not invalidated; nothing to rebase"})
                    continue
                r = app.reconcile(wid, rebase=True)
                out.append({**a, "work_id": wid, "status_before": r["status_before"], "status_after": r["status_after"],
                            "rebased": r["rebased"], "baseline_mismatches": r["baseline_mismatches"]})
            elif a["action"] == "evidence":
                ev = app.record_evidence(a["node"], a["kind"], a["result"], code_rev=f"git:{shas[a['code']][:12]}",
                                         details=a.get("details"))
                out.append({**a, "evidence_id": ev["id"], "spec_revs": ev["spec_revs"]})
            elif a["action"] == "reconcile_all":
                for w in app.work_units():
                    r = app.reconcile(w["id"])
                    out.append({"action": "reconcile", "work_id": w["id"], "targets": w["target_ids"],
                                "status": app.work_unit(w["id"])["status"], "can_verify": r.get("can_verify"),
                                "baseline_mismatches": r.get("baseline_mismatches"),
                                "stale_evidence": [s["evidence_id"] for s in r.get("stale_evidence", [])],
                                "obligations_since_baseline": sorted({(o["target_id"], o["obligation"])
                                                                      for o in r.get("obligations_since_baseline", [])})})
        except (LifecycleError, KeyError) as e:
            out.append({**a, "error": str(e)})
    return out


def run_arm(project: str, arm: str, client_factory: Callable[[str, Callable[[], SourceBundle]], Any] | None,
            workdir: Path, outdir: Path, live: bool, influence_rules: str = "v1") -> dict[str, Any]:
    pdir = HERE / "sources" / project
    hist = json.loads((HERE / "histories" / f"{project}.json").read_text())
    repo, shas = build_repo(pdir, hist, workdir / arm)
    app = GraphADLC(str(workdir / arm / "graph.sqlite"), influence_rules=influence_rules)
    pr = hist["probe"]
    trace: dict[str, Any] = {"project": project, "arm": arm, "commits": shas, "probe": pr, "events": []}
    if influence_rules != "v1":  # absent for v1, so v1 outputs stay byte-identical
        trace["influence_rules"] = influence_rules
    if arm == "H":
        app.seed_graph(json.loads((HERE / "reference" / f"{project}.json").read_text()))
        trace["prediction"] = {"E5": probe(app, pr["node_id"], pr["aspect"])}
        neg = next(e["negative_probe"] for e in hist["events"] if e.get("negative_probe"))
        trace["negative"] = {"E4": {"engine_semantic_probe": probe(app, neg["node_id"], neg["aspect"]),
                                    "note": "no maintenance loop: the engine has no editorial aspect"}}
        app.close()
        return trace
    app.seed_graph(EMPTY_GRAPH)
    state = SourceState(project, pdir, repo, hist["repo"]["name"], shas)
    current: dict[str, SourceBundle] = {}
    client = client_factory(project, lambda: current["bundle"])
    loop = MaintenanceLoop(app, LLMProposer(client))
    reviewer = policy_reviewer()
    labels: dict[str, str] = {}
    outdir.mkdir(parents=True, exist_ok=True)
    for ev in hist["events"]:
        bundle = state.advance(ev)
        current["bundle"] = bundle
        rec: dict[str, Any] = {"event": ev["id"], "kind": ev["kind"], "graph_revision_before": app.graph().revision,
                               "sources": bundle.manifest(), "diffs": [d["source_id"] for d in bundle.diffs]}
        if arm == "G0" and ev["id"] != "E1":
            rec["skipped"] = "G0: no maintenance after initial construction"
            if ev["id"] == pr["event"]:
                trace["prediction"] = {"E5": probe(app, pr["node_id"], pr["aspect"])}
            if ev.get("negative_probe"):
                trace["negative"] = {ev["id"]: {"engine_semantic_probe": probe(app, ev["negative_probe"]["node_id"],
                                                                               ev["negative_probe"]["aspect"])}}
            trace["events"].append(rec)
            continue
        rec["probe_before"] = probe(app, pr["node_id"], pr["aspect"])
        rec["lifecycle_before"] = _lifecycle(app, ev.get("lifecycle_before", []), labels, shas)
        if arm == "G2" and ev.get("challenge"):
            comps = changed_components(app, bundle) if bundle.diffs else []
            change = {"node_id": pr["node_id"], "aspect": pr["aspect"]} if not comps else \
                {"node_id": comps[0], "aspect": "implementation"}
            ch = loop.challenge(change, bundle, use_llm=live, call_key=f"{project}/{ev['id']}/challenge")
            rec["challenge"] = {"change": change, "changed_components": comps, "domain": ch["domain"], "steps": {}}
            for name, st in ch["steps"].items():
                if st is not None:
                    rec["challenge"]["steps"][name] = _run_step(loop, st, bundle, reviewer)
        if ev["id"] == pr["event"]:
            # The scored prediction: impact of the intended change on the graph as maintained so far.
            trace["prediction"] = {"E5": probe(app, pr["node_id"], pr["aspect"])}
            if pr.get("affected_domain"):
                gate = assess_preservation_evidence(app, pr["node_id"], pr["affected_domain"])
                all_ev = [e["id"] for e in app.store.list_evidence() if e["node_id"] == pr["node_id"]]
                trace["prediction"]["exploratory_pruning"] = {
                    "evidence_gate": gate,
                    "engine_pruning_with_all_evidence": probe(app, pr["node_id"], pr["aspect"], evidence_refs=all_ev),
                    "engine_pruning_with_gate_admitted_evidence": probe(
                        app, pr["node_id"], pr["aspect"], evidence_refs=[a["evidence_id"] for a in gate["admissible"]]),
                }
        step = loop.propose(bundle, ev["change_event"], call_key=f"{project}/{ev['id']}/propose")
        rec["proposal_step"] = _run_step(loop, step, bundle, reviewer)
        if ev.get("negative_probe"):
            o = rec["proposal_step"].get("outcome") or {}
            trace.setdefault("negative", {})[ev["id"]] = {
                "committed_obligations": o.get("obligations", 0), "required_actions": o.get("required_actions", {}),
                "deferred_ops": o.get("deferred", []), "no_change_records": o.get("no_change_records", [])}
        rec["probe_after"] = probe(app, pr["node_id"], pr["aspect"])
        rec["lifecycle_after"] = _lifecycle(app, ev.get("lifecycle_after", []), labels, shas)
        rec["graph_revision_after"] = app.graph().revision
        spec = export_spec_markdown(app.graph(), loop.ledger, title=f"{project} specification (after {ev['id']}, arm {arm})")
        (outdir / f"spec_{ev['id']}.md").write_text(spec)
        rec["spec_projection"] = f"spec_{ev['id']}.md"
        trace["events"].append(rec)
    trace["final_graph"] = app.export()
    trace["ledger"] = {"proposals": [{"id": p["id"], "status": p["status"], "change_id": p["change_id"]}
                                     for p in loop.ledger.proposals()],
                       "decisions": [{k: d[k] for k in ("proposal_id", "op_id", "decision", "reviewer", "change_id")}
                                     for d in loop.ledger.decisions()],
                       "calls": len(loop.ledger.calls())}
    trace["change_log"] = [{"id": c["id"], "graph_rev": c["graph_rev"], "actor": c["actor"], "rationale": c["rationale"]}
                           for c in app.log()]
    trace["client"] = client.describe()
    app.close()
    return trace


def run_direct(project: str, client: Any, workdir: Path) -> dict[str, Any]:
    """Arm D (live only): the model reads the E5 bundle and the intended change; no graph."""
    pdir = HERE / "sources" / project
    hist = json.loads((HERE / "histories" / f"{project}.json").read_text())
    repo, shas = build_repo(pdir, hist, workdir / "D")
    state = SourceState(project, pdir, repo, hist["repo"]["name"], shas)
    bundle = None
    for ev in hist["events"]:
        bundle = state.advance(ev)
        if ev["id"] == hist["probe"]["event"]:
            break
    analyst = DirectImpactAnalyst(client)
    out, record = analyst.analyze(bundle, {"description": hist["probe"]["description"]},
                                  call_key=f"{project}/D/direct_impact")
    targets = sorted({i.get("target") for i in (out or {}).get("impacts", []) if i.get("target")})
    return {"project": project, "arm": "D", "prediction": {"E5": {"targets": targets, "raw": out}},
            "call": redact_record(record), "client": client.describe()}


def scripted_client_factory(project: str, bundle_fn: Callable[[], SourceBundle]) -> ScriptedClient:
    data = json.loads((HERE / "scripted" / f"{project}.json").read_text())

    def respond(call_key: str, _user: str) -> Any:
        if call_key not in data["responses"]:
            # The fake has nothing to say for this call (e.g. a challenge): an empty, valid proposal.
            return {"intent": "(scripted fake: no response recorded)", "operations": []}
        return resolve_placeholders(copy.deepcopy(data["responses"][call_key]), bundle_fn())

    return ScriptedClient(respond, label="scripted fixture responses")


def tempdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="gadlc-v03-")


# ---------------------------------------------------------------------------
# Replaying recorded live model outputs (for before/after comparisons of engine rules)
# ---------------------------------------------------------------------------

def recorded_calls(trace: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """call_key -> call records (in trace order) found anywhere in a live trace."""
    out: dict[str, list[dict[str, Any]]] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            c = x.get("call")
            if isinstance(c, dict) and "call_key" in c and "output" in c:
                out.setdefault(c["call_key"], []).append(c)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(trace)
    return out


class ReplayClient:
    """Serves a recorded live response when the prompt is byte-identical (same call key, same input sha256);
    otherwise it calls the live ``fallback`` client. ``describe()`` reports how the latest call was served."""

    is_fake = False

    def __init__(self, records: dict[str, list[dict[str, Any]]], fallback: Any, source: str, strict: bool = False):
        self.records, self.fallback, self.source, self.strict = records, fallback, source, strict
        self.provider = getattr(fallback, "provider", "replay")
        self.api_key_env = getattr(fallback, "api_key_env", None)
        self.used: set[int] = set()
        self.served: list[dict[str, Any]] = []
        self._last: str | None = None

    def available(self) -> bool:
        return self.fallback.available()

    def describe(self) -> dict[str, Any]:
        return {**self.fallback.describe(), "replay_from": self.source, "served": self._last}

    def complete(self, system: str, user: str, *, call_key: str, max_tokens: int = 8000):
        from graph_adlc.llm_graph.model_adapter import ModelResponse
        from graph_adlc.llm_graph.source import sha256_text
        sha = sha256_text(system + "\n" + user)
        for rec in self.records.get(call_key, []):
            if id(rec) not in self.used and rec["input"]["sha256"] == sha:
                self.used.add(id(rec))
                self._last = "replayed"
                self.served.append({"call_key": call_key, "served": "replayed"})
                return ModelResponse(text=rec["output"]["text"], model=rec.get("model_reported") or "",
                                     provider=self.provider, is_fake=False, usage=rec.get("usage") or {},
                                     latency_ms=rec.get("latency_ms"), stop_reason=rec.get("stop_reason"))
        if self.strict:
            raise RuntimeError(f"replay-strict: no recorded call matches {call_key!r} (prompt changed or not recorded)")
        self._last = "live"
        self.served.append({"call_key": call_key, "served": "live",
                            "reason": "prompt differs from the recorded call" if call_key in self.records
                            else "no recorded call with this key"})
        return self.fallback.complete(system, user, call_key=call_key, max_tokens=max_tokens)
