"""`gadlc` command line interface (argparse, stdlib only)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .context import render_text
from .core import GraphADLC
from .lifecycle import LifecycleError
from .validation import GraphError, RevisionConflict, ValidationError


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _dump(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=lambda o: o.to_dict()))


def _p(text: str = "") -> None:
    print(text)


# ---------------------------------------------------------------------------
# human-readable renderers
# ---------------------------------------------------------------------------

def _render_frontier(results: list[Any], rev: int) -> None:
    _p(f"Frontier @ graph rev {rev}")
    for r in results:
        line = f"  {r.requirement_id + '@r' + str(r.rev):<18} {r.readiness.upper():<12}"
        reasons = [f"{b.code}({', '.join(b.node_ids)})" if b.node_ids else b.code for b in r.blockers]
        reasons += [f"~{c.code}({', '.join(c.node_ids)})" for c in r.conditions]
        _p(line + ("  " + "; ".join(reasons) if reasons else ""))


def _render_impact(impact: dict[str, Any], indent: str = "  ") -> None:
    obs = impact["obligations"]
    _p(f"{indent}Impact: {len(obs)} candidate verification obligation(s)")
    for o in obs:
        flag = " [provisional]" if o["provisional"] else (" [low confidence]" if o["confidence"] == "low" else "")
        _p(f"{indent}  {o['target_id']:<16} {o['obligation']:<22} origin={o['origin_id']} "
           f"path={'>'.join(o['path'])} ref={o['change_ref']}{flag}")
        _p(f"{indent}    why: {o['reason']}")
    for pr in impact.get("pruned", []):
        _p(f"{indent}  PRUNED {pr['consumer_id']} via {pr['edge_id']}: {pr['reason']}")
    for n in impact.get("notes", []):
        _p(f"{indent}  note: {n}")


def _render_diff(diff: list[dict[str, Any]], indent: str = "  ") -> None:
    _p(f"{indent}Diff:")
    for d in diff:
        if d["op"] == "update_node":
            _p(f"{indent}  ~ {d['id']} r{d['before']['rev']} -> r{d['after']['rev']} aspects={d['aspects']}")
            for field, ch in d["fields"].items():
                _p(f"{indent}      {field}: {json.dumps(ch['from'])} -> {json.dumps(ch['to'])}")
        elif d["op"] in ("add_node", "add_edge"):
            _p(f"{indent}  + {d['op']} {d['id']}: {json.dumps(d['after'], sort_keys=True)}")
        else:
            _p(f"{indent}  - {d['op']} {d['id']}" + (
                f" (cascade: {d['cascaded_edge_removals']})" if d.get("cascaded_edge_removals") else ""))


def _render_invalidated(inv: dict[str, Any], indent: str = "  ") -> None:
    for w in inv["work_units"]:
        _p(f"{indent}  work {w['work_unit_id']} ({w['status']}) -> needs_revalidation: {'; '.join(w['reasons'])}")
    for e in inv["evidence"]:
        _p(f"{indent}  evidence {e['evidence_id']} on {e['node_id']} stale: {'; '.join(e['reasons'])}")
    if not inv["work_units"] and not inv["evidence"]:
        _p(f"{indent}  (no work units or evidence invalidated)")


def _render_actions(actions: list[Any]) -> None:
    for a in actions:
        _p(f"  {a.rank:>2}. [tier {a.tier}] {a.action:<17} {','.join(a.target_ids):<34} score={a.score:<8} {a.rationale}")


def _render_work(w: dict[str, Any]) -> None:
    _p(f"Work unit {w['id']} [{w['kind']}] targets={w['target_ids']} status={w['status']}")
    _p(f"  baseline graph rev {w['base_graph_revision']}; node revisions "
       + ", ".join(f"{k}@r{v}" for k, v in w["node_revisions"].items()))
    _p(f"  readiness at baseline: {w['readiness_at_baseline']}; code rev: {w.get('code_rev')}")
    for a in w["assumptions"]:
        _p(f"  assumption [{a['id']}] {a['statement']} (from {a['origin_node']}@r{a['origin_rev']}, "
           f"graph rev {a['origin_graph_revision']})")
    _p("  required checks: " + ", ".join(f"{c['kind']}:{c['node_id']}" for c in w["required_checks"]))
    ev = w.get("evaluation")
    if ev:
        for c in ev["satisfied_checks"]:
            _p(f"    satisfied {c['kind']}:{c['node_id']} by {c['evidence_id']}")
        for c in ev["missing_checks"]:
            _p(f"    MISSING  {c['kind']}:{c['node_id']} ({c['status']})")
        for m in ev["baseline_mismatches"]:
            _p(f"    baseline mismatch {m['node_id']}: r{m['baseline_rev']} -> r{m['current_rev']}")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_init(app: GraphADLC, a: argparse.Namespace) -> int:
    rev = app.seed_graph(_load_json(a.seed))
    if a.json:
        _dump({"graph_revision": rev, "frontier": app.frontier()})
    else:
        _p(f"Initialized {a.db} from {a.seed} at graph rev {rev}")
        _render_frontier(app.frontier(), rev)
    return 0


def cmd_status(app: GraphADLC, a: argparse.Namespace) -> int:
    st = app.status()
    if a.json:
        _dump(st)
        return 0
    _p(f"Graph rev {st['graph_revision']}: {st['nodes']} nodes, {st['edges']} edges "
       f"(tombstones: {st['tombstones']['nodes']} nodes, {st['tombstones']['edges']} edges)")
    _p("Leaf requirements (readiness | freshness):")
    for l in st["leaf_requirements"]:
        extra = l["blockers"] + [f"~{c}" for c in l["conditions"]]
        _p(f"  {l['requirement_id'] + '@r' + str(l['rev']):<18} {l['readiness']:<12} | {l['freshness']:<18} {' '.join(extra)}")
    _p(f"Open questions: {', '.join(st['open_questions']) or '(none)'}")
    _p("Work units: " + (", ".join(f"{w['id']}({','.join(w['targets'])}, {w['status']}, base r{w['base_graph_revision']})"
                                     for w in st["work_units"]) or "(none)"))
    _p("Changes: " + (", ".join(f"{c['id']}@rev{c['graph_rev']} by {c['actor']}" for c in st["changes"]) or "(none)"))
    return 0


def cmd_frontier(app: GraphADLC, a: argparse.Namespace) -> int:
    res = app.frontier()
    if a.json:
        _dump(res)
    else:
        _render_frontier(res, app.graph().revision)
    return 0


def cmd_context(app: GraphADLC, a: argparse.Namespace) -> int:
    pkt = app.context(a.requirement_id)
    if a.json:
        _dump(pkt)
    else:
        _p(render_text(pkt))
    return 0


def cmd_impact(app: GraphADLC, a: argparse.Namespace) -> int:
    rep = app.impact(a.node_id, a.aspect, evidence_refs=a.evidence or None)
    if a.json:
        _dump(rep.to_dict())
    else:
        _p(f"Hypothetical {a.aspect} change to {a.node_id} @ graph rev {app.graph().revision}")
        _render_impact(rep.to_dict())
    return 0


def cmd_propose(app: GraphADLC, a: argparse.Namespace) -> int:
    rep = app.propose(_load_json(a.delta))
    if a.json:
        _dump(rep)
        return 0 if rep["committable"] else 2
    _p(f"Proposal (read-only) by {rep['actor']}: base rev {rep['base_graph_revision']} -> proposed rev "
       f"{rep['proposed_graph_revision']} (current {rep['current_graph_revision']})")
    _p(f"  rationale: {rep['rationale']}")
    for c in rep["conflicts"]:
        _p(f"  CONFLICT: {c}")
    for e in rep["errors"]:
        _p(f"  ERROR: {e}")
    for o in rep["aspect_overrides"]:
        _p(f"  aspect override op[{o['op_index']}] {o['node_id']}: {o['declared']} -> {o['effective']} ({o['reason']})")
    _render_diff(rep["diff"])
    if rep["committable"]:
        _render_impact(rep["impact"])
        _p("  Readiness changes: " + (", ".join(f"{c['requirement_id']}: {c['from']} -> {c['to']}"
                                               for c in rep["readiness_changes"]) or "(none)"))
        _p("  Would invalidate:")
        _render_invalidated(rep["would_invalidate"])
        if rep["approvals_required"]:
            _p(f"  approvals required for aspects: {rep['approvals_required']}")
    _p(f"  committable: {rep['committable']}")
    return 0 if rep["committable"] else 2


def cmd_commit(app: GraphADLC, a: argparse.Namespace) -> int:
    res = app.commit(_load_json(a.delta), approvals=a.approve or None)
    if a.json:
        _dump(res)
        return 0
    _p(f"Committed {res['change_id']} -> graph rev {res['graph_revision']}")
    _render_diff(res["diff"])
    _render_impact(res["impact"])
    _p("  Readiness changes: " + (", ".join(f"{c['requirement_id']}: {c['from']} -> {c['to']}"
                                           for c in res["readiness_changes"]) or "(none)"))
    _p("  Invalidated:")
    _render_invalidated(res["invalidated"])
    return 0


def cmd_next(app: GraphADLC, a: argparse.Namespace) -> int:
    acts = app.suggest_actions(limit=a.limit)
    if a.json:
        _dump(acts)
    else:
        _p(f"Suggested actions @ graph rev {app.graph().revision} (ranked, deterministic policy):")
        _render_actions(acts)
    return 0


def cmd_export(app: GraphADLC, a: argparse.Namespace) -> int:
    text = app.export_json()
    if a.out and a.out != "-":
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        _p(f"Exported graph rev {app.graph().revision} to {a.out}")
    else:
        sys.stdout.write(text)
    return 0


def cmd_log(app: GraphADLC, a: argparse.Namespace) -> int:
    changes = app.log()
    if a.json:
        _dump(changes)
        return 0
    for c in changes:
        _p(f"{c['id']} graph rev {c['base_graph_rev']} -> {c['graph_rev']} by {c['actor']} at {c['created_at']}")
        _p(f"  rationale: {c['rationale']}")
        _render_diff(c["diff"])
        _render_impact(c["impact"])
    return 0


def _parse_evidence(items: list[str]) -> list[dict[str, Any]]:
    out = []
    for it in items or []:
        parts = it.split(":")
        if len(parts) == 1:
            node, kind, result = parts[0], "test_result", "passed"
        elif len(parts) == 2:
            node, kind, result = parts[0], parts[1], "passed"
        else:
            node, kind, result = parts[0], parts[1], parts[2]
        out.append({"node_id": node, "kind": kind, "result": result})
    return out


def cmd_work(app: GraphADLC, a: argparse.Namespace) -> int:
    if a.work_cmd == "create":
        w = app.create_work_unit(a.requirement_id, kind=a.kind)
    elif a.work_cmd == "start":
        w = app.start_work_unit(a.work_id)
    elif a.work_cmd == "submit":
        w = app.submit_work_unit(a.work_id, a.code_rev, _parse_evidence(a.evidence))
    elif a.work_cmd == "show":
        w = app.work_unit(a.work_id)
    else:  # list
        ws = app.work_units()
        if a.json:
            _dump(ws)
        else:
            for w in ws:
                _render_work(w)
        return 0
    if a.json:
        _dump(w)
    else:
        _render_work(w)
    return 0


def cmd_evidence(app: GraphADLC, a: argparse.Namespace) -> int:
    if a.evidence_cmd == "add":
        ev = app.record_evidence(a.node_id, a.kind, a.result, code_rev=a.code_rev)
        if a.json:
            _dump(ev)
        else:
            _p(f"Recorded {ev['id']}: {ev['kind']} on {ev['node_id']} = {ev['result']} against "
               + ", ".join(f"{k}@r{v}" for k, v in ev["spec_revs"].items()))
        return 0
    evs = app.evidence()
    if a.json:
        _dump(evs)
    else:
        for e in evs:
            state = "current" if e["current"] else "STALE (" + "; ".join(e["stale_reasons"]) + ")"
            _p(f"  {e['id']} {e['kind']:<22} {e['node_id']:<16} {e['result']:<7} work={e['work_unit_id']} "
               f"code={e['code_rev']} spec=" + ",".join(f"{k}@r{v}" for k, v in e["spec_revs"].items())
               + f"  {state}")
    return 0


def cmd_reconcile(app: GraphADLC, a: argparse.Namespace) -> int:
    rep = app.reconcile(a.work_id, rebase=a.rebase)
    if a.json:
        _dump(rep)
        return 0
    _p(f"Reconcile {rep['work_unit_id']}: {rep['status_before']} -> {rep['status_after']} "
       f"(baseline rev {rep['baseline_graph_revision']}, graph rev {rep['graph_revision']}, rebased={rep['rebased']})")
    for m in rep["baseline_mismatches"]:
        _p(f"  baseline mismatch {m['node_id']}: r{m['baseline_rev']} -> r{m['current_rev']}")
    for s in rep["stale_evidence"]:
        _p(f"  stale evidence {s['evidence_id']} ({s['kind']} on {s['node_id']}): {'; '.join(s['reasons'])}")
    for o in rep["obligations_since_baseline"]:
        _p(f"  obligation {o['change_id']}: {o['target_id']} {o['obligation']} path={'>'.join(o['path'])}")
    if rep.get("new_baseline"):
        nb = rep["new_baseline"]
        _p(f"  new baseline rev {nb['base_graph_revision']}: "
           + ", ".join(f"{k}@r{v}" for k, v in nb["node_revisions"].items()))
    _p("  next: " + ("; ".join(rep["actions"]) or "(nothing)"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gadlc", description="Graph-driven ADLC v0.1 prototype")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name: str, fn: Any, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--db", default=None, help="SQLite database path")
        sp.add_argument("--json", action="store_true", default=False, help="machine-readable output")
        sp.set_defaults(fn=fn)
        return sp

    def nested(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # allow --db/--json after the nested sub-command as well
        sp.add_argument("--db", default=argparse.SUPPRESS, help="SQLite database path")
        sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine-readable output")
        return sp

    add("init", cmd_init, "create database from seed snapshot").add_argument("--seed", required=True)
    add("status", cmd_status, "graph, readiness and freshness summary")
    add("frontier", cmd_frontier, "leaf requirement readiness")
    add("context", cmd_context, "scoped work context").add_argument("requirement_id")
    sp = add("impact", cmd_impact, "hypothetical impact of changing a node")
    sp.add_argument("node_id")
    sp.add_argument("--aspect", required=True, choices=["semantic", "contract", "implementation", "structure", "evidence"])
    sp.add_argument("--evidence", action="append", help="evidence id usable for pruning (repeatable)")
    add("propose", cmd_propose, "read-only preview of a delta").add_argument("--delta", required=True)
    sp = add("commit", cmd_commit, "atomically commit a delta")
    sp.add_argument("--delta", required=True)
    sp.add_argument("--approve", action="append", help="approver id (repeatable)")
    add("next", cmd_next, "ranked suggested actions").add_argument("--limit", type=int)
    add("export", cmd_export, "canonical JSON snapshot").add_argument("--out", default="-")
    add("log", cmd_log, "committed changes with diffs and impact")

    sp = add("work", cmd_work, "work units")
    wsub = sp.add_subparsers(dest="work_cmd", required=True)
    c = nested(wsub.add_parser("create"))
    c.add_argument("requirement_id")
    c.add_argument("--kind", default="implement")
    nested(wsub.add_parser("start")).add_argument("work_id")
    s = nested(wsub.add_parser("submit"))
    s.add_argument("work_id")
    s.add_argument("--code-rev", required=True)
    s.add_argument("--evidence", action="append", default=[],
                   help="NODE[:KIND[:RESULT]] e.g. TEST-RETRY:test_result:passed (repeatable)")
    nested(wsub.add_parser("show")).add_argument("work_id")
    nested(wsub.add_parser("list"))

    sp = add("evidence", cmd_evidence, "record or list evidence")
    esub = sp.add_subparsers(dest="evidence_cmd", required=True)
    e = nested(esub.add_parser("add"))
    e.add_argument("node_id")
    e.add_argument("--kind", required=True, help="e.g. test_result, contract_check, contract_preservation")
    e.add_argument("--result", default="passed", choices=["passed", "failed"])
    e.add_argument("--code-rev")
    nested(esub.add_parser("list"))

    sp = add("reconcile", cmd_reconcile, "reconcile a work unit with the current graph")
    sp.add_argument("work_id")
    sp.add_argument("--rebase", action="store_true", help="accept current revisions as new baseline")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.db:
        parser.error("--db is required")
    app = GraphADLC(args.db)
    try:
        return args.fn(app, args)
    except RevisionConflict as exc:
        print(f"REJECTED (revision conflict): {exc}", file=sys.stderr)
        return 3
    except ValidationError as exc:
        print("REJECTED (invalid): " + "; ".join(exc.issues), file=sys.stderr)
        return 4
    except LifecycleError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 5
    except (GraphError, KeyError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
