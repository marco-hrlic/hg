"""Replay the payment-retry vertical slice end-to-end (SPEC §6, Milestone 4).

Usage::

    python -m graph_adlc.demo [--workdir out] [--fixtures fixtures]

Prints a human-readable trace (the same output the CLI commands produce) and
writes ``<workdir>/trace.json``: a machine-readable record of graph revisions,
frontiers, changes with impact, suggested actions, work units and evidence after
every step, plus ``<workdir>/snapshot.json`` (final canonical export).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any

from . import cli
from .core import GraphADLC

DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def steps(db: str, fx: Path, out: Path) -> list[tuple[str, list[str]]]:
    f = lambda name: str(fx / name)  # noqa: E731
    return [
        ("A. Discovery: ingest partial seed", ["init", "--db", db, "--seed", f("payment_seed.json")]),
        ("A. Suggested next actions", ["next", "--db", db]),
        ("B. Refinement: preview (read-only)", ["propose", "--db", db, "--delta", f("retry_refinement.json")]),
        ("B. Refinement: commit", ["commit", "--db", db, "--delta", f("retry_refinement.json")]),
        ("B. Refined frontier", ["frontier", "--db", db]),
        ("C. Scoped context for REQ-RETRY", ["context", "--db", db, "REQ-RETRY"]),
        ("C. Create work unit (baseline captured)", ["work", "create", "REQ-RETRY", "--db", db]),
        ("C. Start work", ["work", "start", "WORK-1", "--db", db]),
        ("C. Submit code + evidence", ["work", "submit", "WORK-1", "--db", db, "--code-rev", "git:a1b2c3",
                                       "--evidence", "TEST-RETRY:test_result:passed"]),
        ("C. Status after implementation", ["status", "--db", db]),
        ("D. Contradiction: provider idempotency window (preview)",
         ["propose", "--db", db, "--delta", f("idempotency_window_change.json")]),
        ("E. Influence: commit contract change", ["commit", "--db", db, "--delta", f("idempotency_window_change.json")]),
        ("E. Evidence after contract change", ["evidence", "list", "--db", db]),
        ("E. Controller prioritizes reconciliation", ["next", "--db", db]),
        ("E. Attempt to re-submit stale work (must be refused)",
         ["work", "submit", "WORK-1", "--db", db, "--code-rev", "git:a1b2c3", "--evidence", "TEST-RETRY"]),
        ("F. Reconciliation: inspect", ["reconcile", "WORK-1", "--db", db]),
        ("F. Update spec to match new contract", ["commit", "--db", db, "--delta", f("retry_window_reconciliation.json")]),
        ("F. Rebase work unit onto current revisions", ["reconcile", "WORK-1", "--db", db, "--rebase"]),
        ("F. Submit updated code with fresh test + contract evidence",
         ["work", "submit", "WORK-1", "--db", db, "--code-rev", "git:d4e5f6",
          "--evidence", "TEST-RETRY:test_result:passed", "--evidence", "CON-IDEMPOTENCY:contract_check:passed"]),
        ("F. Final status (unaffected work still available)", ["status", "--db", db]),
        ("F. Final suggested actions", ["next", "--db", db]),
        ("Export canonical snapshot", ["export", "--db", db, "--out", str(out / "snapshot.json")]),
    ]


def _state(db: str) -> dict[str, Any]:
    app = GraphADLC(db)
    try:
        changes = app.log()
        return {
            "graph_revision": app.graph().revision,
            "frontier": [{"requirement_id": r.requirement_id, "rev": r.rev, "readiness": r.readiness}
                         for r in app.frontier()],
            "freshness": {r.requirement_id: app.freshness(r.requirement_id) for r in app.frontier()},
            "latest_change": ({"id": changes[-1]["id"], "graph_rev": changes[-1]["graph_rev"],
                               "obligations": changes[-1]["impact"]["obligations"],
                               "invalidated": changes[-1]["impact"]["invalidated"]} if changes else None),
            "work_units": [{"id": w["id"], "status": w["status"], "base_graph_revision": w["base_graph_revision"],
                            "node_revisions": w["node_revisions"]} for w in app.work_units()],
            "evidence": [{"id": e["id"], "node_id": e["node_id"], "kind": e["kind"], "result": e["result"],
                          "spec_revs": e["spec_revs"], "code_rev": e["code_rev"], "current": e["current"]}
                         for e in app.evidence()],
            "suggested_actions": [a.to_dict() for a in app.suggest_actions(limit=5)],
        }
    finally:
        app.close()


def run(workdir: str, fixtures: str | None = None, echo: bool = True) -> dict[str, Any]:
    out = Path(workdir)
    out.mkdir(parents=True, exist_ok=True)
    db = str(out / "demo.sqlite")
    if os.path.exists(db):
        os.remove(db)
    fx = Path(fixtures) if fixtures else DEFAULT_FIXTURES
    trace: list[dict[str, Any]] = []
    for i, (title, argv) in enumerate(steps(db, fx, out), start=1):
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            code = cli.main(argv)
        text = buf_out.getvalue() + buf_err.getvalue()
        if echo:
            print(f"\n=== Step {i}: {title}\n$ gadlc {' '.join(argv)}")
            print(text.rstrip())
        trace.append({"step": i, "title": title, "command": ["gadlc", *argv], "exit_code": code,
                      "output": text, "state": _state(db)})
    doc = {"scenario": "payment-retry vertical slice", "steps": trace}
    with open(out / "trace.json", "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")
    if echo:
        print(f"\nTrace written to {out / 'trace.json'}")
    return doc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workdir", default="out")
    p.add_argument("--fixtures", default=None)
    a = p.parse_args(argv)
    run(a.workdir, a.fixtures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
