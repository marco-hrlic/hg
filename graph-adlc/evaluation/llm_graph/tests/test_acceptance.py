"""v0.3 acceptance tests L01-L15 (KICKOFF §6)."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from helpers import (ARCH, HERE, REQS_V1, REQS_V2, accept_all, bundle, cite, loop_with, new_app, op_edge, op_node,
                     seed_response, snapshot)

from graph_adlc.llm_graph import (Decision, Ledger, MaintenanceLoop, ReviewError, SourceBundle, SourceCatalog,
                                  assess_preservation_evidence, export_spec_markdown, scan_code_dependencies)
from graph_adlc.llm_graph.proposal import parse_proposal
from graph_adlc.llm_graph.validation import validate_proposal

ROOT = HERE.parents[1]


def _seeded(tmp_path):
    app = new_app(tmp_path)
    loop = loop_with(app, {"seed": seed_response()})
    b = bundle(REQS_V1)
    st = loop.propose(b, {"kind": "initial"}, call_key="seed")
    res = loop.review(st, accept_all(st), b)
    return app, loop, b, res


# L01 ------------------------------------------------------------------------------------------
def test_L01_initial_graph_from_independent_source_commits_through_engine(tmp_path):
    app, loop, b, res = _seeded(tmp_path)
    assert REQS_V1.source_type == "requirements" and REQS_V1.origin == "file"  # hand-written file, not rendered
    assert res["change_id"] == "CHG-2" and app.graph().revision == 2
    log = app.log()
    assert len(log) == 1 and log[0]["actor"].startswith("llm-graph:P-0001 reviewed-by:alice")
    assert {n.id for n in app.graph().live_nodes()} == {"CAP-HOLDS", "REQ-PLACE-HOLD", "REQ-HOLD-EXPIRY"}
    # Normal revision checks apply: a second proposal built on the old base cannot silently overwrite.
    stale = loop_with(app, {"k": {"intent": "x", "operations": [
        {"op_id": "op-1", "kind": "update_node", "id": "REQ-PLACE-HOLD", "expected_rev": 1, "aspect": "semantic",
         "patch": {"body": "changed"}, "support": "supported",
         "source_evidence": [cite(REQS_V1, "REQ-PLACE-HOLD — A member can place a hold on a catalogued tool that is on loan.")]}]}})
    st = stale.propose(b, {}, call_key="k")
    # Someone else commits first (bumps REQ-PLACE-HOLD to r2) ...
    app.commit({"base_graph_revision": 2, "actor": "human", "rationale": "edit", "operations": [
        {"op": "update_node", "id": "REQ-PLACE-HOLD", "expected_rev": 1, "aspect": "semantic", "patch": {"body": "human edit"}}]})
    from graph_adlc.llm_graph import NeedsRegeneration
    with pytest.raises(NeedsRegeneration):
        stale.review(st, accept_all(st), b)
    assert app.get_node("REQ-PLACE-HOLD").body == "human edit"


# L02 ------------------------------------------------------------------------------------------
def test_L02_every_committed_entity_has_resolvable_evidence_or_is_hypothesized(tmp_path):
    app, loop, b, _ = _seeded(tmp_path)
    prov = {p["entity_id"]: p for p in loop.ledger.provenance()}
    g = app.graph()
    for n in g.live_nodes():
        assert prov[n.id]["citations"], n.id
        for c in prov[n.id]["citations"]:
            doc = b.get(c["source_id"])
            s, e = c["resolved_lines"]
            assert c["excerpt"] in " ".join(doc.lines[s - 1:e])
    for e in g.live_edges():
        assert e.evidence_ref and e.evidence_ref.startswith("P-0001/")
        assert e.status == "hypothesized" or prov[e.id]["citations"]
    # An unevidenced assertion is quarantined, and a hypothesized edge is stored as hypothesized.
    l2 = loop_with(app, {"k": {"intent": "x", "operations": [
        op_node("op-1", "REQ-NEW", "requirement", "New", {"acceptance_criteria": []}, [], support="supported"),
        op_edge("op-2", "CAP-HOLDS", "refines", "REQ-NEW", [], support="hypothesized")]}})
    st = l2.propose(b, {}, call_key="k")
    v = st.report["ops"]
    assert v["op-1"]["verdict"] == "quarantined" and "unsupported_claim" in [f["code"] for f in v["op-1"]["findings"]]
    assert v["op-2"]["engine_op"]["edge"]["status"] == "hypothesized"


# L03 ------------------------------------------------------------------------------------------
def test_L03_fabricated_and_stale_citations_are_rejected(tmp_path):
    app, loop, b, _ = _seeded(tmp_path)
    cat = SourceCatalog()
    cat.register([REQS_V1, REQS_V2])
    b2 = bundle(REQS_V2)
    raw = {"intent": "x", "operations": [
        op_node("op-1", "REQ-A", "requirement", "A", {"acceptance_criteria": []},
                [cite(REQS_V2, "Members must be able to pay fines online.", lines=[5, 5])]),
        op_node("op-2", "REQ-B", "requirement", "B", {"acceptance_criteria": []},
                [cite(REQS_V1, "REQ-PLACE-HOLD — A member can place a hold on a catalogued tool that is on loan.")]),
        op_node("op-3", "REQ-C", "requirement", "C", {"acceptance_criteria": []},
                [{"source_id": "toolshare:spec/invented.md", "sha256": "0" * 64, "lines": [1, 1], "excerpt": "x"}]),
        op_node("op-4", "REQ-D", "requirement", "D", {"acceptance_criteria": []},
                [cite(REQS_V2, "REQ-PLACE-HOLD", sha="f" * 64)]),
    ]}
    prop = parse_proposal(raw, "P-T", app.graph().revision)
    _, rep = validate_proposal(app, prop, b2, cat, loop.ledger)
    codes = {k: {f["code"] for f in v["findings"]} for k, v in rep["ops"].items()}
    assert rep["ops"]["op-1"]["verdict"] == "invalid" and "fabricated_citation" in codes["op-1"]
    assert rep["ops"]["op-2"]["verdict"] == "invalid" and "stale_citation" in codes["op-2"]  # cites draft 1 hash
    assert "fabricated_citation" in codes["op-3"] and "fabricated_citation" in codes["op-4"]
    before = snapshot(app)
    with pytest.raises(ReviewError):
        from graph_adlc.llm_graph.review import apply_review
        apply_review(app, loop.ledger, prop, rep, [Decision(f"op-{i}", "accept", "alice", acknowledge=True)
                                                   for i in range(1, 5)], b2, cat)
    assert snapshot(app) == before


# L04 ------------------------------------------------------------------------------------------
def test_L04_reprocessing_same_source_is_idempotent_and_paraphrases_are_flagged(tmp_path):
    app, loop, b, _ = _seeded(tmp_path)
    before = snapshot(app)
    again = loop_with(app, {"seed": seed_response()})
    st = again.propose(b, {}, call_key="seed")
    assert set(st.report["summary"]["duplicate"]) == {f"op-{i}" for i in range(1, 6)}
    res = again.review(st, accept_all(st), b)
    assert res["change_id"] is None and snapshot(app) == before
    para = loop_with(app, {"k": {"intent": "x", "operations": [
        op_node("op-1", "REQ-HOLD-PLACEMENT", "requirement", "Place a Hold!", {"acceptance_criteria": []},
                [cite(REQS_V1, "REQ-PLACE-HOLD — A member can place a hold on a catalogued tool that is on loan.")])]}})
    st = para.propose(b, {}, call_key="k")
    assert "duplicate_suspected" in {f["code"] for f in st.report["ops"]["op-1"]["findings"]}


# L05 ------------------------------------------------------------------------------------------
def test_L05_editing_one_requirement_is_a_targeted_delta(tmp_path):
    app, loop, _, _ = _seeded(tmp_path)
    revs_before = {n.id: n.rev for n in app.graph().live_nodes()}
    edges_before = {e.id: e.rev for e in app.graph().live_edges()}
    b2 = bundle(REQS_V2)
    l2 = loop_with(app, {"k": {"intent": "draft 2", "operations": [
        {"op_id": "op-1", "kind": "update_node", "id": "REQ-HOLD-EXPIRY", "expected_rev": 1, "aspect": "semantic",
         "patch": {"data": {"acceptance_criteria": ["The pickup window is 48 hours from the moment the ready notice is sent."]}},
         "support": "supported",
         "source_evidence": [cite(REQS_V2, "- The pickup window is 48 hours from the moment the ready notice is sent.")]}]}})
    st = l2.propose(b2, {}, call_key="k")
    res = l2.review(st, accept_all(st), b2)
    assert [d["op"] for d in res["commit"]["diff"]] == ["update_node"]
    after = {n.id: n.rev for n in app.graph().live_nodes()}
    assert after == {**revs_before, "REQ-HOLD-EXPIRY": 2}
    assert {e.id: e.rev for e in app.graph().live_edges()} == edges_before


# L06 ------------------------------------------------------------------------------------------
def test_L06_silence_does_not_delete_relationships(tmp_path):
    app, loop, _, _ = _seeded(tmp_path)
    eid = next(e.id for e in app.graph().live_edges() if e.dst == "REQ-PLACE-HOLD")
    silent = SourceBundle(project="toolshare", documents=[ARCH])  # a source that never mentions REQ-PLACE-HOLD
    l2 = loop_with(app, {"k": {"intent": "nothing new", "operations": []},
                         "r": {"intent": "remove", "operations": [
                             {"op_id": "op-1", "kind": "remove_edge", "id": eid, "expected_rev": 1, "support": "hypothesized",
                              "reason": "not mentioned any more", "source_evidence": []}]},
                         "r2": {"intent": "remove", "operations": [
                             {"op_id": "op-1", "kind": "remove_edge", "id": eid, "expected_rev": 1, "support": "supported",
                              "reason": "explicitly dropped", "source_evidence": [cite(ARCH, "It implements REQ-PLACE-HOLD and REQ-HOLD-EXPIRY.")]}]}})
    l2.propose(silent, {}, call_key="k")
    assert app.graph().edges[eid].deleted is False
    st = l2.propose(silent, {}, call_key="r")
    v = st.report["ops"]["op-1"]
    assert v["verdict"] == "invalid" and "removal_without_evidence" in {f["code"] for f in v["findings"]}
    st2 = l2.propose(silent, {}, call_key="r2")
    assert st2.report["ops"]["op-1"]["verdict"] == "quarantined"
    with pytest.raises(ReviewError):  # needs explicit authorization
        l2.review(st2, [Decision("op-1", "accept", "alice")], silent)
    assert app.graph().edges[eid].deleted is False


# L07 ------------------------------------------------------------------------------------------
def test_L07_conflict_becomes_question_not_replacement(tmp_path):
    app, loop, b, _ = _seeded(tmp_path)
    code = MaintenanceLoop  # noqa: F841 (readability)
    l2 = loop_with(app, {"k": {"intent": "conflict", "operations": [
        {"op_id": "op-1", "kind": "update_node", "id": "REQ-PLACE-HOLD", "expected_rev": 1, "aspect": "semantic",
         "patch": {"body": "Anyone may place a hold, even with overdue items."}, "support": "supported",
         "source_evidence": [cite(ARCH, "It implements REQ-PLACE-HOLD and REQ-HOLD-EXPIRY.")]}],
        "possible_contradictions": [{"description": "architecture and requirements disagree on who may hold",
                                     "between": ["REQ-PLACE-HOLD"],
                                     "source_evidence": [cite(REQS_V1, "A member without overdue items can place a hold on any catalogued tool.")]}]}})
    b2 = bundle(REQS_V1, ARCH)
    st = l2.propose(b2, {}, call_key="k")
    v = st.report["ops"]
    assert v["op-1"]["verdict"] == "quarantined" and "contradiction" in {f["code"] for f in v["op-1"]["findings"]}
    q = [o for o in st.proposal.operations if o.op_id == "auto-q1"][0]
    assert q.payload["node"]["kind"] == "question" and q.payload["node"]["data"]["blocking"] is True
    res = l2.review(st, [Decision("op-1", "defer", "alice"), Decision("auto-q1", "accept", "alice"),
                         Decision("auto-q1-b1", "accept", "alice")], b2)
    g = app.graph()
    assert g.node("REQ-PLACE-HOLD").rev == 1  # not replaced
    assert app.readiness("REQ-PLACE-HOLD").readiness == "blocked" and res["change_id"]
    # Code evidence alone cannot rewrite intent.
    from graph_adlc.llm_graph import SourceDocument
    code_doc = SourceDocument("git:toolshare@abc:toolshare/holds.py", "toolshare", "toolshare/holds.py", "code", "abc",
                              "def place_hold(member, tool):\n    allow_overdue = True\n")
    l3 = loop_with(app, {"k": {"intent": "code says otherwise", "operations": [
        {"op_id": "op-1", "kind": "update_node", "id": "REQ-HOLD-EXPIRY", "expected_rev": 1, "aspect": "semantic",
         "patch": {"body": "Overdue members may hold."}, "support": "supported",
         "source_evidence": [cite(code_doc, "allow_overdue = True")]}]}})
    st3 = l3.propose(bundle(code_doc), {}, call_key="k")
    assert "authority_conflict" in {f["code"] for f in st3.report["ops"]["op-1"]["findings"]}


# L08 ------------------------------------------------------------------------------------------
def test_L08_reject_and_defer_leave_graph_untouched_accept_is_audited(tmp_path):
    app = new_app(tmp_path)
    loop = loop_with(app, {"seed": seed_response()})
    b = bundle(REQS_V1)
    st = loop.propose(b, {}, call_key="seed")
    before, changes = snapshot(app), len(app.log())
    res = loop.review(st, [Decision(o.op_id, "reject" if i % 2 else "defer", "alice", "no")
                           for i, o in enumerate(st.proposal.operations)], b)
    assert res["change_id"] is None and snapshot(app) == before and len(app.log()) == changes
    with pytest.raises(ReviewError):  # every op needs a decision
        loop.review(st, [Decision("op-1", "accept", "alice")], b)
    st = loop.propose(b, {}, call_key="seed")
    # Ops rejected above come back quarantined (resubmitted unchanged); deferred ones are admissible again.
    rejected = {o.op_id for i, o in enumerate(st.proposal.operations) if i % 2}
    for oid, v in st.report["ops"].items():
        codes = {f["code"] for f in v["findings"]}
        assert (v["verdict"], "previously_rejected" in codes) == (("quarantined", True) if oid in rejected else ("admissible", False))
    with pytest.raises(ReviewError):
        loop.review(st, accept_all(st), b)
    res = loop.review(st, accept_all(st, ack=True), b)  # explicit, acknowledged human override
    ch = app.log()[-1]
    assert ch["id"] == res["change_id"] and "P-0002" in ch["rationale"]
    decs = Ledger(app).decisions("P-0002")
    assert {d["decision"] for d in decs} == {"accept"} and all(d["change_id"] == res["change_id"] for d in decs)
    # Rejected suggestions are not silently resubmitted unchanged.
    rej = Ledger(app).decisions("P-0001")
    assert any(d["decision"] == "reject" for d in rej)


def test_L08b_rejected_suggestion_resubmitted_unchanged_is_quarantined(tmp_path):
    app, loop, b, _ = _seeded(tmp_path)
    resp = {"intent": "x", "operations": [op_edge("op-1", "REQ-PLACE-HOLD", "depends_on", "REQ-HOLD-EXPIRY",
                                                  [cite(REQS_V1, "Holds on the same tool are served first come, first served.")])]}
    l2 = loop_with(app, {"k": resp})
    st = l2.propose(b, {}, call_key="k")
    l2.review(st, [Decision("op-1", "reject", "alice", "not a dependency")], b)
    st = l2.propose(b, {}, call_key="k")
    assert "previously_rejected" in {f["code"] for f in st.report["ops"]["op-1"]["findings"]}


# L09 ------------------------------------------------------------------------------------------
def test_L09_code_diff_reveals_missing_dependency_with_real_evidence(tmp_path):
    import replay_lib as rl
    hist = json.loads((HERE / "histories" / "toolshare.json").read_text())
    repo, shas = rl.build_repo(HERE / "sources" / "toolshare", hist, tmp_path / "repo")
    ref = json.loads((HERE / "reference" / "toolshare.json").read_text())
    ref["edges"] = [e for e in ref["edges"] if not (e["src"] == "CMP-HOLDS" and e["dst"] == "CMP-NOTIFY")]
    app = __import__("graph_adlc.core", fromlist=["GraphADLC"]).GraphADLC(str(tmp_path / "g.sqlite"))
    app.seed_graph(ref)
    st = rl.SourceState("toolshare", HERE / "sources" / "toolshare", repo, "toolshare", shas)
    st.advance(hist["events"][0])
    b = st.advance(hist["events"][2])  # c2 + diff c1..c2
    loop = MaintenanceLoop(app)
    before = rl.probe(app, "CON-NOTICE-FORMAT", "contract")["targets"]
    res = loop.challenge({"node_id": "CMP-HOLDS", "aspect": "implementation"}, b)
    scan = res["steps"]["scan"]
    ops = [o for o in scan.proposal.operations if o.kind == "add_edge"]
    assert [(o.payload["edge"]["src"], o.payload["edge"]["dst"]) for o in ops] == [("CMP-HOLDS", "CMP-NOTIFY")]
    c = scan.report["ops"][ops[0].op_id]["citations"][0]
    assert c["status"] == "verified" and c["excerpt"] == "from toolshare.notifications import send_email"
    assert c["source_id"].startswith(f"git:toolshare@{shas['c2']}:")
    assert app.graph().out_edges("CMP-HOLDS", "depends_on")[0].dst == "CMP-INVENTORY"  # not inserted yet
    loop.review(scan, accept_all(scan), b)
    after = rl.probe(app, "CON-NOTICE-FORMAT", "contract")["targets"]
    assert "CMP-HOLDS" not in before and "CMP-HOLDS" in after


# L10 ------------------------------------------------------------------------------------------
def test_L10_sequential_changes_on_one_persistent_graph_with_before_after_report(scripted_run):
    t = json.loads((scripted_run / "toolshare" / "G2" / "trace.json").read_text())
    changes = [c["id"] for c in t["change_log"]]
    assert len(changes) >= 4 and changes == [f"CHG-{i}" for i in range(2, 2 + len(changes))]
    revs = [(e["graph_revision_before"], e["graph_revision_after"]) for e in t["events"]]
    assert all(a[1] == b[0] for a, b in zip(revs, revs[1:]))  # never reset
    e3 = next(e for e in t["events"] if e["event"] == "E3")
    assert "CMP-HOLDS" not in e3["probe_before"]["targets"] and "CMP-HOLDS" in e3["probe_after"]["targets"]
    assert (scripted_run / "toolshare" / "G2" / "spec_E5.md").exists()


# L11 ------------------------------------------------------------------------------------------
def test_L11_wording_only_change_does_not_propagate(tmp_path):
    app, loop, b, _ = _seeded(tmp_path)
    rev = app.graph().revision
    l2 = loop_with(app, {"k": {"intent": "reword", "operations": [
        {"op_id": "op-1", "kind": "no_change", "target": "REQ-PLACE-HOLD", "change_class": "editorial",
         "support": "supported", "source_evidence": [cite(REQS_V1, "REQ-PLACE-HOLD — A member can place a hold on a catalogued tool that is on loan.")]},
        {"op_id": "op-2", "kind": "update_node", "id": "REQ-HOLD-EXPIRY", "expected_rev": 1, "aspect": "semantic",
         "patch": {"title": "hold   EXPIRY."}, "support": "supported",
         "source_evidence": [cite(REQS_V1, "REQ-HOLD-EXPIRY — When a held tool becomes available, the member has a pickup window to collect it. If they do not")]}]}})
    st = l2.propose(b, {}, call_key="k")
    assert "editorial_suspected" in {f["code"] for f in st.report["ops"]["op-2"]["findings"]}
    assert st.report["ops"]["op-2"]["verdict"] == "quarantined"
    res = l2.review(st, [Decision("op-1", "accept", "alice"), Decision("op-2", "defer", "alice")], b)
    assert res["no_change_records"] == ["op-1"] and res["change_id"] is None and app.graph().revision == rev


# L12 ------------------------------------------------------------------------------------------
def test_L12_passing_current_but_incomplete_evidence_cannot_prune(scripted_run, tmp_path):
    t = json.loads((scripted_run / "ledgerlite" / "G2" / "trace.json").read_text())
    ep = t["prediction"]["exploratory_pruning"]
    assert ep["evidence_gate"]["admissible"] == []
    reasons = " ".join(ep["evidence_gate"]["refused"][0]["reasons"])
    assert "lacks ['1000+']" in reasons
    # The engine on its own WOULD prune the critical consumer (the SC-14 failure, reproduced) ...
    assert "CMP-CSV" in {p["consumer_id"] for p in ep["engine_pruning_with_all_evidence"]["pruned"]}
    # ... the default (no pruning) and the gated condition keep it.
    assert "CMP-CSV" in t["prediction"]["E5"]["targets"]
    assert "CMP-CSV" in ep["engine_pruning_with_gate_admitted_evidence"]["targets"]
    # No declared domain -> nothing admissible; a model claim is flagged, never evidence.
    app = new_app(tmp_path)
    app.commit({"base_graph_revision": 1, "actor": "t", "rationale": "t", "operations": [
        {"op": "add_node", "node": {"id": "CON-X", "kind": "contract", "title": "X", "data": {"status": "accepted"}}},
        {"op": "add_node", "node": {"id": "CMP-P", "kind": "component", "title": "P", "data": {}}},
        {"op": "add_edge", "edge": {"id": "E1", "src": "CMP-P", "dst": "CON-X", "relation": "constrained_by"}}]})
    app.record_evidence("CMP-P", "contract_preservation", "passed", details={"coverage": {"amount_range": ["0-999"]}})
    assert assess_preservation_evidence(app, "CMP-P", None)["admissible"] == []
    assert assess_preservation_evidence(app, "CMP-P", {"amount_range": ["0-999"]})["admissible"]
    loop = loop_with(app, {"k": {"intent": "x", "operations": [
        {"op_id": "op-1", "kind": "update_node", "id": "CMP-P", "expected_rev": 1, "aspect": "implementation",
         "patch": {"data": {"impl": "v2"}}, "contract_preserved": True, "support": "supported",
         "source_evidence": [cite(REQS_V1, "Owner: Priya. Status: draft 1 (2026-09-02).")]}]}})
    st = loop.propose(bundle(REQS_V1), {}, call_key="k")
    assert "unverified_preservation_claim" in {f["code"] for f in st.report["ops"]["op-1"]["findings"]}


# L13 ------------------------------------------------------------------------------------------
def test_L13_spec_export_is_plain_markdown_and_code_stays_in_git(scripted_run, tmp_path):
    md = (scripted_run / "toolshare" / "G2" / "spec_E5.md").read_text()
    for needle in ("# toolshare specification", "**REQ-HOLD-EXPIRY**", "## Contracts", "## Open questions",
                   "## Decisions", "Proposed — not approved", "read-only"):
        assert needle in md
    assert "def " not in md and "import " not in md  # no source code in the spec projection
    import replay_lib as rl
    hist = json.loads((HERE / "histories" / "toolshare.json").read_text())
    repo, shas = rl.build_repo(HERE / "sources" / "toolshare", hist, tmp_path)
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    from graph_adlc.llm_graph import load_git_sources
    load_git_sources(repo, "toolshare", "toolshare", shas["c1"])
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True).stdout
    assert status == "" and head == shas["c2"]


# L14 ------------------------------------------------------------------------------------------
def test_L14_v01_and_v02_suites_still_pass():
    for target, expect in (("tests", "44 passed"), ("evaluation/tests", "18 passed")):
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", target], cwd=ROOT,
                           capture_output=True, text=True, env={**__import__("os").environ,
                                                                "PYTHONPATH": str(ROOT / "evaluation")})
        assert r.returncode == 0 and expect in r.stdout, r.stdout[-2000:]


# L15 ------------------------------------------------------------------------------------------
def test_L15_scripted_runs_are_reproducible_and_never_reported_as_llm(scripted_run, tmp_path):
    import run_replay
    preds = json.loads((scripted_run / "predictions.json").read_text())
    assert preds["mode"] == "scripted" and "PIPELINE SELF-CHECK" in preds["evidential_status"]
    for proj in preds["projects"].values():
        for arm, p in proj.items():
            if arm != "H":
                assert p["client"]["is_fake"] is True and p["costs"]["fake_calls"] == p["costs"]["model_calls"]
                assert p["costs"]["input_tokens"] is None
    run_replay.main(["--mode", "scripted", "--out", str(tmp_path / "again")])
    for f in sorted(scripted_run.rglob("*")):
        if f.is_file() and f.name != "metrics.json":
            assert (tmp_path / "again" / f.relative_to(scripted_run)).read_bytes() == f.read_bytes(), f
    assert run_replay.main(["--mode", "live", "--out", str(tmp_path / "live")]) == 2  # refuses without config
    assert not (tmp_path / "live").exists()
