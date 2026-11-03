import pytest

from graph_adlc import LifecycleError

from conftest import delta, load


def test_blocked_requirement_cannot_get_work_unit(app):
    with pytest.raises(LifecycleError) as exc:
        app.create_work_unit("REQ-RETRY")
    assert exc.value.details["readiness"]["readiness"] == "blocked"


def test_work_unit_baseline_captures_requirement_and_contract_revisions(refined):
    w = refined.create_work_unit("REQ-RETRY")
    assert w["base_graph_revision"] == 2
    assert w["node_revisions"] == {"CON-IDEMPOTENCY": 1, "REQ-RETRY": 2, "TEST-RETRY": 1}
    assert w["acceptance_criteria"][0] == "At most three attempts are made"
    assert w["required_checks"] == [{"kind": "test_result", "node_id": "TEST-RETRY", "reason": "test verifies target"}]
    assert w["status"] == "proposed"


def test_verification_requires_required_checks(refined):
    w = refined.create_work_unit("REQ-RETRY")
    out = refined.submit_work_unit(w["id"], "git:1", [])
    assert out["status"] == "submitted"
    assert out["evaluation"]["missing_checks"][0]["node_id"] == "TEST-RETRY"
    out = refined.submit_work_unit(w["id"], "git:1", [{"node_id": "TEST-RETRY", "result": "failed"}])
    assert out["status"] == "submitted" and out["evaluation"]["missing_checks"][0]["status"] == "failed"
    out = refined.submit_work_unit(w["id"], "git:2", [{"node_id": "TEST-RETRY", "result": "passed"}])
    assert out["status"] == "verified" and out["code_rev"] == "git:2"
    assert refined.freshness("REQ-RETRY") == "current"


def test_T11_evidence_freshness_after_contract_revision(verified_retry):
    app, wid = verified_retry
    [ev] = app.evidence()
    assert ev["current"] and ev["spec_revs"]["CON-IDEMPOTENCY"] == 1
    res = app.commit(load("idempotency_window_change.json"))
    assert app.get_node("CON-IDEMPOTENCY").rev == 2
    assert {e["evidence_id"] for e in res["invalidated"]["evidence"]} == {ev["id"]}
    assert [w["work_unit_id"] for w in res["invalidated"]["work_units"]] == [wid]
    [ev2] = app.evidence()
    assert not ev2["current"]
    assert ev2["invalidated_by"] == "CHG-3"
    assert any("CON-IDEMPOTENCY recorded at r1, now r2" in r for r in ev2["stale_reasons"])
    assert app.work_unit(wid)["status"] == "needs_revalidation"
    # Readiness and freshness are separate dimensions.
    assert app.readiness("REQ-RETRY").readiness == "ready"
    assert app.freshness("REQ-RETRY") == "needs_revalidation"


def test_propose_previews_invalidation_without_applying(verified_retry):
    app, wid = verified_retry
    rep = app.propose(load("idempotency_window_change.json"))
    assert [w["work_unit_id"] for w in rep["would_invalidate"]["work_units"]] == [wid]
    assert app.work_unit(wid)["status"] == "verified"


def test_T13_context_is_bounded(refined):
    pkt = refined.context("REQ-RETRY")
    ids = set(pkt["node_revisions"])
    assert {"REQ-RETRY", "CAP-PAYMENTS", "GOAL-CHECKOUT", "CON-IDEMPOTENCY", "CMP-PAYMENT", "TEST-RETRY",
            "Q-RETRY-BOUNDS"} <= ids
    assert not ids & {"REQ-REPORT", "CMP-REPORTING", "CAP-REPORTING", "REQ-CHARGE", "ART-PAYMENT"}
    assert pkt["node_revisions"]["REQ-RETRY"] == 2 and pkt["graph_revision"] == 2
    assert all(n["included_because"] for n in pkt["nodes"])
    assert all(e["src"] in ids and e["dst"] in ids for e in pkt["edges"])

    # An explicit relevant edge brings the connected node in (but not its unrelated requirement).
    refined.commit(delta(2, {"op": "add_edge", "edge": {"id": "E-RPT", "src": "REQ-RETRY", "dst": "CMP-REPORTING",
                                                          "relation": "depends_on"}}))
    ids2 = set(refined.context("REQ-RETRY")["node_revisions"])
    assert "CMP-REPORTING" in ids2 and "REQ-REPORT" not in ids2


def test_T14_stale_work_cannot_be_verified_without_reconciliation(verified_retry):
    app, wid = verified_retry
    app.commit(load("idempotency_window_change.json"))

    with pytest.raises(LifecycleError):
        app.submit_work_unit(wid, "git:abc123", [{"node_id": "TEST-RETRY", "result": "passed"}])
    assert app.work_unit(wid)["status"] == "needs_revalidation"

    report = app.reconcile(wid)  # inspection only
    assert not report["can_verify"] and report["status_after"] == "needs_revalidation"
    assert report["baseline_mismatches"] == [{"node_id": "CON-IDEMPOTENCY", "baseline_rev": 1, "current_rev": 2}]
    assert any(o["target_id"] == "REQ-RETRY" and o["obligation"] == "check_contract"
               for o in report["obligations_since_baseline"])

    rebased = app.reconcile(wid, rebase=True)
    assert rebased["rebased"] and rebased["status_after"] == "active"
    checks = {(c["kind"], c["node_id"]) for c in rebased["new_baseline"]["required_checks"]}
    assert checks == {("test_result", "TEST-RETRY"), ("contract_check", "CON-IDEMPOTENCY")}

    # Fresh test alone is not enough; the contract check is still required.
    out = app.submit_work_unit(wid, "git:fed987", [{"node_id": "TEST-RETRY", "result": "passed"}])
    assert out["status"] == "submitted"
    assert [c["node_id"] for c in out["evaluation"]["missing_checks"]] == ["CON-IDEMPOTENCY"]
    out = app.submit_work_unit(wid, "git:fed987", [{"node_id": "CON-IDEMPOTENCY", "kind": "contract_check",
                                                    "result": "passed"}])
    assert out["status"] == "verified"
    assert app.freshness("REQ-RETRY") == "current"


def test_old_evidence_does_not_count_after_rebase(verified_retry):
    app, wid = verified_retry
    app.commit(load("idempotency_window_change.json"))
    app.reconcile(wid, rebase=True)
    out = app.submit_work_unit(wid, "git:x", [])
    assert {c["node_id"] for c in out["evaluation"]["missing_checks"]} == {"TEST-RETRY", "CON-IDEMPOTENCY"}


def test_scheduler_prioritizes_reconciliation_and_keeps_unaffected_work(verified_retry):
    app, wid = verified_retry
    app.commit(load("idempotency_window_change.json"))
    acts = app.suggest_actions()
    assert acts[0].action == "reconcile" and acts[0].target_ids[0] == wid
    implement = [a.target_ids[0] for a in acts if a.action == "implement"]
    assert "REQ-REPORT" in implement and "REQ-CHARGE" in implement


def test_scheduler_ranks_unblocking_question_before_implementation(app):
    acts = app.suggest_actions()
    assert acts[0].action == "resolve_question" and acts[0].target_ids == ["Q-RETRY-BOUNDS", "REQ-RETRY"]
    assert [a.action for a in acts[1:3]] == ["implement", "implement"]
    # smaller scope first: REQ-REPORT has fewer criteria than REQ-CHARGE
    assert acts[1].target_ids == ["REQ-REPORT"]
