from conftest import delta, load


def _states(app):
    return {r.requirement_id: r.readiness for r in app.frontier()}


def test_T02_local_readiness(app):
    assert _states(app) == {"REQ-CHARGE": "ready", "REQ-REPORT": "ready", "REQ-RETRY": "blocked"}
    retry = app.readiness("REQ-RETRY")
    codes = {b.code for b in retry.blockers}
    assert codes == {"open_blocking_question", "missing_acceptance_criteria"}
    assert any("Q-RETRY-BOUNDS" in b.node_ids for b in retry.blockers)  # structured reasons, not a boolean


def test_T03_conditional_with_tracked_assumption_copied_to_work_unit(refined):
    assumption = {"id": "ASM-PROVIDER-WINDOW", "statement": "Provider honours idempotency keys for 24h",
                  "about": "CON-IDEMPOTENCY", "status": "accepted"}
    crit = refined.get_node("REQ-RETRY").data["acceptance_criteria"]
    refined.commit(delta(2, {"op": "update_node", "id": "REQ-RETRY", "expected_rev": 2, "aspect": "semantic",
                             "patch": {"data": {"acceptance_criteria": crit, "assumptions": [assumption]}}}))
    r = refined.readiness("REQ-RETRY")
    assert r.readiness == "conditional"
    assert [c.code for c in r.conditions] == ["tracked_assumption"]

    w = refined.create_work_unit("REQ-RETRY")
    [a] = w["assumptions"]
    assert a["id"] == "ASM-PROVIDER-WINDOW" and a["about"] == "CON-IDEMPOTENCY"
    assert a["origin_node"] == "REQ-RETRY" and a["origin_rev"] == 3 and a["origin_graph_revision"] == 3
    assert w["readiness_at_baseline"] == "conditional"


def test_T03b_hypothesized_dependency_is_conditional_not_ready(refined):
    refined.commit(delta(2, {"op": "add_edge", "edge": {"id": "E-HYP", "src": "REQ-RETRY", "dst": "CMP-REPORTING",
                                                          "relation": "depends_on", "status": "hypothesized"}}))
    r = refined.readiness("REQ-RETRY")
    assert r.readiness == "conditional"
    assert any(c.code == "hypothesized_edge" for c in r.conditions)


def test_unaccepted_assumption_blocks(refined):
    refined.commit(delta(2, {"op": "update_node", "id": "REQ-RETRY", "expected_rev": 2,
                             "patch": {"data": {"assumptions": [{"statement": "x", "status": "proposed"}]}}}))
    assert refined.readiness("REQ-RETRY").readiness == "blocked"


def test_T04_refinement_unblocks_retry_without_touching_unrelated_revisions(app):
    before = app.graph()
    report_rev, charge_rev = before.nodes["REQ-REPORT"].rev, before.nodes["REQ-CHARGE"].rev
    preview = app.propose(load("retry_refinement.json"))
    assert preview["newly_ready"] == ["REQ-RETRY"]
    assert app.readiness("REQ-RETRY").readiness == "blocked"  # propose is read-only

    res = app.commit(load("retry_refinement.json"))
    assert res["graph_revision"] == 2
    after = app.graph()
    assert app.readiness("REQ-RETRY").readiness == "ready"
    assert after.nodes["REQ-REPORT"].rev == report_rev
    assert after.nodes["REQ-CHARGE"].rev == charge_rev
    assert {nid for nid in after.nodes if after.nodes[nid].rev != before.nodes[nid].rev} == {"REQ-RETRY", "Q-RETRY-BOUNDS"}
    assert _states(app) == {"REQ-CHARGE": "ready", "REQ-REPORT": "ready", "REQ-RETRY": "ready"}


def test_contract_status_drives_readiness(app):
    app.commit(delta(1, {"op": "update_node", "id": "CON-IDEMPOTENCY", "expected_rev": 1, "aspect": "contract",
                         "patch": {"data": {"status": "incompatible"}}}))
    assert app.readiness("REQ-CHARGE").readiness == "blocked"
    assert app.readiness("REQ-REPORT").readiness == "ready"  # not a global phase
