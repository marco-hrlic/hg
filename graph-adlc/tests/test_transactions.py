import pytest

from graph_adlc import RevisionConflict, ValidationError

from conftest import delta, load


def _db_state(app):
    return app.export(), app.log(), app.store.get_meta("graph_revision")


def test_propose_is_read_only(app):
    before = _db_state(app)
    events_before = app.store.list_events()
    rep = app.propose(load("retry_refinement.json"))
    assert rep["committable"] and rep["proposed_graph_revision"] == 2
    assert _db_state(app) == before
    assert app.store.list_events() == events_before


def test_T10_second_proposal_from_same_base_rejected_atomically(app):
    first = load("retry_refinement.json")
    second = delta(1, {"op": "update_node", "id": "REQ-REPORT", "expected_rev": 1, "aspect": "semantic",
                       "patch": {"title": "View payment summary (paginated)"}},
                   {"op": "add_node", "node": {"id": "Q-PAGINATION", "kind": "question", "title": "Page size?",
                                               "data": {"blocking": False, "state": "open"}}})
    assert app.propose(first)["committable"] and app.propose(second)["committable"]
    app.commit(first)
    snapshot = _db_state(app)
    with pytest.raises(RevisionConflict) as exc:
        app.commit(second)
    assert "obsolete" in str(exc.value)
    assert _db_state(app) == snapshot  # no partial changes
    assert app.graph().node("Q-PAGINATION") is None
    assert app.get_node("REQ-REPORT").rev == 1
    assert not app.propose(second)["committable"]


def test_stale_expected_rev_rejected(refined):
    snapshot = _db_state(refined)
    d = delta(2, {"op": "update_node", "id": "REQ-REPORT", "expected_rev": 1, "patch": {"body": "ok"}},
              {"op": "update_node", "id": "REQ-RETRY", "expected_rev": 1, "patch": {"body": "stale"}})
    with pytest.raises(RevisionConflict) as exc:
        refined.commit(d)
    assert "REQ-RETRY expected_rev 1 but current rev is 2" in str(exc.value)
    assert _db_state(refined) == snapshot


def test_invariant_failure_mid_delta_is_atomic(app):
    snapshot = _db_state(app)
    d = delta(1, {"op": "update_node", "id": "REQ-REPORT", "expected_rev": 1, "patch": {"body": "fine"}},
              {"op": "add_edge", "edge": {"id": "EX", "src": "TEST-RETRY", "dst": "CMP-PAYMENT", "relation": "verifies"}})
    with pytest.raises(ValidationError):
        app.commit(d)
    assert _db_state(app) == snapshot


def test_commit_is_audited_with_diff_and_versions(app):
    res = app.commit(load("retry_refinement.json"))
    [chg] = app.log()
    assert chg["id"] == res["change_id"] == "CHG-2"
    assert chg["graph_rev"] == 2 and chg["base_graph_rev"] == 1 and chg["actor"] == "discovery-agent"
    retry_diff = next(d for d in chg["diff"] if d["id"] == "REQ-RETRY")
    assert retry_diff["before"]["rev"] == 1 and retry_diff["after"]["rev"] == 2
    assert "data.acceptance_criteria" in retry_diff["fields"]
    assert any(e["kind"] == "committed" and e["ref_id"] == "CHG-2" for e in app.store.list_events())


def test_approval_policy(app):
    app.store.set_meta("approval_required_aspects", "contract")
    d = delta(1, {"op": "update_node", "id": "CON-IDEMPOTENCY", "expected_rev": 1, "aspect": "contract",
                  "patch": {"data": {"guarantee_window": "24h"}}})
    assert app.propose(d)["approvals_required"] == ["contract"]
    with pytest.raises(ValidationError):
        app.commit(d)
    assert app.commit(d, approvals=["architect"])["graph_revision"] == 2
