import json

import pytest

from graph_adlc import GraphADLC, ValidationError
from graph_adlc.validation import classify_update

from conftest import delta, load


def test_T01_partial_seed_imports_all_nodes_and_edges(app, seed):
    g = app.graph()
    assert g.revision == 1
    assert {n["id"] for n in seed["nodes"]} == {n.id for n in g.live_nodes()}
    assert {e["id"] for e in seed["edges"]} == {e.id for e in g.live_edges()}
    # Incomplete regions are permitted: empty criteria + open blocking question.
    assert app.get_node("REQ-RETRY").data["acceptance_criteria"] == []
    assert app.get_node("Q-RETRY-BOUNDS").data["state"] == "open"
    # No global "complete" flag/phase exists; status is per-leaf.
    st = app.status()
    assert "phase" not in st and "complete" not in st
    assert {l["requirement_id"] for l in st["leaf_requirements"]} == {"REQ-CHARGE", "REQ-RETRY", "REQ-REPORT"}


def test_export_import_roundtrip_is_canonical(app, tmp_path):
    snap = app.export()
    other = GraphADLC(str(tmp_path / "copy.sqlite"))
    other.seed_graph(json.loads(app.export_json()))
    assert other.export() == snap
    assert other.export_json() == app.export_json()
    other.close()


def test_seed_rejects_invalid_endpoint_kinds(tmp_path, seed):
    seed["edges"].append({"id": "BAD", "src": "REQ-CHARGE", "dst": "CMP-PAYMENT", "relation": "implements"})
    a = GraphADLC(str(tmp_path / "x.sqlite"))
    with pytest.raises(ValidationError) as exc:
        a.seed_graph(seed)
    assert any("implements not allowed" in i for i in exc.value.issues)
    assert not a.store.initialized()


@pytest.mark.parametrize("ops,needle", [
    # second refinement parent
    ([{"op": "add_edge", "edge": {"id": "EX", "src": "CAP-REPORTING", "dst": "REQ-CHARGE", "relation": "refines"}}],
     "multiple refinement parents"),
    # refinement cycle (move goal under a capability; goal can't be a refines target anyway)
    ([{"op": "add_node", "node": {"id": "REQ-SUB", "kind": "requirement", "title": "Sub", "data": {}}},
      {"op": "add_edge", "edge": {"id": "EX1", "src": "REQ-CHARGE", "dst": "REQ-SUB", "relation": "refines"}},
      {"op": "remove_edge", "id": "E03", "expected_rev": 1},
      {"op": "add_edge", "edge": {"id": "EX2", "src": "REQ-SUB", "dst": "REQ-CHARGE", "relation": "refines"}}],
     "refinement cycle"),
    # dangling reference
    ([{"op": "add_edge", "edge": {"id": "EX", "src": "CMP-PAYMENT", "dst": "NOPE", "relation": "depends_on"}}],
     "does not exist"),
    # wrong direction for verifies
    ([{"op": "add_edge", "edge": {"id": "EX", "src": "REQ-RETRY", "dst": "TEST-RETRY", "relation": "verifies"}}],
     "verifies not allowed"),
    # immutable field
    ([{"op": "update_node", "id": "REQ-CHARGE", "expected_rev": 1, "patch": {"kind": "contract"}}],
     "cannot patch fields"),
    # payload schema
    ([{"op": "update_node", "id": "Q-RETRY-BOUNDS", "expected_rev": 1, "patch": {"data": {"state": "maybe"}}}],
     "question requires data.state"),
])
def test_invalid_deltas_rejected(app, ops, needle):
    before = app.export()
    rep = app.propose(delta(1, *ops))
    assert not rep["committable"]
    assert any(needle in e for e in rep["errors"]), rep["errors"]
    with pytest.raises(ValidationError):
        app.commit(delta(1, *ops))
    assert app.export() == before


def test_remove_node_tombstones_and_cascades(app):
    res = app.commit(delta(1, {"op": "remove_node", "id": "ART-PAYMENT", "expected_rev": 1}))
    g = app.graph()
    assert g.nodes["ART-PAYMENT"].deleted and g.edges["E12"].deleted
    assert g.node("ART-PAYMENT") is None
    assert "CMP-PAYMENT" in {o["target_id"] for o in res["impact"]["obligations"]}
    # ids are never reused
    rep = app.propose(delta(2, {"op": "add_node", "node": {"id": "ART-PAYMENT", "kind": "artifact",
                                                             "title": "x", "data": {"path": "a.py"}}}))
    assert any("tombstoned" in e for e in rep["errors"])


def test_aspect_evidence_cannot_suppress_impact(app):
    # Declaring a requirement change as "evidence" is not accepted.
    d = delta(1, {"op": "update_node", "id": "REQ-CHARGE", "expected_rev": 1, "aspect": "evidence",
                  "patch": {"body": "Charge exactly once; returns a transaction id or a typed error."}})
    rep = app.propose(d)
    assert rep["aspect_overrides"][0]["effective"] == ["semantic"]
    targets = {o["target_id"] for o in rep["impact"]["obligations"]}
    assert {"CAP-PAYMENTS", "CMP-PAYMENT"} <= targets


def test_aspect_classification_table():
    assert classify_update("test", "evidence", {"data": {"last_result": "passed"}}).effective == {"evidence"}
    assert classify_update("test", "evidence", {"data": {"command": "x"}}).overridden
    assert classify_update("contract", "semantic", {"body": "x"}).effective == {"contract", "semantic"}
    assert classify_update("requirement", "implementation", {"body": "x"}).effective == {"semantic"}
    assert classify_update("artifact", "implementation", {"data": {"path": "y"}}).effective == {"implementation"}
