from conftest import by_target, delta, load


def test_T05_semantic_change_to_consumer_does_not_invalidate_provider(app):
    rep = app.propose(delta(1, {"op": "update_node", "id": "CMP-REPORTING", "expected_rev": 1, "aspect": "semantic",
                                "patch": {"body": "Provides transaction summaries, now paginated."}}))
    targets = by_target(rep["impact"]["obligations"])
    assert "CMP-PAYMENT" not in targets
    assert "CON-IDEMPOTENCY" not in targets
    assert targets.get("REQ-REPORT") == {"check_contract"}  # its own implemented requirement is checked
    # Same via the hypothetical API for a consumer requirement.
    hyp = by_target(app.impact("REQ-REPORT", "semantic").records)
    assert "CMP-PAYMENT" not in hyp and "CON-IDEMPOTENCY" not in hyp


def test_T06_contract_change_reaches_consumers_and_tests_with_paths(refined):
    rep = refined.propose(load("idempotency_window_change.json"))
    assert rep["committable"]
    obs = rep["impact"]["obligations"]
    t = by_target(obs)
    assert "check_contract" in t["REQ-RETRY"]
    assert "check_contract" in t["REQ-CHARGE"]
    assert "check_contract" in t["CMP-PAYMENT"]
    assert "rerun_test" in t["TEST-RETRY"]
    # provider-dependent consumer receives a review obligation (no validated boundary exists)
    assert "check_contract" in t["CMP-REPORTING"]
    # unrelated leaf sharing the same root is not invalidated
    assert "REQ-REPORT" not in t and "CAP-REPORTING" not in t and "GOAL-CHECKOUT" not in t
    paths = {(o["target_id"], o["obligation"]): o["path"] for o in obs}
    assert paths[("REQ-RETRY", "check_contract")] == ["E07"]
    assert paths[("TEST-RETRY", "rerun_test")] == ["E11"]
    assert paths[("CMP-REPORTING", "check_contract")] == ["E16", "E13"]


def _impl_change(base: int, **extra) -> dict:
    op = {"op": "update_node", "id": "CMP-PAYMENT", "expected_rev": 1, "aspect": "implementation",
          "patch": {"data": {"implementation_note": "refactor retry loop into a helper"}}}
    op.update(extra.pop("op_extra", {}))
    return delta(base, op, **extra)


def test_T07_pruning_requires_validation_evidence(app):
    # No evidence: consumer is provisionally flagged.
    plain = app.propose(_impl_change(1))
    flagged = [o for o in plain["impact"]["obligations"] if o["target_id"] == "CMP-REPORTING"]
    assert flagged and flagged[0]["obligation"] == "check_contract" and flagged[0]["provisional"]
    assert plain["impact"]["pruned"] == []

    # Unverified claim: still cannot prune.
    claimed = app.propose(_impl_change(1, op_extra={"contract_preserved": True}))
    assert "CMP-REPORTING" in by_target(claimed["impact"]["obligations"])
    assert any("claim ignored" in n for n in claimed["impact"]["notes"])

    # Recorded contract-preservation evidence at current contract revision: prune.
    ev = app.record_evidence("CMP-PAYMENT", "contract_preservation", "passed", code_rev="git:def456")
    assert ev["spec_revs"]["CON-IDEMPOTENCY"] == 1
    pruned = app.propose(_impl_change(1, evidence_refs=[ev["id"]]))
    t = by_target(pruned["impact"]["obligations"])
    assert "CMP-REPORTING" not in t
    [p] = pruned["impact"]["pruned"]
    assert p["consumer_id"] == "CMP-REPORTING" and p["evidence_id"] == ev["id"] and p["edge_id"] == "E13"
    # The provider's own requirements still get checked (pruning only stops beyond the contract).
    assert "check_contract" in t["REQ-CHARGE"] and "check_contract" in t["REQ-RETRY"]

    # Failed preservation result cannot prune.
    bad = app.record_evidence("CMP-PAYMENT", "contract_preservation", "failed")
    assert "CMP-REPORTING" in by_target(app.propose(_impl_change(1, evidence_refs=[bad["id"]]))["impact"]["obligations"])


def test_T07b_preservation_evidence_goes_stale_after_contract_change(app):
    ev = app.record_evidence("CMP-PAYMENT", "contract_preservation", "passed")
    app.commit(delta(1, {"op": "update_node", "id": "CON-IDEMPOTENCY", "expected_rev": 1, "aspect": "contract",
                         "patch": {"data": {"guarantee_window": "24h"}}}))
    rep = app.propose(_impl_change(2, evidence_refs=[ev["id"]]))
    assert "CMP-REPORTING" in by_target(rep["impact"]["obligations"])
    assert rep["impact"]["pruned"] == []


def test_T07c_hypothesized_edge_never_pruned_and_raises_investigate(app):
    app.commit(delta(1, {"op": "add_node", "node": {"id": "CMP-AUDIT", "kind": "component", "title": "Audit", "data": {}}},
                     {"op": "add_edge", "edge": {"id": "E-AUD", "src": "CMP-AUDIT", "dst": "CMP-PAYMENT",
                                                   "relation": "depends_on", "status": "hypothesized"}}))
    # (recorded after the structural change: that change itself raised investigate_unknown on CMP-PAYMENT,
    # which conservatively invalidates earlier evidence about it)
    ev = app.record_evidence("CMP-PAYMENT", "contract_preservation", "passed")
    rep = app.propose(_impl_change(2, evidence_refs=[ev["id"]]))
    t = by_target(rep["impact"]["obligations"])
    assert "CMP-REPORTING" not in t  # confirmed edge pruned by evidence
    assert t["CMP-AUDIT"] == {"check_contract", "investigate_unknown"}  # hypothesized edge: no pruning
    low = [o for o in rep["impact"]["obligations"] if o["target_id"] == "CMP-AUDIT"]
    assert all(o["confidence"] == "low" for o in low)


def test_T08_removed_edge_surfaces_both_endpoints(app):
    rep = app.propose(delta(1, {"op": "remove_edge", "id": "E13", "expected_rev": 1}))
    obs = rep["impact"]["obligations"]
    t = by_target(obs)
    assert "check_contract" in t["CMP-REPORTING"]
    assert "review_spec" in t["CMP-PAYMENT"]
    for o in obs:
        if o["target_id"] in ("CMP-REPORTING", "CMP-PAYMENT"):
            assert o["path"] == ["E13"] and o["origin_id"] == "E13"
    # And: a removed edge is still traversed when evaluating other changes in the same delta.
    rep2 = app.propose(delta(1, {"op": "remove_edge", "id": "E13", "expected_rev": 1},
                             {"op": "update_node", "id": "CMP-PAYMENT", "expected_rev": 1, "aspect": "implementation",
                              "patch": {"data": {"implementation_note": "x"}}}))
    assert "CMP-REPORTING" in by_target(rep2["impact"]["obligations"])
    # The union graph used for analysis keeps the removed edge, flagged as removed.
    from graph_adlc.delta import apply_delta
    from graph_adlc.influence import UnionGraph
    applied = apply_delta(app.graph(), delta(1, {"op": "remove_edge", "id": "E13", "expected_rev": 1}), "t")
    union = UnionGraph(applied.before, applied.after)
    assert union.edges["E13"][1] == "removed"
    assert ("E13" in {e.id for e, _ in union.incident("CMP-REPORTING")})


def test_T09_cycle_terminates_with_unique_obligations(app):
    res = app.commit(delta(1, {"op": "add_edge", "edge": {"id": "E-REV", "src": "CMP-PAYMENT", "dst": "CMP-REPORTING",
                                                            "relation": "depends_on"}}))
    assert res["graph_revision"] == 2
    for node, aspect in [("CMP-PAYMENT", "implementation"), ("CMP-REPORTING", "implementation"),
                         ("CMP-PAYMENT", "semantic"), ("CON-IDEMPOTENCY", "contract"), ("GOAL-CHECKOUT", "semantic")]:
        recs = app.impact(node, aspect).records
        keys = [(r.target_id, r.obligation) for r in recs]
        assert len(keys) == len(set(keys)), (node, aspect)
    both = by_target(app.impact("CMP-PAYMENT", "implementation").records)
    assert "CMP-REPORTING" in both
    back = by_target(app.impact("CMP-REPORTING", "implementation").records)
    assert "CMP-PAYMENT" in back


def test_semantic_parent_change_recurses_down_child_change_terminal_up(app):
    down = by_target(app.impact("GOAL-CHECKOUT", "semantic").records)
    assert {"CAP-PAYMENTS", "CAP-REPORTING", "REQ-CHARGE", "REQ-RETRY", "REQ-REPORT"} <= set(down)
    assert all(obs == {"review_spec"} for obs in down.values())
    up = by_target(app.impact("REQ-CHARGE", "semantic").records)
    assert up["CAP-PAYMENTS"] == {"review_spec"}
    assert "GOAL-CHECKOUT" not in up  # terminal parent obligation
    assert "REQ-RETRY" not in up or "review_spec" not in up["REQ-RETRY"]  # no sibling fan-out


def test_T12_every_obligation_has_origin_reason_and_path(refined):
    reports = [
        refined.propose(load("idempotency_window_change.json"))["impact"],
        refined.propose(delta(2, {"op": "remove_edge", "id": "E13", "expected_rev": 1}))["impact"],
        refined.impact("GOAL-CHECKOUT", "semantic").to_dict(),
        refined.impact("ART-PAYMENT", "implementation").to_dict(),
    ]
    committed = refined.commit(load("idempotency_window_change.json"))
    reports.append(committed["impact"])
    for rep in reports:
        assert rep["obligations"]
        for o in rep["obligations"]:
            assert o["origin_id"] and o["reason"] and o["path"], o
            assert o["change_ref"]
    assert all(o["change_ref"].startswith("CHG-3/") for o in committed["impact"]["obligations"])
    stored = refined.log()[-1]
    assert stored["id"] == "CHG-3" and stored["impact"]["obligations"] == committed["impact"]["obligations"]
