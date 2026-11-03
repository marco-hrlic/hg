"""End-to-end replay of the vertical slice through the CLI (SPEC §6 / ACCEPTANCE trace shape)."""

from graph_adlc.demo import run


def test_vertical_slice_trace(tmp_path):
    doc = run(str(tmp_path), echo=False)
    steps = {s["title"]: s for s in doc["steps"]}

    def st(prefix):
        return next(s for t, s in steps.items() if t.startswith(prefix))

    # 1. init -> rev 1, retry blocked, charge/report ready
    s = st("A. Discovery")
    assert s["exit_code"] == 0 and s["state"]["graph_revision"] == 1
    assert {f["requirement_id"]: f["readiness"] for f in s["state"]["frontier"]} == {
        "REQ-CHARGE": "ready", "REQ-REPORT": "ready", "REQ-RETRY": "blocked"}

    # 2. propose -> newly ready retry + parent consistency obligation; nothing committed
    s = st("B. Refinement: preview")
    assert "REQ-RETRY: blocked -> ready" in s["output"]
    assert "CAP-PAYMENTS" in s["output"] and "review_spec" in s["output"]
    assert s["state"]["graph_revision"] == 1

    # 3. commit -> rev 2
    assert st("B. Refinement: commit")["state"]["graph_revision"] == 2

    # 4/5. work unit baseline; verified only after required check
    s = st("C. Create work unit")
    assert s["state"]["work_units"][0]["node_revisions"] == {"CON-IDEMPOTENCY": 1, "REQ-RETRY": 2, "TEST-RETRY": 1}
    assert st("C. Submit")["state"]["work_units"][0]["status"] == "verified"

    # 6. contract change preview: retry + tests + provider-dependent consumer
    out = st("D. Contradiction")["output"]
    for frag in ("REQ-RETRY", "TEST-RETRY", "CMP-REPORTING", "check_contract", "rerun_test"):
        assert frag in out

    # 7. commit -> evidence stale; reconciliation prioritized
    s = st("E. Influence")
    assert s["state"]["graph_revision"] == 3
    assert s["state"]["work_units"][0]["status"] == "needs_revalidation"
    assert all(not e["current"] for e in s["state"]["evidence"])
    targets = {o["target_id"] for o in s["state"]["latest_change"]["obligations"]}
    assert "REQ-REPORT" not in targets
    assert st("E. Controller")["state"]["suggested_actions"][0]["action"] == "reconcile"
    assert st("E. Attempt")["exit_code"] == 5  # refused

    # 8. reconcile -> current; unaffected work remains available
    final = st("F. Final status")["state"]
    assert final["graph_revision"] == 4
    assert final["work_units"][0]["status"] == "verified"
    assert final["work_units"][0]["base_graph_revision"] == 4
    assert final["freshness"]["REQ-RETRY"] == "current"
    current = [e for e in final["evidence"] if e["current"]]
    assert {(e["node_id"], e["kind"]) for e in current} == {("TEST-RETRY", "test_result"),
                                                            ("CON-IDEMPOTENCY", "contract_check")}
    assert all(e["code_rev"] == "git:d4e5f6" for e in current)
    actions = [(a["action"], a["target_ids"][0]) for a in final["suggested_actions"]]
    assert ("implement", "REQ-REPORT") in actions and ("implement", "REQ-CHARGE") in actions
    assert all(s["exit_code"] == 0 for s in doc["steps"] if not s["title"].startswith("E. Attempt"))
    assert (tmp_path / "trace.json").exists() and (tmp_path / "snapshot.json").exists()
