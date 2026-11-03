"""Influence rule set v1.1 (consumer -> implemented requirement) and the replay client used to evaluate it."""
import json

import pytest
from helpers import HERE

from graph_adlc import GraphADLC
from graph_adlc.influence import check_rules


def _ref(tmp_path, project, rules):
    n = next(_ref.counter)
    app = GraphADLC(str(tmp_path / f"{project}-{rules}-{n}.sqlite"), influence_rules=rules)
    app.seed_graph(json.loads((HERE / "reference" / f"{project}.json").read_text()))
    return app


_ref.counter = __import__("itertools").count()


def _pairs(app, node, aspect):
    return {(r.target_id, r.obligation.value if hasattr(r.obligation, "value") else r.obligation): r
            for r in app.impact(node, aspect).records}


def test_v1_is_the_default_and_unknown_rule_sets_are_refused(tmp_path):
    assert GraphADLC(str(tmp_path / "g.sqlite")).influence_rules == "v1"
    with pytest.raises(ValueError):
        check_rules("v2")


def test_v11_reaches_the_requirement_a_flagged_consumer_implements(tmp_path):
    v1 = _pairs(_ref(tmp_path, "ledgerlite", "v1"), "CMP-MONEY", "implementation")
    v11 = _pairs(_ref(tmp_path, "ledgerlite", "v1.1"), "CMP-MONEY", "implementation")
    assert ("REQ-CSV-EXPORT", "check_contract") not in v1        # the v0.3 critical miss
    rec = v11[("REQ-CSV-EXPORT", "check_contract")]
    assert set(v1) < set(v11)                                     # v1.1 only adds obligations
    assert rec.provisional and "rules v1.1" in rec.reason         # inherits the consumer's provisional flag
    assert len(rec.path) == 2                                     # depends_on, then implements: one extra hop


def test_v11_adds_one_hop_only_and_not_for_a_component_checking_its_own_requirement(tmp_path):
    v1 = _pairs(_ref(tmp_path, "toolshare", "v1"), "CON-NOTICE-FORMAT", "contract")
    v11 = _pairs(_ref(tmp_path, "toolshare", "v1.1"), "CON-NOTICE-FORMAT", "contract")
    added = {k for k in set(v11) - set(v1)}
    # CMP-HOLDS is a consumer of CMP-NOTIFY: its two requirements are added, nothing beyond them.
    assert added == {("REQ-HOLD-EXPIRY", "check_contract"), ("REQ-PLACE-HOLD", "check_contract")}
    assert all(len(v11[k].path) == len(v1[("CMP-HOLDS", "check_contract")].path) + 1 for k in added)


def test_replay_client_serves_identical_prompts_and_falls_back_otherwise():
    import replay_lib as rl
    from graph_adlc.llm_graph.source import sha256_text

    class Live:
        provider, api_key_env, calls = "openai", "X", []

        def available(self):
            return True

        def describe(self):
            return {"provider": "openai", "model": "m"}

        def complete(self, system, user, *, call_key, max_tokens=8000):
            self.calls.append(call_key)
            return "live-response"

    rec = {"call_key": "k", "input": {"sha256": sha256_text("S\nU")}, "output": {"text": "recorded"},
           "usage": {"input_tokens": 1}, "latency_ms": 5.0, "stop_reason": "stop", "model_reported": "m"}
    trace = {"events": [{"proposal_step": {"call": rec}}]}
    live = Live()
    c = rl.ReplayClient(rl.recorded_calls(trace), live, "src")
    assert c.complete("S", "U", call_key="k").text == "recorded" and c.describe()["served"] == "replayed"
    assert c.complete("S", "U", call_key="k") == "live-response"      # each recording is used once
    assert c.complete("S", "changed", call_key="k2") == "live-response" and live.calls == ["k", "k2"]
    assert [x["served"] for x in c.served] == ["replayed", "live", "live"]
    strict = rl.ReplayClient(rl.recorded_calls(trace), live, "src", strict=True)
    with pytest.raises(RuntimeError):
        strict.complete("S", "changed", call_key="k")


def test_v12_reruns_tests_of_the_contracts_an_implementation_change_is_constrained_by(tmp_path):
    v11 = _pairs(_ref(tmp_path, "ledgerlite", "v1.1"), "CMP-MONEY", "implementation")
    v12 = _pairs(_ref(tmp_path, "ledgerlite", "v1.2"), "CMP-MONEY", "implementation")
    assert set(v12) - set(v11) == {("TEST-MONEY-FORMAT", "rerun_test")}   # the v0.3 contract-test miss
    rec = v12[("TEST-MONEY-FORMAT", "rerun_test")]
    assert "rules v1.2" in rec.reason and len(rec.path) == 2             # constrained_by, then verifies
    assert ("CON-MONEY-FORMAT", "check_contract") not in v12             # the unchanged contract is not flagged


def test_v12_adds_nothing_to_a_contract_change_and_v1_never_has_the_rule(tmp_path):
    v11 = _pairs(_ref(tmp_path, "toolshare", "v1.1"), "CON-NOTICE-FORMAT", "contract")
    v12 = _pairs(_ref(tmp_path, "toolshare", "v1.2"), "CON-NOTICE-FORMAT", "contract")
    assert set(v12) == set(v11)
    v1 = _pairs(_ref(tmp_path, "toolshare", "v1"), "CMP-NOTIFY", "implementation")
    assert ("TEST-NOTICE-FORMAT", "rerun_test") not in v1
    assert ("TEST-NOTICE-FORMAT", "rerun_test") in _pairs(_ref(tmp_path, "toolshare", "v1.2"), "CMP-NOTIFY",
                                                          "implementation")
