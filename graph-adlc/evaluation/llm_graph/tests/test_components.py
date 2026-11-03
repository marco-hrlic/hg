"""Component tests for the v0.3 layer (M1/M2 contract checks) and harness guarantees."""

from __future__ import annotations

import builtins
import http.server
import json
import pathlib
import threading

import pytest
from helpers import HERE, REQS_V1, bundle, cite, loop_with, new_app, op_edge, op_node, seed_response

from graph_adlc.llm_graph import (AnthropicMessagesClient, LLMNotConfigured, ProposalFormatError, SourceBundle,
                                  load_prompt, pack_sources, parse_proposal)


def _seed(tmp_path):
    app = new_app(tmp_path)
    loop = loop_with(app, {"seed": seed_response()})
    b = bundle(REQS_V1)
    st = loop.propose(b, {}, call_key="seed")
    loop.review(st, [__import__("graph_adlc.llm_graph", fromlist=["Decision"]).Decision(o.op_id, "accept", "a")
                     for o in st.proposal.operations], b)
    return app, b


def _codes(step, op="op-1"):
    return {f["code"] for f in step.report["ops"][op]["findings"]}


def test_unsupported_relation_and_invalid_endpoints(tmp_path):
    app, b = _seed(tmp_path)
    ev = [cite(REQS_V1, "Holds on the same tool are served first come, first served.")]
    l = loop_with(app, {"k": {"intent": "x", "operations": [
        op_edge("op-1", "CAP-HOLDS", "influences", "REQ-PLACE-HOLD", ev),
        op_edge("op-2", "REQ-PLACE-HOLD", "implements", "CAP-HOLDS", ev),
        op_edge("op-3", "CAP-HOLDS", "depends_on", "CMP-NOWHERE", ev)]}})
    st = l.propose(b, {}, call_key="k")
    assert "unsupported_relation" in _codes(st) and st.report["ops"]["op-1"]["verdict"] == "invalid"
    assert "invalid_endpoints" in _codes(st, "op-2") and "unknown_endpoint" in _codes(st, "op-3")


def test_duplicate_ids_and_revision_conflicts(tmp_path):
    app, b = _seed(tmp_path)
    ev = [cite(REQS_V1, "Holds on the same tool are served first come, first served.")]
    l = loop_with(app, {"k": {"intent": "x", "operations": [
        op_node("op-1", "REQ-PLACE-HOLD", "requirement", "Something else entirely", {"acceptance_criteria": []}, ev),
        op_node("op-2", "REQ-Z", "requirement", "Z", {"acceptance_criteria": []}, ev),
        op_node("op-3", "REQ-Z", "requirement", "Z2", {"acceptance_criteria": []}, ev),
        {"op_id": "op-4", "kind": "update_node", "id": "REQ-PLACE-HOLD", "expected_rev": 7, "aspect": "semantic",
         "patch": {"body": "x"}, "support": "supported", "source_evidence": ev}]}})
    st = l.propose(b, {}, call_key="k")
    assert "id_exists" in _codes(st)
    assert st.report["ops"]["op-3"]["verdict"] == "invalid"  # engine rejects the second add of REQ-Z
    assert "revision_conflict" in _codes(st, "op-4")


def test_stale_base_is_reported_and_rebased_only_when_safe(tmp_path):
    app, b = _seed(tmp_path)
    ev = [cite(REQS_V1, "Holds on the same tool are served first come, first served.")]
    l = loop_with(app, {"k": {"intent": "x", "operations": [op_node("op-1", "REQ-NEW", "requirement", "New req",
                                                                    {"acceptance_criteria": ["a"]}, ev)]}})
    st = l.propose(b, {}, call_key="k")
    app.commit({"base_graph_revision": 2, "actor": "h", "rationale": "unrelated", "operations": [
        {"op": "add_node", "node": {"id": "CAP-OTHER", "kind": "capability", "title": "Other", "data": {}}}]})
    from graph_adlc.llm_graph import Decision
    res = l.review(st, [Decision("op-1", "accept", "alice")], b)
    assert res["rebased"] is True and app.graph().node("REQ-NEW") is not None
    assert "(rebased)" in app.log()[-1]["actor"]


def test_proposal_format_errors():
    with pytest.raises(ProposalFormatError):
        parse_proposal({"operations": [{"kind": "explode"}]}, "P", 1)
    with pytest.raises(ProposalFormatError):
        parse_proposal({"operations": [{"kind": "add_node", "support": "certain"}]}, "P", 1)
    p = parse_proposal({"operations": [{"kind": "add_node", "node": {"id": "X"}, "confidence": 0.99}]}, "P", 1)
    assert p.operations[0].support == "insufficient"  # never inferred from confidence


def test_unparseable_model_output_is_recorded_not_applied(tmp_path):
    app, b = _seed(tmp_path)
    before = json.dumps(app.export(), sort_keys=True)
    l = loop_with(app, {"k": "I think you should add REQ-FOO. ```not json```"})
    st = l.propose(b, {}, call_key="k")
    assert st.proposal is None and st.errors and json.dumps(app.export(), sort_keys=True) == before
    assert l.ledger.proposals()[-1]["status"] == "invalid"


def test_packer_is_deterministic_and_discloses_truncation():
    b = SourceBundle(project="t", documents=[REQS_V1])
    p1, p2 = pack_sources(b, max_chars_per_doc=400), pack_sources(b, max_chars_per_doc=400)
    assert p1 == p2 and p1["truncation"][0]["source_id"] == REQS_V1.source_id
    assert "[TRUNCATED: showing lines 1-" in p1["text"] and REQS_V1.sha256 in p1["text"]
    assert pack_sources(b)["truncation"] == []


def test_live_client_requires_explicit_model_and_never_records_the_key(monkeypatch):
    with pytest.raises(LLMNotConfigured):
        AnthropicMessagesClient(model="")
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["key"] = self.headers.get("x-api-key")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["content-length"])))
            out = json.dumps({"model": "test-model", "content": [{"type": "text", "text": '{"operations": []}'}],
                              "usage": {"input_tokens": 11, "output_tokens": 3}, "stop_reason": "end_turn"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GADLC_TEST_KEY", "sk-test-secret")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    from graph_adlc.llm_graph import LLMProposer
    from graph_adlc.models import Graph
    c = AnthropicMessagesClient(model="test-model", api_key_env="GADLC_TEST_KEY",
                                base_url=f"http://127.0.0.1:{srv.server_port}")
    prop, record = LLMProposer(c).propose_update(Graph(revision=1), bundle(REQS_V1), {"kind": "t"},
                                                 proposal_id="P-1", call_key="k")
    srv.shutdown()
    assert seen["key"] == "sk-test-secret" and seen["body"]["model"] == "test-model"
    assert "Untrusted" in seen["body"]["system"] or "untrusted" in seen["body"]["system"]
    assert prop is not None and record["usage"] == {"input_tokens": 11, "output_tokens": 3}
    assert "sk-test-secret" not in json.dumps(record) and record["is_fake"] is False
    assert record["prompt"]["sha256"] == load_prompt("propose_update.v1")["sha256"]


def test_prompts_are_frozen():
    lock = json.loads((HERE / "prompts" / "PROMPTS.lock.json").read_text())["prompts"]
    assert {k: load_prompt(k)["sha256"] for k in lock} == lock


def test_prediction_never_reads_the_oracle(monkeypatch, tmp_path):
    import run_replay
    real_open, real_read = builtins.open, pathlib.Path.read_text

    def guard(path):
        if "oracle" in pathlib.Path(str(path)).parts:
            raise AssertionError(f"oracle read during prediction: {path}")

    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (guard(f), real_open(f, *a, **k))[1])
    monkeypatch.setattr(pathlib.Path, "read_text", lambda self, *a, **k: (guard(self), real_read(self, *a, **k))[1])
    assert run_replay.main(["--mode", "scripted", "--out", str(tmp_path / "o")]) == 0


def test_committed_outputs_are_current(scripted_run):
    committed = HERE / "out" / "scripted"
    for f in sorted(scripted_run.rglob("*")):
        if f.is_file() and f.name != "metrics.json":
            assert (committed / f.relative_to(scripted_run)).read_bytes() == f.read_bytes(), f"stale: {f.name}"


def test_openai_client_shape_and_no_key_in_records(monkeypatch):
    from graph_adlc.llm_graph import LLMProposer, OpenAIChatClient, client_from_env
    from graph_adlc.models import Graph
    with pytest.raises(LLMNotConfigured):
        OpenAIChatClient(model="")
    seen = {"calls": 0}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            seen["calls"] += 1
            seen["auth"], seen["path"] = self.headers.get("authorization"), self.path
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen["body"] = body
            if "temperature" in body:  # emulate a model that only accepts its default temperature
                out, code = b'{"error": {"message": "Unsupported value: temperature"}}', 400
            else:
                out, code = json.dumps({"model": "m", "choices": [{"message": {"content": '{"operations": []}'},
                                                                   "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 9, "completion_tokens": 2}}).encode(), 200
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for k in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(k, "127.0.0.1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GADLC_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("GADLC_LLM_API_KEY_ENV", raising=False)
    monkeypatch.delenv("GADLC_LLM_MAX_TOKENS", raising=False)
    monkeypatch.setenv("GADLC_LLM_MODEL", "m")
    monkeypatch.setenv("GADLC_LLM_BASE_URL", f"http://127.0.0.1:{srv.server_port}")
    c = client_from_env()
    assert isinstance(c, OpenAIChatClient) and c.available()
    prop, record = LLMProposer(c).propose_update(Graph(revision=1), bundle(REQS_V1), {"kind": "t"},
                                                 proposal_id="P-1", call_key="k")
    srv.shutdown()
    assert seen["path"] == "/v1/chat/completions" and seen["auth"] == "Bearer sk-openai-secret"
    assert seen["body"]["response_format"] == {"type": "json_object"} and seen["calls"] == 2
    assert prop is not None and record["usage"] == {"input_tokens": 9, "output_tokens": 2}
    assert record["client"]["temperature"] is None  # the fallback is recorded
    assert "sk-openai-secret" not in json.dumps(record)
    assert seen["body"]["max_completion_tokens"] == 8000 and record["client"]["max_output_tokens"] is None


def test_max_tokens_override_is_explicit_sent_and_recorded(monkeypatch):
    from graph_adlc.llm_graph import AnthropicMessagesClient, OpenAIChatClient, client_from_env
    monkeypatch.setenv("GADLC_LLM_MODEL", "m")
    for provider, cls in (("openai", OpenAIChatClient), ("anthropic", AnthropicMessagesClient)):
        monkeypatch.setenv("GADLC_LLM_PROVIDER", provider)
        monkeypatch.setenv("GADLC_LLM_MAX_TOKENS", "32000")
        c = client_from_env()
        assert isinstance(c, cls) and c.describe()["max_output_tokens"] == 32000
        monkeypatch.setenv("GADLC_LLM_MAX_TOKENS", "lots")
        with pytest.raises(LLMNotConfigured):
            client_from_env()


def test_policy_defers_edges_whose_new_endpoint_is_not_accepted():
    # Live run 2 crashed here: admissible auto `blocks` edge, quarantined auto question node (toolshare/G1, P-0004).
    from types import SimpleNamespace as NS

    from graph_adlc.llm_graph.review import check_decisions, policy_reviewer
    ops = [NS(op_id="auto-q1", kind="add_node", payload={"node": {"id": "Q-P-0004-C1"}}),
           NS(op_id="auto-q1-b1", kind="add_edge", payload={"edge": {"src": "Q-P-0004-C1", "dst": "CMP-X"}}),
           NS(op_id="op-1", kind="add_edge", payload={"edge": {"src": "CMP-X", "dst": "CMP-Y"}})]
    report = {"ops": {"auto-q1": {"verdict": "quarantined", "findings": []},
                      "auto-q1-b1": {"verdict": "admissible", "findings": []},
                      "op-1": {"verdict": "admissible", "findings": []}}}
    proposal = NS(proposal_id="P-0004", operations=ops)
    ds = policy_reviewer()(proposal, report)
    assert [d.decision for d in ds] == ["defer", "defer", "accept"]
    assert check_decisions(proposal, report, ds) == []
