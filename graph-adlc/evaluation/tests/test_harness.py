"""Regression gate for the v0.2 evaluation harness (HANDOFF.md §7 M4)."""

import builtins
import json
import shutil
from pathlib import Path

import pytest

from graph_adlc.models import Edge, Graph, Node
from graph_adlc_eval import run as run_mod
from graph_adlc_eval.dataset import ScenarioError, file_sha256, load_dataset, load_scenario, project_root
from graph_adlc_eval.document import parse_sections, render_spec
from graph_adlc_eval.normalize import normalize
from graph_adlc_eval.scoring import load_oracle, pruning_audit, score

ROOT = project_root()
DATASET = ROOT / "evaluation" / "scenarios"
ORACLE = ROOT / "evaluation" / "oracle"


@pytest.fixture(scope="module")
def outdir(tmp_path_factory):
    out = tmp_path_factory.mktemp("eval-out")
    assert run_mod.main(["--dataset", str(DATASET), "--out", str(out), "--repeats", "1"]) == 0
    return out


def _preds(out: Path) -> list[dict]:
    return [json.loads(ln) for ln in (out / "predictions.jsonl").read_text().splitlines()]


def _pairs(out: Path, sid: str, method: str) -> set[tuple[str, str]]:
    return {(p["target_id"], p["obligation"]) for p in _preds(out) if p["scenario_id"] == sid and p["method"] == method}


def _raw(out: Path, sid: str, method: str) -> dict:
    return json.loads((out / "raw" / f"{sid}.{method}.json").read_text())


# -- corpus -------------------------------------------------------------------------------

def test_corpus_shape_and_coverage():
    sc = load_dataset(DATASET)
    assert len(sc) == 24
    fam = {f: sum(s.family == f for s in sc) for f in {s.family for s in sc}}
    assert fam == {"semantic": 6, "contract": 6, "implementation": 5, "structure": 4, "adversarial": 3}
    tags = [t for s in sc for t in s.tags]
    assert tags.count("true_negative") >= 3
    assert sum(t.startswith("missing_edge") for t in tags) >= 3
    assert {"evidence_valid", "evidence_failed", "evidence_stale"} <= set(tags)
    assert {"cycle", "multihop", "edge_remove", "hypothesized_edge"} <= set(tags)


def test_every_scenario_has_an_oracle_and_true_negatives_are_empty():
    sc = load_dataset(DATASET)
    oracle = load_oracle(ORACLE, "v1")
    assert set(oracle) == {s.id for s in sc}
    for s in sc:
        if "true_negative" in s.tags:
            assert oracle[s.id]["necessary_checks"] == []


def test_scenario_with_oracle_keys_is_rejected(tmp_path):
    raw = json.loads((DATASET / "cases" / "SC-01.json").read_text())
    raw["necessary_checks"] = []
    p = tmp_path / "leak.json"
    p.write_text(json.dumps(raw))
    with pytest.raises(ScenarioError):
        load_scenario(p)


# -- isolation / leakage / determinism -------------------------------------------------------

def test_prediction_path_never_reads_oracle(tmp_path, monkeypatch):
    real_open = builtins.open
    oracle_str = str(ORACLE)

    def guarded(file, *a, **kw):
        if oracle_str in str(Path(file).resolve()) if isinstance(file, (str, Path)) else False:
            raise AssertionError(f"prediction path opened oracle file {file}")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw: (
        (_ for _ in ()).throw(AssertionError(f"read {self}")) if oracle_str in str(self.resolve())
        else real_open(self, encoding=kw.get("encoding", "utf-8")).read()))
    assert run_mod.main(["--dataset", str(DATASET), "--out", str(tmp_path), "--repeats", "1"]) == 0
    with pytest.raises(AssertionError):  # the guard really does fire on oracle reads
        load_oracle(ORACLE, "v1")


def test_prediction_modules_do_not_reference_the_oracle():
    pkg = ROOT / "evaluation" / "graph_adlc_eval"
    for mod in ("dataset.py", "document.py", "methods.py", "normalize.py", "run.py"):
        code = "\n".join(ln for ln in (pkg / mod).read_text().splitlines() if not ln.lstrip().startswith(("#", '"""')))
        assert "load_oracle" not in code and "oracle/" not in code and "scoring" not in code, mod


def test_runs_are_deterministic(outdir, tmp_path):
    assert run_mod.main(["--dataset", str(DATASET), "--out", str(tmp_path), "--repeats", "1"]) == 0
    for f in ("predictions.jsonl", "excluded_predictions.jsonl", "costs.jsonl", "run_manifest.json"):
        assert (tmp_path / f).read_bytes() == (outdir / f).read_bytes(), f
    for f in sorted((outdir / "raw").iterdir()):
        assert (tmp_path / "raw" / f.name).read_bytes() == f.read_bytes(), f.name


def test_inputs_not_mutated_and_committed_snapshots_current(outdir):
    manifest = json.loads((outdir / "run_manifest.json").read_text())
    for rel, sha in manifest["input_sha256"].items():
        assert file_sha256(ROOT / rel) == sha
    for f in sorted((outdir / "documents").iterdir()):
        assert (DATASET / "documents" / f.name).read_text() == f.read_text(), f"stale snapshot {f.name}"


def test_committed_results_match_a_fresh_run(outdir):
    committed = ROOT / "evaluation" / "out" / "predictions.jsonl"
    assert committed.read_bytes() == (outdir / "predictions.jsonl").read_bytes()


def test_original_fixtures_untouched_by_evaluation(outdir):
    seed = json.loads((ROOT / "fixtures" / "payment_seed.json").read_text())
    assert seed["graph_revision"] == 1 and len(seed["nodes"]) == 12 and len(seed["edges"]) == 16


# -- engine behaviour surfaced through the harness ----------------------------------------------

def test_old_union_new_removed_edge_still_surfaces_checks(outdir):
    b = _raw(outdir, "SC-18", "B")
    reporting = [o for o in b["impact"]["obligations"] if o["target_id"] == "CMP-REPORTING"]
    assert reporting and reporting[0]["path"] == ["E13"]


def test_cycle_terminates_with_unique_obligations(outdir):
    for m in ("B", "C"):
        obs = _raw(outdir, "SC-21", m)["impact"]["obligations"]
        keys = [(o["target_id"], o["obligation"]) for o in obs]
        assert keys and len(keys) == len(set(keys))


def test_evidence_validity_controls_pruning(outdir):
    assert _raw(outdir, "SC-09", "C")["impact"]["pruned"]  # valid -> prunes
    assert _raw(outdir, "SC-15", "C")["impact"]["pruned"]
    assert _raw(outdir, "SC-10", "C")["impact"]["pruned"] == []  # failed -> no prune
    assert _raw(outdir, "SC-11", "C")["impact"]["pruned"] == []  # stale -> no prune
    for sid in ("SC-09", "SC-10", "SC-11", "SC-14", "SC-15", "SC-24"):
        assert _raw(outdir, sid, "B")["impact"]["pruned"] == []  # B never prunes
        assert _raw(outdir, sid, "B")["evidence_refs_passed"] == []
    c24 = _raw(outdir, "SC-24", "C")["impact"]
    assert all(p["consumer_id"] != "CMP-NOTIFIER" for p in c24["pruned"])  # hypothesized edge never pruned
    assert ("CMP-NOTIFIER", "investigate_unknown") in _pairs(outdir, "SC-24", "C")


def test_c_only_ever_removes_relative_to_b(outdir):
    for s in load_dataset(DATASET):
        assert _pairs(outdir, s.id, "C") <= _pairs(outdir, s.id, "B"), s.id


# -- normalization / scoring units --------------------------------------------------------------

def _tiny_graph() -> Graph:
    g = Graph(revision=1)
    for nid, kind in [("CMP-X", "component"), ("ART-X", "artifact"), ("REQ-X", "requirement")]:
        g.nodes[nid] = Node(id=nid, kind=kind, title=nid, data={"path": "x.py"} if kind == "artifact" else {})
    g.edges["E1"] = Edge(id="E1", src="CMP-X", dst="ART-X", relation="realized_by")
    return g


def _p(t, ob, m="B"):
    return {"method": m, "target_id": t, "obligation": ob, "explanation": "x", "path_edge_ids": [],
            "pruned": False, "provisional": False, "confidence": None}


def test_normalization_collapses_dedupes_and_reports_exclusions():
    delta = {"operations": [{"op": "update_node", "id": "REQ-X"}]}
    scored, excluded = normalize("S", [_p("ART-X", "review_implementation"), _p("CMP-X", "review_implementation"),
                                       _p("REQ-X", "review_spec"), _p("CMP-X", "make_coffee")], _tiny_graph(), delta, {})
    assert [(r["target_id"], r["obligation"]) for r in scored] == [("CMP-X", "review_implementation")]
    assert scored[0]["merged_from"] == ["CMP-X"]
    reasons = sorted(e["excluded_reason"] for e in excluded)
    assert reasons == ["target is a directly changed entity", "unmappable obligation"]
    # changed artifact collapses to its component, which is then excluded
    _, ex2 = normalize("S", [_p("CMP-X", "check_contract")], _tiny_graph(),
                       {"operations": [{"op": "update_node", "id": "ART-X"}]}, {})
    assert ex2 and ex2[0]["excluded_reason"].startswith("target is a directly changed")


def test_null_denominators_and_unknown_masking():
    oracle = {"N": {"necessary_checks": [], "unknown_checks": []},
              "U": {"necessary_checks": [{"target_id": "T1", "obligation": "rerun_test", "severity": "major"}],
                    "unknown_checks": [{"target_id": "T2", "obligation": "check_contract"}]}}
    scen = [{"id": "N", "family": "semantic", "tags": []}, {"id": "U", "family": "semantic", "tags": []}]
    res = score([], oracle, scen[:1], ["B"])
    t = res["totals"]["B"]
    assert t["micro_precision"] is None and t["micro_recall"] is None and t["micro_f1"] is None
    assert t["macro_recall_nonempty_gold"] is None and t["negative_cases"]["cases"] == 1
    preds = [{"scenario_id": "U", "method": "B", "target_id": "T2", "obligation": "check_contract"},
             {"scenario_id": "U", "method": "B", "target_id": "T1", "obligation": "rerun_test"}]
    t = score(preds, oracle, scen[1:], ["B"])["totals"]["B"]
    assert (t["tp"], t["fp"], t["fn"], t["unknown_predicted"]) == (1, 0, 0, 1)
    assert t["micro_precision"] == 1.0


def test_pruning_audit_flags_required_removals(outdir):
    oracle = load_oracle(ORACLE, "v1")
    audit = {a["scenario_id"]: a for a in pruning_audit(_preds(outdir), outdir / "raw", oracle,
                                                          [s.id for s in load_dataset(DATASET)])}
    assert [r["target_id"] for r in audit["SC-14"]["pruning_misses"]] == ["CMP-ORDERS"]
    assert audit["SC-09"]["pruning_misses"] == [] and audit["SC-15"]["pruning_misses"] == []


def test_v1_1_review_covers_every_unlabeled_prediction(outdir):
    oracle = load_oracle(ORACLE, "v1")
    reviewed = {(d["scenario_id"], d["target_id"], d["obligation"])
                for d in map(json.loads, (ORACLE / "review_v1_1.jsonl").read_text().splitlines())}
    for p in _preds(outdir):
        o = oracle[p["scenario_id"]]
        labeled = {(c["target_id"], c["obligation"]) for c in o["necessary_checks"] + o["unknown_checks"]}
        if (p["target_id"], p["obligation"]) not in labeled:
            assert (p["scenario_id"], p["target_id"], p["obligation"]) in reviewed, p


# -- document baseline ----------------------------------------------------------------------

def test_document_round_trip_parses_every_node_and_reference():
    g = _tiny_graph()
    doc = render_spec(g, [], "t")
    secs = parse_sections(doc)
    assert set(secs) == {"CMP-X", "ART-X", "REQ-X"}
    assert secs["CMP-X"].kind == "component" and "ART-X" in secs["CMP-X"].refs and "CMP-X" in secs["ART-X"].refs
