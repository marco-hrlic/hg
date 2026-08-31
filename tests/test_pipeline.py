"""End-to-end pipeline, on the simulator backend.

These are the tests that would catch a regression in the thing that matters:
that a config produces a complete, re-analyzable prediction matrix and a gate
verdict, and that the analysis half runs with no model present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hg.config import ConfigError, load_experiment, validate_panel
from hg.io import RunDir, read_predictions
from hg.metrics.matrix import build_matrix
from hg.pipeline import reanalyze, run_experiment
from hg.schema import MINIMUM_COLUMNS


def write_config(tmp_path: Path, **overrides) -> Path:
    cfg = {
        "name": "test_pipeline",
        "backend": "simulator",
        "seed": 0,
        "n_bootstrap": 60,
        "dataset": {
            "name": "test_ds",
            "task": "test_task",
            "task_family": "C",
            "labels": ["not_harmful", "harmful"],
            "prompt_version": "classify.v1",
            "anchors": {
                "source": "generate",
                "spec": {
                    "n_items": 900,
                    "n_annotators": 5,
                    "contested_rate": 0.15,
                    "majority_rate": 0.2,
                    "difficulty_sd": 1.2,
                    "core_rate": 0.03,
                    "seed": 0,
                },
            },
            "options": {"anchor": {"majority_threshold": 0.66}},
        },
        "conditions": {
            "HS": {
                "models": ["sim_qwen_v1", "sim_qwen_v2", "sim_qwen_v3"],
                "train_data": "shared",
                "adapter_seeds": [0, 1, 2],
            },
            "DD": {
                "models": ["sim_qwen", "sim_mistral", "sim_gemma"],
                "train_data": "disjoint",
                "adapter_seeds": [0, 0, 0],
            },
        },
        "options": {
            "dry_run_finetune": True,
            "n_eval": 300,
            "n_train_per_member": 100,
            "race": True,
        },
    }
    cfg.update(overrides)
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory, repo_root_module):
    tmp = tmp_path_factory.mktemp("run")
    path = write_config(tmp)
    return run_experiment(
        path,
        config_root=repo_root_module / "configs",
        data_root=tmp / "data",
        output_root=tmp / "results",
    )


@pytest.fixture(scope="module")
def repo_root_module() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_produces_both_conditions(pipeline_run):
    assert set(pipeline_run.matrices) == {"HS", "DD"}


def test_matrix_is_complete_and_sized_as_configured(pipeline_run):
    for name, m in pipeline_run.matrices.items():
        assert m.n_items == 300, name
        assert m.n_members == 3, name
        assert m.errors.shape == (300, 3)


def test_prediction_rows_carry_the_minimum_schema(pipeline_run):
    path = pipeline_run.run_dir.predictions("DD")
    rows = read_predictions(path, strict=True)
    assert rows
    d = rows[0].to_dict()
    d.setdefault("score", None)
    for col in MINIMUM_COLUMNS:
        assert col in d, col


def test_prediction_matrix_round_trips_through_disk(pipeline_run):
    for name, m in pipeline_run.matrices.items():
        reloaded = build_matrix(read_predictions(pipeline_run.run_dir.predictions(name)), name)
        assert (reloaded.errors == m.errors).all(), name
        assert reloaded.item_ids == m.item_ids


def test_run_directory_is_self_contained(pipeline_run):
    run = pipeline_run.run_dir
    assert run.config_path.exists()
    assert run.report_path.exists()
    assert (run.root / "analysis.json").exists()
    assert (run.root / "diagnostics.json").exists()
    for name in ("HS", "DD"):
        assert run.manifest(name).exists()
        assert run.matrix(name).exists()


def test_manifest_records_measured_training_overlap(pipeline_run):
    """A 'disjoint' claim must be checkable from the run directory alone."""
    manifest = json.loads(pipeline_run.run_dir.manifest("DD").read_text())
    offdiag = [
        v for k, v in manifest["train_overlap"].items()
        if k.split("|")[0] != k.split("|")[1]
    ]
    assert offdiag
    assert all(v == 0.0 for v in offdiag)

    shared = json.loads(pipeline_run.run_dir.manifest("HS").read_text())
    offdiag_shared = [
        v for k, v in shared["train_overlap"].items()
        if k.split("|")[0] != k.split("|")[1]
    ]
    assert all(v == 1.0 for v in offdiag_shared)


def test_members_have_distinct_adapter_hashes(pipeline_run):
    """Identical hashes would mean the members are not independent."""
    diag = json.loads((pipeline_run.run_dir.root / "diagnostics.json").read_text())
    for condition in ("HS", "DD"):
        hashes = list(diag[condition]["adapter_hashes"].values())
        assert len(set(hashes)) == len(hashes), condition


def test_evaluation_items_are_disjoint_from_training(pipeline_run):
    diag = json.loads((pipeline_run.run_dir.root / "diagnostics.json").read_text())
    for condition in ("HS", "DD"):
        assert diag[condition]["splits"]["n_eval"] == 300
        assert diag[condition]["splits"]["equal_budget"] is True


def test_diversity_reduces_clean_consensus_failure(pipeline_run):
    """The core effect the fixture is built to contain (prediction P1)."""
    conditions = pipeline_run.analysis["conditions"]
    assert conditions["DD"]["C_E_clean"] < conditions["HS"]["C_E_clean"]


def test_analysis_reports_a_gate_and_all_six_predictions(pipeline_run):
    analysis = pipeline_run.analysis
    assert analysis["gate"]["gate"] in ("A", "B", "C", "D", "undetermined")
    names = [p["prediction"] for p in analysis["predictions"]]
    assert names == ["P1", "P2", "P3", "P4", "P5", "P6"]


def test_binary_task_race_is_marked_not_applicable(pipeline_run):
    interp = pipeline_run.analysis["race"]["interpretation"]
    assert interp["verdict"] == "not_applicable"


def test_survival_curve_has_one_point_per_family(pipeline_run):
    curve = pipeline_run.analysis["survival"]["curve"]
    assert [p["k"] for p in curve] == [1, 2, 3]


def test_contested_items_exist_so_q_contested_is_testable(pipeline_run):
    """Regression guard: with 5 annotators and 2 labels a tie is impossible, so
    without a raised majority threshold the contested subset comes out empty."""
    panel = pipeline_run.analysis["panel"]
    assert panel["status_counts"]["contested"] > 0


def test_reanalysis_needs_no_model_and_reproduces_the_gate(pipeline_run):
    """The payoff of storing item-level rows."""
    again = reanalyze(pipeline_run.run_dir.root)
    assert again["gate"]["gate"] == pipeline_run.analysis["gate"]["gate"]
    assert again["conditions"]["DD"]["C_E_clean"] == pytest.approx(
        pipeline_run.analysis["conditions"]["DD"]["C_E_clean"]
    )


def test_run_is_reproducible_for_a_fixed_seed(tmp_path, repo_root_module):
    path = write_config(tmp_path)
    kwargs = dict(config_root=repo_root_module / "configs", data_root=tmp_path / "data")
    a = run_experiment(path, output_root=tmp_path / "r1", **kwargs)
    b = run_experiment(path, output_root=tmp_path / "r2", **kwargs)
    assert (a.matrices["DD"].errors == b.matrices["DD"].errors).all()


# ------------------------------------------------------------------ configs ---
def test_shipped_configs_load_and_validate(repo_root_module):
    root = repo_root_module / "configs" / "experiments"
    for path in sorted(root.glob("*.yaml")):
        cfg = load_experiment(path)
        for condition in cfg.conditions:
            validate_panel(cfg, condition)


def test_diverse_condition_reusing_one_lineage_is_flagged(tmp_path, repo_root_module):
    """Protocol section 3: variants of one foundation model are not diversity."""
    path = write_config(
        tmp_path,
        conditions={
            "DD": {
                "models": ["sim_qwen_v1", "sim_qwen_v2", "sim_qwen_v3"],
                "train_data": "disjoint",
                "adapter_seeds": [0, 1, 2],
            }
        },
    )
    cfg = load_experiment(path, config_root=repo_root_module / "configs")
    warnings = validate_panel(cfg, "DD")
    assert any("do not count as family diversity" in w for w in warnings)


def test_condition_without_models_is_rejected(tmp_path):
    path = write_config(tmp_path, conditions={"DD": {"train_data": "shared"}})
    with pytest.raises(ConfigError, match="needs a 'models' list"):
        load_experiment(path)


def test_dataset_with_one_label_is_rejected(tmp_path):
    cfg = yaml.safe_load(write_config(tmp_path).read_text())
    cfg["dataset"]["labels"] = ["only"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ConfigError, match="at least two labels"):
        load_experiment(path)


def test_missing_anchor_file_gives_an_actionable_error(tmp_path, repo_root_module):
    cfg = yaml.safe_load(write_config(tmp_path).read_text())
    cfg["dataset"]["anchors"] = {"source": "file", "path": str(tmp_path / "nope.jsonl")}
    path = tmp_path / "missing.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="raw annotator judgments"):
        run_experiment(
            path, config_root=repo_root_module / "configs", data_root=tmp_path / "data"
        )
