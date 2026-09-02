"""Run manifest building/writing -- torch-free (falls back gracefully without it)."""

from __future__ import annotations

import json

from hg_native.training.manifest import build_manifest, config_hash, deploy_provenance, git_is_dirty, read_manifest, write_manifest


def test_config_hash_deterministic_and_order_independent():
    a = {"variant": "hg", "seed": 0, "transformer": {"d_model": 768, "n_layers": 12}}
    b = {"seed": 0, "transformer": {"n_layers": 12, "d_model": 768}, "variant": "hg"}
    assert config_hash(a) == config_hash(b)


def test_config_hash_differs_for_different_configs():
    a = {"seed": 0}
    b = {"seed": 1}
    assert config_hash(a) != config_hash(b)


def test_config_hash_is_not_pythons_hash():
    # A real regression here would only show up across processes with
    # different PYTHONHASHSEED; assert it disagrees with this process's
    # hash() directly, catching an accidental `hash(...)` implementation.
    d = {"seed": 0}
    assert config_hash(d) != str(hash(json.dumps(d, sort_keys=True)))


def test_git_is_dirty_returns_a_bool_or_none():
    # A bool inside a git checkout; None off a plain rsync mirror with no
    # .git (e.g. scripts/spark_deploy.sh's sync target) -- both legitimate,
    # so accept either rather than assuming this process runs inside git.
    assert git_is_dirty() in (True, False, None)


def test_build_manifest_has_every_spec_section_16_field():
    manifest = build_manifest(
        run_id="20260101T000000Z_test",
        seed=3,
        config_dict={"variant": "baseline"},
        tokenizer_hash="ByteTokenizer:v259",
        model_parameter_count=12345,
    )
    assert manifest.run_id == "20260101T000000Z_test"
    assert manifest.seed == 3
    assert manifest.tokenizer_hash == "ByteTokenizer:v259"
    assert manifest.model_parameter_count == 12345
    assert manifest.device in ("cpu", "cuda")
    # A real commit hash inside a git checkout; None off a plain rsync
    # mirror with no .git -- both legitimate (see git_is_dirty's test above).
    assert manifest.git_commit is None or len(manifest.git_commit) == 40
    assert manifest.git_dirty in (True, False, None)
    assert manifest.created_at.endswith("Z")


# --- deploy_provenance (kickoff section 17's provenance fix) ---


def test_deploy_provenance_absent_when_no_files(tmp_path):
    assert deploy_provenance(tmp_path) == (None, None)


def test_deploy_provenance_reads_sha_and_dirty_files(tmp_path):
    (tmp_path / "DEPLOY_GIT_SHA").write_text("abc123\n", encoding="utf-8")
    (tmp_path / "DEPLOY_GIT_DIRTY").write_text("true\n", encoding="utf-8")
    assert deploy_provenance(tmp_path) == ("abc123", True)


def test_deploy_provenance_dirty_false(tmp_path):
    (tmp_path / "DEPLOY_GIT_SHA").write_text("abc123", encoding="utf-8")
    (tmp_path / "DEPLOY_GIT_DIRTY").write_text("false", encoding="utf-8")
    assert deploy_provenance(tmp_path) == ("abc123", False)


def test_deploy_provenance_ignores_malformed_dirty_file(tmp_path):
    (tmp_path / "DEPLOY_GIT_SHA").write_text("abc123", encoding="utf-8")
    (tmp_path / "DEPLOY_GIT_DIRTY").write_text("maybe", encoding="utf-8")
    sha, dirty = deploy_provenance(tmp_path)
    assert sha == "abc123"
    assert dirty is None


def test_build_manifest_falls_back_to_deploy_provenance_outside_git(tmp_path, monkeypatch):
    # Simulate a Spark deploy target: no .git (git_commit/git_is_dirty both
    # return None there), but DEPLOY_GIT_SHA/DEPLOY_GIT_DIRTY present.
    (tmp_path / "DEPLOY_GIT_SHA").write_text("deadbeef", encoding="utf-8")
    (tmp_path / "DEPLOY_GIT_DIRTY").write_text("true", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("hg_native.training.manifest.git_commit", lambda: None)
    monkeypatch.setattr("hg_native.training.manifest.git_is_dirty", lambda: None)
    manifest = build_manifest(
        run_id="r", seed=0, config_dict={}, tokenizer_hash="t", model_parameter_count=1
    )
    assert manifest.git_commit == "deadbeef"
    assert manifest.git_dirty is True


def test_write_and_read_manifest_round_trip(tmp_path):
    manifest = build_manifest(
        run_id="r1", seed=0, config_dict={"a": 1}, tokenizer_hash="t", model_parameter_count=99, extra={"note": "smoke"}
    )
    path = tmp_path / "sub" / "manifest.json"
    write_manifest(path, manifest)
    assert path.exists()
    restored = read_manifest(path)
    assert restored == manifest
