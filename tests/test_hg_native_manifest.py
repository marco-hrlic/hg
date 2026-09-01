"""Run manifest building/writing -- torch-free (falls back gracefully without it)."""

from __future__ import annotations

import json

from hg_native.training.manifest import build_manifest, config_hash, git_is_dirty, read_manifest, write_manifest


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


def test_write_and_read_manifest_round_trip(tmp_path):
    manifest = build_manifest(
        run_id="r1", seed=0, config_dict={"a": 1}, tokenizer_hash="t", model_parameter_count=99, extra={"note": "smoke"}
    )
    path = tmp_path / "sub" / "manifest.json"
    write_manifest(path, manifest)
    assert path.exists()
    restored = read_manifest(path)
    assert restored == manifest
