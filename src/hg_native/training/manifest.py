"""The run manifest (spec section 16): git commit, dirty-tree status,
hostname/device, CUDA/PyTorch versions, RNG seeds, config hash, tokenizer
hash, data-generator version, model parameter count.

Reuses `hg.io.git_commit`/`ensure_dir` rather than reimplementing them.
Adds `git_is_dirty` and `config_hash`, which spec section 16 requires and
nothing in the `hg` package provides yet (see [[dgx-spark-access]] /
[[hg-v2-project-state]]: no git-dirty-status check or config-hash exists
elsewhere in this repo).

Spec section 16 also bans Python `hash()` for stable cross-process
randomization; `config_hash` uses `hashlib.sha256` over a canonical JSON
dump, not `hash()`, for the same reason `hg_native.data.schema.rng_for` does.
"""

from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hg.io import ensure_dir, git_commit


def git_is_dirty(cwd: str | Path = ".") -> bool | None:
    """True if the working tree has uncommitted changes; None if not in a git checkout."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd), capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def config_hash(config_dict: dict[str, Any]) -> str:
    """A stable digest of a resolved config dict, for provenance and run comparison."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deploy_provenance(cwd: str | Path = ".") -> tuple[str | None, bool | None]:
    """Fallback git SHA / dirty-tree status from plain files.

    `scripts/spark_deploy.sh` rsyncs to the Spark with `.git` excluded (see
    [[dgx-spark-access]]), which leaves `git_commit()`/`git_is_dirty()` with
    nothing to shell out to. The deploy script writes `DEPLOY_GIT_SHA` and
    `DEPLOY_GIT_DIRTY` at the repo root *before* that rsync, so they still
    cross over even though `.git` itself does not; this reads them back.
    Returns `(None, None)` if either file is absent (e.g. a plain local
    checkout, where `git_commit()`/`git_is_dirty()` already work directly).
    """
    root = Path(cwd)
    sha_path = root / "DEPLOY_GIT_SHA"
    dirty_path = root / "DEPLOY_GIT_DIRTY"
    sha = sha_path.read_text(encoding="utf-8").strip() if sha_path.exists() else ""
    dirty: bool | None = None
    if dirty_path.exists():
        raw = dirty_path.read_text(encoding="utf-8").strip().lower()
        if raw in ("true", "false"):
            dirty = raw == "true"
    return (sha or None, dirty)


def data_generator_version() -> str:
    """Spec section 16 wants a "data-generator version"; the generators have
    no separate release scheme, so the commit they live at doubles as it."""
    return git_commit() or "unknown"


def _torch_environment() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_version": None, "cuda_available": None, "cuda_version": None, "device": "cpu"}
    cuda_available = torch.cuda.is_available()
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device": "cuda" if cuda_available else "cpu",
    }


@dataclass(slots=True)
class RunManifest:
    run_id: str
    git_commit: str | None
    git_dirty: bool | None
    hostname: str
    device: str
    torch_version: str | None
    cuda_available: bool | None
    cuda_version: str | None
    seed: int
    config_hash: str
    tokenizer_hash: str
    data_generator_version: str
    model_parameter_count: int
    created_at: str
    extra: dict[str, Any] = field(default_factory=dict)


def build_manifest(
    *,
    run_id: str,
    seed: int,
    config_dict: dict[str, Any],
    tokenizer_hash: str,
    model_parameter_count: int,
    extra: dict[str, Any] | None = None,
) -> RunManifest:
    env = _torch_environment()
    commit = git_commit()
    dirty = git_is_dirty()
    if commit is None or dirty is None:
        fallback_sha, fallback_dirty = deploy_provenance()
        commit = commit if commit is not None else fallback_sha
        dirty = dirty if dirty is not None else fallback_dirty
    return RunManifest(
        run_id=run_id,
        git_commit=commit,
        git_dirty=dirty,
        hostname=socket.gethostname(),
        device=env["device"],
        torch_version=env["torch_version"],
        cuda_available=env["cuda_available"],
        cuda_version=env["cuda_version"],
        seed=seed,
        config_hash=config_hash(config_dict),
        tokenizer_hash=tokenizer_hash,
        data_generator_version=data_generator_version(),
        model_parameter_count=model_parameter_count,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        extra=extra or {},
    )


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(path: str | Path) -> RunManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunManifest(**data)
