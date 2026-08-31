"""Reading and writing the artifacts that outlive the code.

The protocol is explicit that "the expensive artifact is the prediction matrix,
not the summary plot" (section 7). So:

* prediction rows go to JSONL, one file per condition, append-safe;
* every run directory carries its resolved config and a manifest;
* the dense matrix form used by the metrics is derived on load, never stored,
  because a dense array cannot record which cells are missing.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml

from .schema import ItemResult, RunManifest, SchemaError, check_minimum_columns


def git_commit(cwd: str | Path = ".") -> str | None:
    """Best-effort commit id for provenance; None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd), capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = out.stdout.strip()
    return commit or None


def run_id(experiment: str, condition: str | None = None) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{stamp}_{experiment}" + (f"_{condition}" if condition else "")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]], append: bool = False) -> int:
    """Write rows as JSONL. Returns the number of rows written.

    Writes through a temp file on a fresh write so an interrupted run leaves
    either the old file or the new one, never a truncated matrix.
    """
    path = Path(path)
    ensure_dir(path.parent)
    n = 0
    if append:
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                n += 1
        return n
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def write_predictions(path: str | Path, rows: Sequence[ItemResult]) -> int:
    for r in rows:
        r.validate()
    return write_jsonl(path, [r.to_dict() for r in rows])


def read_predictions(path: str | Path, strict: bool = True) -> list[ItemResult]:
    """Load a prediction matrix.

    `strict` also enforces the section 14 minimum column set, which catches
    matrices written by an older or hand-rolled producer before any metric
    silently treats the missing column as null.
    """
    out: list[ItemResult] = []
    for row in read_jsonl(path):
        if strict:
            filled = dict(row)
            filled.setdefault("score", None)
            filled.setdefault("confidence", None)
            filled.setdefault("adapter_seed", None)
            filled.setdefault("train_split", None)
            check_minimum_columns(filled)
        out.append(ItemResult.from_dict(row))
    return out


def write_yaml(path: str | Path, data: Mapping[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(_yaml_safe(data), fh, sort_keys=False, default_flow_style=False)


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {k: _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(_yaml_safe(data), fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_manifest(path: str | Path, manifest: RunManifest) -> None:
    if manifest.created_at is None:
        manifest.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if manifest.git_commit is None:
        manifest.git_commit = git_commit()
    write_json(path, manifest.to_dict())


def read_manifest(path: str | Path) -> RunManifest:
    return RunManifest.from_dict(read_json(path))


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    """Small CSV writer for report tables. No pandas dependency by design."""
    import csv

    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(columns) if columns else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class RunDir:
    """Layout of one experiment run under `results/`.

    results/<run_id>/
        config.resolved.yaml     inlined config, self-contained
        manifest.<cond>.json     per-condition provenance
        predictions/<cond>.jsonl item-level rows (the artifact that matters)
        matrices/                derived error matrices, npz
        bootstrap/               resampled metric draws
        figures/                 rendered plots and CSV fallbacks
        report.md                decision-gate summary
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create(self) -> "RunDir":
        for sub in ("predictions", "matrices", "bootstrap", "figures"):
            ensure_dir(self.root / sub)
        return self

    @property
    def config_path(self) -> Path:
        return self.root / "config.resolved.yaml"

    def predictions(self, condition: str) -> Path:
        return self.root / "predictions" / f"{condition}.jsonl"

    def manifest(self, condition: str) -> Path:
        return self.root / f"manifest.{condition}.json"

    def matrix(self, condition: str) -> Path:
        return self.root / "matrices" / f"{condition}.npz"

    def bootstrap(self, name: str) -> Path:
        return self.root / "bootstrap" / f"{name}.json"

    def figure(self, name: str) -> Path:
        return self.root / "figures" / name

    @property
    def report_path(self) -> Path:
        return self.root / "report.md"

    def available_conditions(self) -> list[str]:
        d = self.root / "predictions"
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.jsonl"))
