"""Scenario schema, loader and isolated per-run environment construction.

Oracle data is deliberately not part of this module: a scenario file must not
contain answer keys, and ``load_scenario`` rejects any that do.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from graph_adlc import GraphADLC

FIXED_TIME = "2026-01-01T00:00:00Z"
REQUIRED_KEYS = {"id", "dataset_version", "family", "tags", "description", "seed_ref", "setup", "proposal"}
FAMILIES = {"semantic", "contract", "implementation", "structure", "adversarial"}
FORBIDDEN_KEYS = {"oracle", "necessary_checks", "material_impacts", "unknown_checks", "expected"}


class ScenarioError(ValueError):
    pass


@dataclass
class Scenario:
    id: str
    family: str
    tags: list[str]
    description: str
    seed_ref: str
    setup: list[dict[str, Any]]
    proposal: dict[str, Any]  # resolved delta
    dataset_version: str
    path: Path
    sha256: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def project_root() -> Path:
    """graph-adlc/ (scenario refs are relative to it)."""
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_scenario(path: Path, root: Path | None = None) -> Scenario:
    root = root or project_root()
    raw = _read_json(path)
    missing = REQUIRED_KEYS - set(raw)
    if missing:
        raise ScenarioError(f"{path}: missing keys {sorted(missing)}")
    leaked = FORBIDDEN_KEYS & set(raw)
    if leaked:
        raise ScenarioError(f"{path}: scenario files must not contain oracle keys {sorted(leaked)}")
    if raw["family"] not in FAMILIES:
        raise ScenarioError(f"{path}: unknown family {raw['family']!r}")
    proposal = raw["proposal"]
    if "delta_ref" in proposal:
        proposal = _read_json(root / proposal["delta_ref"])
    for step in raw["setup"]:
        if step.get("type") not in ("commit", "evidence"):
            raise ScenarioError(f"{path}: unknown setup step {step!r}")
    return Scenario(id=raw["id"], family=raw["family"], tags=list(raw["tags"]), description=raw["description"],
                    seed_ref=raw["seed_ref"], setup=copy.deepcopy(raw["setup"]), proposal=copy.deepcopy(proposal),
                    dataset_version=raw["dataset_version"], path=path, sha256=file_sha256(path), raw=raw)


def load_dataset(dataset_dir: Path, root: Path | None = None) -> list[Scenario]:
    manifest = _read_json(dataset_dir / "manifest.json")
    scenarios = [load_scenario(dataset_dir / rel, root) for rel in manifest["scenarios"]]
    ids = [s.id for s in scenarios]
    if len(ids) != len(set(ids)):
        raise ScenarioError("duplicate scenario ids")
    return scenarios


def input_files(s: Scenario, root: Path | None = None) -> list[Path]:
    """Every file a scenario reads (for hashing and the no-mutation check)."""
    root = root or project_root()
    files = [s.path, root / s.seed_ref]
    if "delta_ref" in s.raw["proposal"]:
        files.append(root / s.raw["proposal"]["delta_ref"])
    files += [root / st["delta_ref"] for st in s.setup if "delta_ref" in st]
    return files


@dataclass
class Environment:
    """A freshly seeded, set-up database for one (scenario, method) run."""

    app: GraphADLC
    db_path: str
    evidence_ids: list[str]
    setup_log: list[dict[str, Any]]


@contextlib.contextmanager
def fixed_clock() -> Iterator[None]:
    old = os.environ.get("GADLC_FIXED_TIME")
    os.environ["GADLC_FIXED_TIME"] = FIXED_TIME
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GADLC_FIXED_TIME", None)
        else:
            os.environ["GADLC_FIXED_TIME"] = old


@contextlib.contextmanager
def environment(s: Scenario, root: Path | None = None) -> Iterator[Environment]:
    """Seed a private temporary SQLite DB and replay the setup steps through the public API."""
    root = root or project_root()
    with fixed_clock(), tempfile.TemporaryDirectory(prefix=f"gadlc-eval-{s.id}-") as tmp:
        db = os.path.join(tmp, "graph.sqlite")
        app = GraphADLC(db)
        try:
            app.seed_graph(_read_json(root / s.seed_ref))
            evidence_ids, log = [], []
            for step in s.setup:
                if step["type"] == "commit":
                    delta = _read_json(root / step["delta_ref"]) if "delta_ref" in step else step["delta"]
                    res = app.commit(copy.deepcopy(delta))
                    log.append({"type": "commit", "change_id": res["change_id"],
                                "graph_revision": res["graph_revision"]})
                else:
                    ev = app.record_evidence(step["node_id"], step["kind"], step["result"],
                                             code_rev=step.get("code_rev"), details=step.get("details"))
                    evidence_ids.append(ev["id"])
                    log.append({"type": "evidence", "evidence_id": ev["id"], "node_id": ev["node_id"],
                                "kind": ev["kind"], "result": ev["result"], "spec_revs": ev["spec_revs"]})
            yield Environment(app=app, db_path=db, evidence_ids=evidence_ids, setup_log=log)
        finally:
            app.close()


def validate_only(s: Scenario) -> dict[str, Any]:
    """Committability check that exposes no impact information (safe before oracle freeze)."""
    with environment(s) as env:
        out = env.app.propose(copy.deepcopy(s.proposal))
        return {"id": s.id, "committable": out["committable"], "errors": out["errors"],
                "conflicts": out["conflicts"], "aspect_overrides": out["aspect_overrides"],
                "evidence": env.evidence_ids}
