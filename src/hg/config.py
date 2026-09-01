"""Config resolution.

Protocol section 14: "The experiment should be reproducible from configuration
files rather than notebook state." So configs are the only input to a run, and
the resolved config is written into the run directory verbatim.

Three layers, each a YAML file under `configs/`:

    models/<name>.yaml       one ensemble member family + checkpoint + backend
    datasets/<name>.yaml     task, label space, anchor source, split policy
    experiments/<name>.yaml   conditions, panel composition, seeds, metrics

An experiment config references the other two by name. `load_experiment`
inlines them, so what lands in `results/<run>/config.resolved.yaml` is
self-contained and does not depend on the repo state at analysis time.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_CONFIG_ROOT = Path("configs")


class ConfigError(ValueError):
    pass


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} in strings so paths and model roots stay machine-local."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(copy.deepcopy(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_config_file(path: str | Path) -> dict[str, Any]:
    """Read one YAML config file with ${VAR} expansion.

    The shared primitive behind `load_experiment` and the HG-v2 grounding
    config loader, exposed so a second experiment family does not have to
    re-implement reading-and-expanding (or reach into a private helper).
    """
    return _expand_env(_read_yaml(Path(path)))


@dataclass(slots=True)
class ModelConfig:
    """One model family in the diversity panel (protocol section 3)."""

    name: str
    family: str
    checkpoint: str
    backend: str = "simulator"
    # Lineage groups families that share a training ancestry. Two members with
    # the same lineage do NOT count as family diversity (section 3's warning
    # about five variants of one foundation model), and the panel validator
    # enforces that for diverse conditions.
    lineage: str | None = None
    params_b: float | None = None
    dtype: str = "bfloat16"
    max_new_tokens: int = 8
    # Backend-specific knobs (device_map, tensor_parallel_size, simulator
    # skill/bias parameters, ...).
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.lineage is None:
            self.lineage = self.family

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "ModelConfig":
        d = dict(d)
        d.pop("name", None)
        known = {
            "family", "checkpoint", "backend", "lineage",
            "params_b", "dtype", "max_new_tokens", "options",
        }
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"model {name}: unknown keys {sorted(unknown)}")
        if "family" not in d or "checkpoint" not in d:
            raise ConfigError(f"model {name}: 'family' and 'checkpoint' are required")
        return cls(name=name, **d)  # type: ignore[arg-type]


@dataclass(slots=True)
class DatasetConfig:
    """A task, its label space, and where its human anchors come from."""

    name: str
    task: str
    # One of protocol section 5.2's task families: A/B/C/D/E.
    task_family: str
    labels: list[str]
    prompt_version: str = "v1"
    # Anchor source: a JSONL panel file, or a generator spec for the pilot.
    anchors: dict[str, Any] = field(default_factory=dict)
    n_eval: int | None = None
    n_train_per_member: int | None = None
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "DatasetConfig":
        d = dict(d)
        d.pop("name", None)
        known = {
            "task", "task_family", "labels", "prompt_version",
            "anchors", "n_eval", "n_train_per_member", "options",
        }
        unknown = set(d) - known
        if unknown:
            raise ConfigError(f"dataset {name}: unknown keys {sorted(unknown)}")
        for required in ("task", "task_family", "labels"):
            if required not in d:
                raise ConfigError(f"dataset {name}: {required!r} is required")
        if len(d["labels"]) < 2:
            raise ConfigError(f"dataset {name}: need at least two labels")
        return cls(name=name, **d)  # type: ignore[arg-type]


@dataclass(slots=True)
class ConditionConfig:
    """One of HS / HD / DS / DD (protocol section 4)."""

    name: str
    # Member model config names, length E. For homogeneous conditions these
    # repeat one family with different adapter seeds.
    models: list[str]
    # "shared": every member trains on the same slice.
    # "disjoint": members get non-overlapping slices.
    train_data: str = "shared"
    adapter_seeds: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.train_data not in ("shared", "disjoint"):
            raise ConfigError(
                f"condition {self.name}: train_data must be 'shared' or 'disjoint'"
            )
        if not self.adapter_seeds:
            self.adapter_seeds = list(range(len(self.models)))
        if len(self.adapter_seeds) != len(self.models):
            raise ConfigError(
                f"condition {self.name}: {len(self.models)} models but "
                f"{len(self.adapter_seeds)} adapter seeds"
            )


@dataclass(slots=True)
class ExperimentConfig:
    name: str
    dataset: DatasetConfig
    conditions: dict[str, ConditionConfig]
    models: dict[str, ModelConfig]
    seed: int = 0
    n_bootstrap: int = 2000
    output_root: Path = Path("results")
    # Backend override applied to every member, e.g. force "simulator" in CI.
    backend: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def ensemble_size(self, condition: str) -> int:
        return len(self.conditions[condition].models)

    def member_models(self, condition: str) -> list[ModelConfig]:
        return [self.models[m] for m in self.conditions[condition].models]


def load_experiment(
    path: str | Path,
    config_root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load an experiment config and inline its model and dataset references."""
    path = Path(path)
    root = Path(config_root) if config_root else path.parent.parent
    raw = _expand_env(_read_yaml(path))
    if overrides:
        raw = _merge(raw, overrides)

    name = raw.get("name") or path.stem

    ds_ref = raw.get("dataset")
    if not ds_ref:
        raise ConfigError(f"{path}: 'dataset' is required")
    if isinstance(ds_ref, str):
        ds_raw = _expand_env(_read_yaml(root / "datasets" / f"{ds_ref}.yaml"))
        ds_name = ds_ref
    else:
        ds_raw, ds_name = dict(ds_ref), ds_ref.get("name", "inline")
    dataset = DatasetConfig.from_dict(ds_name, ds_raw)

    conditions: dict[str, ConditionConfig] = {}
    for cond_name, cond_raw in (raw.get("conditions") or {}).items():
        if not isinstance(cond_raw, Mapping) or "models" not in cond_raw:
            raise ConfigError(f"{path}: condition {cond_name} needs a 'models' list")
        conditions[cond_name] = ConditionConfig(
            name=cond_name,
            models=list(cond_raw["models"]),
            train_data=cond_raw.get("train_data", "shared"),
            adapter_seeds=list(cond_raw.get("adapter_seeds", [])),
        )
    if not conditions:
        raise ConfigError(f"{path}: at least one condition is required")

    # Load every model referenced by any condition, plus any inline overrides.
    inline_models = raw.get("models") or {}
    models: dict[str, ModelConfig] = {}
    referenced = {m for c in conditions.values() for m in c.models}
    for m in sorted(referenced):
        if isinstance(inline_models.get(m), Mapping):
            m_raw = _expand_env(dict(inline_models[m]))
        else:
            m_raw = _expand_env(_read_yaml(root / "models" / f"{m}.yaml"))
        models[m] = ModelConfig.from_dict(m, m_raw)

    backend = raw.get("backend")
    if backend:
        for mc in models.values():
            mc.backend = backend

    return ExperimentConfig(
        name=name,
        dataset=dataset,
        conditions=conditions,
        models=models,
        seed=int(raw.get("seed", 0)),
        n_bootstrap=int(raw.get("n_bootstrap", 2000)),
        output_root=Path(raw.get("output_root", "results")),
        backend=backend,
        options=dict(raw.get("options") or {}),
        raw=raw,
    )


def resolved_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """The fully-inlined config, for writing into the run directory."""
    return {
        "name": cfg.name,
        "seed": cfg.seed,
        "n_bootstrap": cfg.n_bootstrap,
        "backend": cfg.backend,
        "options": cfg.options,
        "dataset": {
            "name": cfg.dataset.name,
            "task": cfg.dataset.task,
            "task_family": cfg.dataset.task_family,
            "labels": cfg.dataset.labels,
            "prompt_version": cfg.dataset.prompt_version,
            "anchors": cfg.dataset.anchors,
            "n_eval": cfg.dataset.n_eval,
            "n_train_per_member": cfg.dataset.n_train_per_member,
            "options": cfg.dataset.options,
        },
        "conditions": {
            name: {
                "models": c.models,
                "train_data": c.train_data,
                "adapter_seeds": c.adapter_seeds,
            }
            for name, c in cfg.conditions.items()
        },
        "models": {
            name: {
                "family": m.family,
                "checkpoint": m.checkpoint,
                "backend": m.backend,
                "lineage": m.lineage,
                "params_b": m.params_b,
                "dtype": m.dtype,
                "max_new_tokens": m.max_new_tokens,
                "options": m.options,
            }
            for name, m in cfg.models.items()
        },
    }


def validate_panel(cfg: ExperimentConfig, condition: str) -> list[str]:
    """Check a condition's panel against the protocol's diversity requirement.

    Returns a list of warnings rather than raising: a deliberately homogeneous
    condition (HS, HD) is *supposed* to reuse one lineage. The check matters
    for DS/DD, where reusing a lineage would silently answer a different
    question than the one being asked (section 3).
    """
    warnings: list[str] = []
    cond = cfg.conditions[condition]
    models = cfg.member_models(condition)
    lineages = [m.lineage for m in models]
    distinct = len(set(lineages))

    diverse = condition in ("DS", "DD")
    if diverse and distinct < len(models):
        warnings.append(
            f"{condition}: {len(models)} members but only {distinct} distinct "
            f"lineages {sorted(set(lineages))} -- variants of one foundation "
            "model do not count as family diversity (protocol section 3)"
        )
    if not diverse and distinct > 1:
        warnings.append(
            f"{condition}: homogeneous condition mixes lineages "
            f"{sorted(set(lineages))}; the HS/HD baseline should hold family fixed"
        )
    if not diverse and len(set(cond.adapter_seeds)) < len(cond.adapter_seeds):
        warnings.append(
            f"{condition}: repeated adapter seeds make members identical, "
            "leaving no seed/order variation to measure"
        )
    return warnings
