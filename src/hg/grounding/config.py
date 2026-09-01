"""Config for the HG-v2 grounding experiments.

Config-first, per the repository's own rule: no hard-coded task sizes, seeds,
policies or thresholds (plan section 2.3). A separate object from
`hg.config.ExperimentConfig` because that one requires a dataset, a set of
ensemble conditions and per-member model checkpoints, none of which exist in
Phase 0 -- satisfying it would mean inventing a fake model and a fake condition
to get past validators that protect a different claim. See
experiments/hg_v2/REUSE_AUDIT.md.

The acceptance thresholds live here too, so the Phase 0 gate is something a
config declares and a run checks, not a number buried in a script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..config import ConfigError, read_config_file
from .noisy_index import NoisyIndexTaskSpec
from .route_choice import PREFERENCE_VARIABLES, RouteChoiceSpec


@dataclass(slots=True)
class IndexSweep:
    """The task-size sweep n = 2, 4, 8, ... (plan section 7)."""

    n_bits: list[int]
    n_items: int

    def __post_init__(self) -> None:
        if not self.n_bits:
            raise ConfigError("task.n_bits must list at least one size")
        if any(n < 2 for n in self.n_bits):
            raise ConfigError(f"task.n_bits must all be >= 2, got {self.n_bits}")
        if len(set(self.n_bits)) != len(self.n_bits):
            raise ConfigError(f"task.n_bits has duplicates: {self.n_bits}")
        if self.n_items < 1:
            raise ConfigError(f"task.n_items must be >= 1, got {self.n_items}")


@dataclass(slots=True)
class Phase0Gate:
    """Declared acceptance criteria (plan section 8)."""

    # ACT should approach chance. The tolerance is a sampling-error allowance,
    # so it must be read together with n_items: at 2000 items the standard
    # error on an accuracy is ~0.011, and 0.05 is ~4.5 of those.
    act_accuracy_tol: float = 0.05
    # The ask policies decode deterministically, so anything below 1.0 is a bug
    # in the channel or the decode, not sampling noise.
    min_ask_accuracy: float = 1.0
    # Generic response cost must scale linearly with n, targeted must not.
    expect_generic_bits_equal_n: bool = True
    expect_targeted_bits: float = 1.0
    # The ablated control must fail to resolve the decision.
    max_ablated_accuracy: float = 0.55


@dataclass(slots=True)
class Phase1Gate:
    """Declared acceptance criterion for gate G1 (plan section 11).

    "Learned routing must beat BOTH always ACT and always ASK for some
     nontrivial range of external-information costs."

    `min_winning_lambdas` is what turns "some nontrivial range" into a number.
    Two is the minimum that can distinguish a range from a single lucky price;
    raise it to demand a wider band.
    """

    min_winning_lambdas: int = 2

    def __post_init__(self) -> None:
        if self.min_winning_lambdas < 1:
            raise ConfigError("gate.min_winning_lambdas must be >= 1")


@dataclass(slots=True)
class Phase2Gate:
    """Declared acceptance criterion for gate G2 (plan sections 17 and 27).

    "At equal final distortion, does HG-v2 reduce external response cost?"

    `distortion_tolerance` is how much worse the synthesized-query arm's final
    loss may be than generic escalation's and still count as equal quality. It
    is one-sided on purpose: a query policy that decides *better* has not
    cheated, but one that buys cheapness with distortion has not compressed
    anything.
    """

    distortion_tolerance: float = 0.01

    def __post_init__(self) -> None:
        if self.distortion_tolerance < 0:
            raise ConfigError("gate.distortion_tolerance must be >= 0")


@dataclass(slots=True)
class GroundingConfig:
    name: str
    task_kind: str
    policies: list[str]
    # Exactly one of these is set, according to `task_kind`.
    sweep: IndexSweep | None = None
    phase1_task: NoisyIndexTaskSpec | None = None
    phase2_task: RouteChoiceSpec | None = None
    seed: int = 0
    n_bootstrap: int = 1000
    alpha: float = 0.05
    # Price of one external bit. Kept separate from the raw bit count in every
    # episode so a saved run can be repriced at a different lambda without
    # rerunning it (Phase 1 sweeps this; Phase 0 leaves it at 1.0).
    lambda_bit: float = 1.0
    # Phase 1 sweeps the price of an external bit rather than fixing it. The
    # episodes store raw bits separately from priced cost, so this is a
    # reporting grid, not a property baked into the artifact.
    lambda_grid: list[float] = field(default_factory=list)
    # Share of items used to fit the router; the rest is held out for scoring.
    train_fraction: float = 0.5
    output_root: Path = Path("results")
    gate: Phase0Gate | Phase1Gate | Phase2Gate = field(default_factory=Phase0Gate)
    # Where this config was loaded from, recorded so a run's manifest and its
    # reproduce command name the real file rather than a guess from the name.
    source_path: Path | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def resolved_dict(self) -> dict[str, Any]:
        """The fully-resolved config, written verbatim into the run directory."""
        if self.task_kind == "index":
            assert self.sweep is not None
            task: dict[str, Any] = {
                "kind": self.task_kind,
                "n_bits": list(self.sweep.n_bits),
                "n_items": self.sweep.n_items,
            }
        elif self.task_kind == "noisy_index":
            t = self.phase1_task
            assert t is not None
            task = {
                "kind": self.task_kind,
                "n_bits": t.n_bits,
                "n_items": t.n_items,
                "n_observations": t.n_observations,
                "p_self_sufficient": t.p_self_sufficient,
                "error_rates": list(t.error_rates),
            }
        else:
            r = self.phase2_task
            assert r is not None
            task = {
                "kind": self.task_kind,
                "n_items": r.n_items,
                "p_a_faster": r.p_a_faster,
                "p_accepts": r.p_accepts,
                "variables": list(r.variables),
            }
        return {
            "name": self.name,
            "task": task,
            "policies": list(self.policies),
            "seed": self.seed,
            "n_bootstrap": self.n_bootstrap,
            "alpha": self.alpha,
            "lambda_bit": self.lambda_bit,
            "lambda_grid": list(self.lambda_grid),
            "train_fraction": self.train_fraction,
            "router": dict(self.raw.get("router") or {}),
            "backend": self.raw.get("backend"),
            "backend_options": dict(self.raw.get("backend_options") or {}),
            "synthesizer": dict(self.raw.get("synthesizer") or {}),
            "self_consistency_samples": self.raw.get("self_consistency_samples"),
            "temperature": self.raw.get("temperature"),
            "max_new_tokens": self.raw.get("max_new_tokens"),
            "output_root": str(self.output_root),
            "source_path": str(self.source_path) if self.source_path else None,
            "gate": {
                slot: getattr(self.gate, slot) for slot in type(self.gate).__slots__
            },
        }


# Which task each phase uses, and which policy vocabulary and gate go with it.
TASK_KINDS = ("index", "noisy_index", "route_choice")


def load_grounding_config(
    path: str | Path, overrides: Mapping[str, Any] | None = None
) -> GroundingConfig:
    """Load and validate an HG-v2 experiment config.

    The task kind selects the phase: `index` is Phase 0's exact
    complementarity experiment, `noisy_index` is Phase 1's routing task. They
    have different policy vocabularies and different gates, so the kind is
    validated first and everything else follows from it.
    """

    raw = read_config_file(path)
    if overrides:
        raw = {**raw, **dict(overrides)}

    # `resolved_dict` records where a config came from, so a config read back
    # out of a run directory still names the original file rather than the
    # copy. Reproduce instructions have to point at something a person edits.
    recorded_source = raw.pop("source_path", None)
    # `router` is a passthrough block read by the phase 1 runner; keep it in
    # `raw` rather than promoting it to a typed field for now.

    task = raw.get("task")
    if not isinstance(task, Mapping):
        raise ConfigError(f"{path}: 'task' mapping is required")
    kind = str(task.get("kind", "index"))
    if kind not in TASK_KINDS:
        raise ConfigError(
            f"{path}: unknown task kind {kind!r}; expected one of {list(TASK_KINDS)}"
        )

    sweep: IndexSweep | None = None
    phase1_task: NoisyIndexTaskSpec | None = None
    phase2_task: RouteChoiceSpec | None = None
    gate_cls: type[Phase0Gate] | type[Phase1Gate]

    if kind == "index":
        from .synthetic import POLICIES as TASK_POLICIES

        n_bits = task.get("n_bits")
        if not isinstance(n_bits, (list, tuple)):
            raise ConfigError(f"{path}: task.n_bits must be a list of sizes")
        sweep = IndexSweep(
            n_bits=[int(n) for n in n_bits], n_items=int(task.get("n_items", 2000))
        )
        gate_cls = Phase0Gate
    elif kind == "noisy_index":
        from .phase1 import POLICIES as TASK_POLICIES

        unknown_task = set(task) - {
            "kind", "n_bits", "n_items", "n_observations",
            "p_self_sufficient", "error_rates", "seed",
        }
        if unknown_task:
            raise ConfigError(f"{path}: unknown task keys {sorted(unknown_task)}")
        try:
            phase1_task = NoisyIndexTaskSpec(
                n_bits=int(task.get("n_bits", 8)),
                n_items=int(task.get("n_items", 8000)),
                n_observations=int(task.get("n_observations", 5)),
                p_self_sufficient=float(task.get("p_self_sufficient", 0.4)),
                error_rates=tuple(float(e) for e in task.get("error_rates", (0.25, 0.45))),
                seed=int(raw.get("seed", 0)),
            )
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        gate_cls = Phase1Gate
    else:
        from .phase2 import BASELINES as TASK_POLICIES

        unknown_task = set(task) - {
            "kind", "n_items", "p_a_faster", "p_accepts", "variables", "seed",
        }
        if unknown_task:
            raise ConfigError(f"{path}: unknown task keys {sorted(unknown_task)}")
        try:
            phase2_task = RouteChoiceSpec(
                n_items=int(task.get("n_items", 400)),
                p_a_faster=float(task.get("p_a_faster", 0.5)),
                p_accepts=float(task.get("p_accepts", 0.5)),
                variables=tuple(task.get("variables") or PREFERENCE_VARIABLES),
                seed=int(raw.get("seed", 0)),
            )
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        gate_cls = Phase2Gate

    policies = list(raw.get("policies") or TASK_POLICIES)
    unknown = [p for p in policies if p not in TASK_POLICIES]
    if unknown:
        raise ConfigError(
            f"{path}: unknown policies {unknown} for task {kind!r}; "
            f"expected from {list(TASK_POLICIES)}"
        )
    if not policies:
        raise ConfigError(f"{path}: at least one policy is required")

    gate_raw = dict(raw.get("gate") or {})
    unknown_gate = set(gate_raw) - set(gate_cls.__slots__)
    if unknown_gate:
        raise ConfigError(
            f"{path}: unknown gate keys {sorted(unknown_gate)} for task {kind!r}; "
            f"expected from {sorted(gate_cls.__slots__)}"
        )

    lambda_grid = [float(v) for v in (raw.get("lambda_grid") or [])]
    if kind == "noisy_index" and not lambda_grid:
        raise ConfigError(
            f"{path}: task {kind!r} needs a `lambda_grid`; gate G1 is a claim "
            "about a range of external-information prices, so a single price "
            "cannot test it"
        )
    if len(set(lambda_grid)) != len(lambda_grid):
        raise ConfigError(f"{path}: lambda_grid has duplicates: {lambda_grid}")
    if any(v < 0 for v in lambda_grid):
        raise ConfigError(f"{path}: lambda_grid must be non-negative: {lambda_grid}")

    train_fraction = float(raw.get("train_fraction", 0.5))
    if not 0.0 < train_fraction < 1.0:
        raise ConfigError(f"{path}: train_fraction must be in (0, 1)")

    return GroundingConfig(
        name=str(raw.get("name") or Path(path).stem),
        task_kind=kind,
        sweep=sweep,
        phase1_task=phase1_task,
        phase2_task=phase2_task,
        policies=policies,
        seed=int(raw.get("seed", 0)),
        n_bootstrap=int(raw.get("n_bootstrap", 1000)),
        alpha=float(raw.get("alpha", 0.05)),
        lambda_bit=float(raw.get("lambda_bit", 1.0)),
        lambda_grid=lambda_grid,
        train_fraction=train_fraction,
        output_root=Path(raw.get("output_root", "results")),
        gate=gate_cls(**gate_raw),
        source_path=Path(recorded_source) if recorded_source else Path(path),
        raw=dict(raw),
    )
