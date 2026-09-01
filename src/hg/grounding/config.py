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
class FrontierSpec:
    """E1's sweep: task sizes, and the tolerances of the lossy relaxation.

    The query budgets are not listed: for each n the frontier runs
    r = 0..ceil(log2 n), which is the whole range over which b*(n, r) varies.
    Letting a config pick a subset would make it possible to report a frontier
    with its inconvenient cells omitted.
    """

    n_bits: list[int]
    n_items: int = 400
    n_items_lossy: int = 200
    epsilons: list[float] = field(default_factory=list)
    # Above this fiber size a lossy codebook is designed by a structured scheme
    # with closed-form distortion instead of by 2**s enumeration.
    max_enumerable_fiber: int = 10

    def __post_init__(self) -> None:
        if not self.n_bits:
            raise ConfigError("task.n_bits must list at least one size")
        if any(n < 2 for n in self.n_bits):
            raise ConfigError(f"task.n_bits must all be >= 2, got {self.n_bits}")
        if len(set(self.n_bits)) != len(self.n_bits):
            raise ConfigError(f"task.n_bits has duplicates: {self.n_bits}")
        if self.n_items < 1 or self.n_items_lossy < 1:
            raise ConfigError("task.n_items and task.n_items_lossy must be >= 1")
        if any(not 0.0 <= e < 0.5 for e in self.epsilons):
            raise ConfigError(
                f"task.epsilons must lie in [0, 0.5), got {self.epsilons}: at "
                "eps >= 0.5 the source can say nothing and still meet the "
                "tolerance, so the cell measures no protocol at all"
            )
        if self.max_enumerable_fiber < 1:
            raise ConfigError("task.max_enumerable_fiber must be >= 1")


@dataclass(slots=True)
class E1Gate:
    """Declared acceptance criteria for gate H0 (improved plan section 18).

    Only the lossy tolerance is a number worth configuring. The zero-error half
    is an exact identity: `b*(n, r) = ceil(n / 2**r)` at accuracy 1.0, with
    b* - 1 bits failing. There is no tolerance to set on an identity, and
    offering one would invite widening it.
    """

    # Slack on the realized error rate of a lossy cell, allowing for the
    # sampling error of a finite item count. At 200 items the standard error on
    # an error rate near 0.3 is ~0.032.
    epsilon_tolerance: float = 0.05
    # The truncated-protocol control fails with probability (s - b) / 2s, which
    # is 1/128 at n = 64, r = 0. Each cell is therefore run at whatever item
    # count buys this many expected failures, so "the bound is tight here" is
    # backed by the same evidence at every point of the frontier.
    min_expected_failures: int = 20

    def __post_init__(self) -> None:
        if self.epsilon_tolerance < 0:
            raise ConfigError("gate.epsilon_tolerance must be >= 0")
        if self.min_expected_failures < 1:
            raise ConfigError("gate.min_expected_failures must be >= 1")


@dataclass(slots=True)
class E2Gate:
    """Declared acceptance criteria for gates H1 and H2 (plan section 18).

    H1: `B_learned_route+synth < B_always_ask+synth` at matched final
    distortion. H2: at least one query intervention reduces response cost or
    raises eta_Q without materially reducing soundness or decision quality.
    """

    # How much worse an arm's final loss may be than its comparator and still
    # count as matched distortion. One-sided, for the same reason as Phase 2's.
    distortion_tolerance: float = 0.02
    # H1 is a claim about routing, which is only meaningful at a price of
    # external information -- so, as in gate G1, it must hold at more than one.
    min_winning_lambdas: int = 2
    # H2's guard: a minimality intervention may not drop soundness by more than
    # this against the unmodified synthesizer. Falsification F2 is exactly the
    # case where cheapness is bought with soundness.
    max_soundness_loss: float = 0.05

    def __post_init__(self) -> None:
        if self.distortion_tolerance < 0:
            raise ConfigError("gate.distortion_tolerance must be >= 0")
        if self.min_winning_lambdas < 1:
            raise ConfigError("gate.min_winning_lambdas must be >= 1")
        if not 0.0 <= self.max_soundness_loss <= 1.0:
            raise ConfigError("gate.max_soundness_loss must be in [0, 1]")


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
    frontier_task: FrontierSpec | None = None
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
    gate: Phase0Gate | Phase1Gate | Phase2Gate | E1Gate | E2Gate = field(
        default_factory=Phase0Gate
    )
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
        elif self.task_kind == "bounded_query":
            f = self.frontier_task
            assert f is not None
            task = {
                "kind": self.task_kind,
                "n_bits": list(f.n_bits),
                "n_items": f.n_items,
                "n_items_lossy": f.n_items_lossy,
                "epsilons": list(f.epsilons),
                "max_enumerable_fiber": f.max_enumerable_fiber,
            }
        else:
            r = self.phase2_task
            assert r is not None
            task = {
                "kind": self.task_kind,
                "n_items": r.n_items,
                "p_a_faster": r.p_a_faster,
                "p_accepts": r.p_accepts,
                "p_concession_on_a": r.p_concession_on_a,
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
            "query_variants": list(self.raw.get("query_variants") or []),
            "query_variant": self.raw.get("query_variant"),
            "arms": list(self.raw.get("arms") or []),
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
TASK_KINDS = (
    "index",
    "noisy_index",
    "route_choice",
    "bounded_query",
    "route_choice_integrated",
)


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
    frontier_task: FrontierSpec | None = None
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
    elif kind == "bounded_query":
        from .bounded_query import POLICIES as TASK_POLICIES

        unknown_task = set(task) - {
            "kind", "n_bits", "n_items", "n_items_lossy", "epsilons",
            "max_enumerable_fiber",
        }
        if unknown_task:
            raise ConfigError(f"{path}: unknown task keys {sorted(unknown_task)}")
        n_bits = task.get("n_bits")
        if not isinstance(n_bits, (list, tuple)):
            raise ConfigError(f"{path}: task.n_bits must be a list of sizes")
        try:
            frontier_task = FrontierSpec(
                n_bits=[int(n) for n in n_bits],
                n_items=int(task.get("n_items", 400)),
                n_items_lossy=int(task.get("n_items_lossy", 200)),
                epsilons=[float(e) for e in task.get("epsilons") or ()],
                max_enumerable_fiber=int(task.get("max_enumerable_fiber", 10)),
            )
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        gate_cls = E1Gate
    elif kind == "route_choice_integrated":
        from .integrated import ARMS as TASK_POLICIES

        unknown_task = set(task) - {
            "kind", "n_items", "p_a_faster", "p_accepts", "p_concession_on_a",
            "variables", "seed",
        }
        if unknown_task:
            raise ConfigError(f"{path}: unknown task keys {sorted(unknown_task)}")
        try:
            phase2_task = RouteChoiceSpec(
                n_items=int(task.get("n_items", 1000)),
                p_a_faster=float(task.get("p_a_faster", 0.5)),
                p_accepts=float(task.get("p_accepts", 0.5)),
                p_concession_on_a=float(task.get("p_concession_on_a", 0.5)),
                variables=tuple(task.get("variables") or PREFERENCE_VARIABLES),
                seed=int(raw.get("seed", 0)),
            )
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from exc
        gate_cls = E2Gate
    else:
        from .phase2 import BASELINES as TASK_POLICIES

        unknown_task = set(task) - {
            "kind", "n_items", "p_a_faster", "p_accepts", "p_concession_on_a",
            "variables", "seed",
        }
        if unknown_task:
            raise ConfigError(f"{path}: unknown task keys {sorted(unknown_task)}")
        try:
            phase2_task = RouteChoiceSpec(
                n_items=int(task.get("n_items", 400)),
                p_a_faster=float(task.get("p_a_faster", 0.5)),
                p_accepts=float(task.get("p_accepts", 0.5)),
                p_concession_on_a=float(task.get("p_concession_on_a", 0.5)),
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
    if kind in ("noisy_index", "route_choice_integrated") and not lambda_grid:
        raise ConfigError(
            f"{path}: task {kind!r} needs a `lambda_grid`; the routing gates "
            "are claims about a range of external-information prices, so a "
            "single price cannot test them"
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
        frontier_task=frontier_task,
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
