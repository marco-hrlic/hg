"""Distortion, external information cost, and coordination overhead.

Everything here is computed from saved episodes and nothing here calls a model,
which is the property that makes a re-analysis cost seconds (plan section 2.2).

The quantities, and what they are honestly claims about
------------------------------------------------------
`mean_loss_before/after`  measured 0/1 decision loss. No inference involved.
`delta_distortion`        D_before - D_after, the value the external
                          information actually delivered on this task.
`mean_external_bits`      mean over items of log2 |Y| for the response the
                          source had to transmit. A fixed-length *upper bound*
                          on C_-> (plan section 16), not an entropy estimate.
`coordination_overhead`   Omega = B_actual - H(J | M_A). Exact only where
                          H(J | M_A) is known by construction, which in this
                          repository means the Phase 0 synthetic tasks. The
                          residual deficit is therefore passed in explicitly
                          rather than estimated, so no caller can accidentally
                          treat a proxy as a Shannon quantity.
`information_efficiency`  delta_distortion / mean_external_bits, reported
                          alongside its numerator and denominator because the
                          ratio is undefined at zero bits and misleading near
                          it (plan sections 10 and 16).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..metrics.bootstrap import BootstrapResult
from .episodes import GroundingEpisode

# A plug-in entropy estimate needs this many samples per alphabet symbol
# before it is worth reporting. The estimator is biased downward, and the bias
# grows with |Y|/n_samples: for a generic n-bit query every sampled answer is
# distinct, so the estimate would report log2(n_samples) and look like a
# result. Even at |Y| = 256 with 2000 samples it reads ~7.89 against a true 8.
# Report nothing rather than a number that is mostly bias.
MIN_SAMPLES_PER_SYMBOL = 20


def zero_one_loss(action: Any, target: Any) -> float:
    """0 if the action matches the reference decision, else 1."""
    return 0.0 if action == target else 1.0


# Why an arm's Omega may be undefined.
OVERHEAD_DEFINED = "defined"
OVERHEAD_NO_COMMUNICATION = "no_communication"
OVERHEAD_INSUFFICIENT = "insufficient_communication"


def coordination_overhead(bits: float, residual_deficit_bits: float) -> float:
    """Omega = C_-> - H(J | M_A), the excess over the decision deficit.

    Only meaningful when the communication actually closed the deficit. The
    subtraction itself is happy to return nonsense in two cases that matter
    here, which is why callers should go through `overhead_status`:

    * a policy that never communicates (ACT) gets Omega = -H(J | M_A) < 0,
      apparently beating the source-coding bound. It has not compressed
      anything; it simply left the decision ungrounded.
    * a policy whose answer was insufficient (the ablated control) can show
      Omega = 0 while resolving nothing, which reads as optimal and is the
      most misleading number this module could print.

    A negative Omega from a *sufficient* arm is a bug -- in the bit accounting
    or in the claimed deficit -- not a discovery.
    """
    return float(bits) - float(residual_deficit_bits)


def overhead_status(*, communicated: bool, sufficient: bool) -> str:
    """Whether Omega is interpretable for an arm, and if not, why not."""
    if not communicated:
        return OVERHEAD_NO_COMMUNICATION
    if not sufficient:
        return OVERHEAD_INSUFFICIENT
    return OVERHEAD_DEFINED


def information_efficiency(delta_distortion: float, bits: float) -> float | None:
    """Distortion reduced per external bit. None when no bits were spent.

    None rather than inf or nan: "acted for free" is not an infinitely
    efficient use of external information, it is the absence of one, and a
    sentinel that propagates through a mean would corrupt the comparison the
    experiment exists to make.
    """
    if bits <= 0.0:
        return None
    return float(delta_distortion) / float(bits)


def bootstrap_mean(
    values: Sequence[float],
    *,
    name: str = "mean",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    null_value: float | None = None,
) -> BootstrapResult:
    """Item-level bootstrap of a mean, reported as a `BootstrapResult`.

    Reuses the M2-LLM result type so confidence intervals read identically
    across both experiments in the same results tree. The resampler is new only
    because `hg.metrics.bootstrap.bootstrap_metric` is typed on `ErrorMatrix`.
    """
    x = np.asarray(list(values), dtype=float)
    if x.size == 0:
        return BootstrapResult(
            name=name, point=float("nan"), ci_low=float("nan"),
            ci_high=float("nan"), se=float("nan"), n_boot=0, n_valid=0,
            alpha=alpha, p_value=None, null_value=null_value, draws=[],
        )
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = float(np.mean(x[rng.integers(0, x.size, size=x.size)]))
    lo, hi = (float(np.percentile(draws, 100 * alpha / 2)),
              float(np.percentile(draws, 100 * (1 - alpha / 2))))
    p_value = None
    if null_value is not None:
        # Two-sided bootstrap p, matching hg.metrics.bootstrap's convention.
        left = float(np.mean(draws <= null_value))
        right = float(np.mean(draws >= null_value))
        p_value = float(min(1.0, 2.0 * min(left, right)))
    return BootstrapResult(
        name=name,
        point=float(np.mean(x)),
        ci_low=lo,
        ci_high=hi,
        se=float(np.std(draws, ddof=1)) if n_boot > 1 else float("nan"),
        n_boot=n_boot,
        n_valid=n_boot,
        alpha=alpha,
        p_value=p_value,
        null_value=null_value,
        draws=[float(v) for v in draws],
    )


def empirical_response_entropy(episodes: Sequence[GroundingEpisode]) -> float | None:
    """Plug-in entropy of the responses actually transmitted, or None.

    Returns None when no episode asked anything, when the response alphabet is
    unknown, or when there are too few samples per symbol to estimate it (see
    MIN_SAMPLES_PER_SYMBOL). None is the honest answer; a number would be an
    artifact of the sample size rather than a property of the channel.
    """
    contents: list[Any] = []
    alphabet: int | None = None
    for e in episodes:
        if e.external_response is None or e.query is None:
            continue
        contents.append(e.external_response.get("content"))
        size = e.query.get("response_alphabet_size")
        if size is None:
            return None
        alphabet = size if alphabet is None else max(alphabet, int(size))
    if not contents or alphabet is None:
        return None
    if len(contents) < MIN_SAMPLES_PER_SYMBOL * alphabet:
        return None
    counts = Counter(_hashable(c) for c in contents)
    n = sum(counts.values())
    return float(-sum((c / n) * math.log2(c / n) for c in counts.values()))


def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


# --------------------------------------------------------------- aggregation ---
# The key an episode is grouped under for reporting. `n_bits` is the index
# task's size n; None for tasks without one.
ArmKey = tuple[str, int | None]


def arm_key(episode: GroundingEpisode) -> ArmKey:
    n_bits = episode.ai_private_state.get("n_bits")
    return (episode.policy, int(n_bits) if n_bits is not None else None)


def group_by_arm(
    episodes: Sequence[GroundingEpisode],
) -> dict[ArmKey, list[GroundingEpisode]]:
    """Group episodes into (policy, n) arms, preserving first-seen order."""
    out: dict[ArmKey, list[GroundingEpisode]] = {}
    for e in episodes:
        out.setdefault(arm_key(e), []).append(e)
    return out


@dataclass(slots=True)
class ArmMetrics:
    """Everything Phase 0 reports for one (policy, n) arm."""

    policy: str
    task: str
    n_items: int
    n_bits: int | None
    # H(J | M_A): known by construction for the Phase 0 tasks.
    residual_deficit_bits: float

    accuracy: float
    mean_loss_before: float
    mean_loss_after: float
    delta_distortion: float

    external_query_rate: float
    mean_external_bits: float
    mean_external_cost: float
    # None unless the arm both communicated and closed the deficit; read
    # `overhead_status` for which case applies.
    coordination_overhead: float | None
    overhead_status: str
    # Did the grounded decision actually come out right, within tolerance?
    sufficient: bool
    information_efficiency: float | None
    empirical_response_entropy_bits: float | None
    # None when the arm never asked: "what share of answers were informative"
    # has no value where there were no answers. None rather than NaN because
    # NaN is not representable in strict JSON -- json.dump would emit a bare
    # `NaN` token and any conforming reader would reject the artifact.
    informative_response_rate: float | None

    accuracy_ci: tuple[float, float] | None = None
    external_bits_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "task": self.task,
            "n_items": self.n_items,
            "n_bits": self.n_bits,
            "residual_deficit_bits": self.residual_deficit_bits,
            "accuracy": self.accuracy,
            "accuracy_ci": list(self.accuracy_ci) if self.accuracy_ci else None,
            "mean_loss_before": self.mean_loss_before,
            "mean_loss_after": self.mean_loss_after,
            "delta_distortion": self.delta_distortion,
            "external_query_rate": self.external_query_rate,
            "mean_external_bits": self.mean_external_bits,
            "external_bits_ci": (
                list(self.external_bits_ci) if self.external_bits_ci else None
            ),
            "mean_external_cost": self.mean_external_cost,
            "coordination_overhead": self.coordination_overhead,
            "overhead_status": self.overhead_status,
            "sufficient": self.sufficient,
            # Reported with its parts, never on its own: the ratio is undefined
            # at zero bits (plan sections 10 and 16).
            "information_efficiency": self.information_efficiency,
            "efficiency_numerator": self.delta_distortion,
            "efficiency_denominator": self.mean_external_bits,
            "empirical_response_entropy_bits": self.empirical_response_entropy_bits,
            "informative_response_rate": self.informative_response_rate,
        }


def aggregate_arm(
    episodes: Sequence[GroundingEpisode],
    *,
    residual_deficit_bits: float | None = None,
    n_boot: int = 0,
    alpha: float = 0.05,
    seed: int = 0,
    sufficiency_tol: float = 1e-9,
) -> ArmMetrics:
    """Aggregate one arm's episodes into the Phase 0 metric set.

    `residual_deficit_bits` defaults to the value the episodes carry in their
    decision state, and disagreement across episodes is an error rather than an
    average: Omega is only meaningful against a single known deficit.

    `n_boot=0` skips the bootstrap, which keeps the unit tests exact.

    `sufficiency_tol` is how far below perfect accuracy an arm may sit and
    still count as having closed the deficit. It defaults to exact because in
    the Phase 0 tasks a sufficient answer is decoded deterministically: any
    error at all means the information did not suffice.
    """
    if not episodes:
        raise ValueError("cannot aggregate an empty arm")

    policies = {e.policy for e in episodes}
    if len(policies) != 1:
        raise ValueError(f"arm mixes policies {sorted(policies)}")
    tasks = {e.task for e in episodes}
    if len(tasks) != 1:
        raise ValueError(f"arm mixes tasks {sorted(tasks)}")

    if residual_deficit_bits is None:
        deficits = {
            float(e.decision_state["residual_deficit_bits"])
            for e in episodes
            if "residual_deficit_bits" in e.decision_state
        }
        if len(deficits) != 1:
            raise ValueError(
                "episodes carry "
                f"{len(deficits)} distinct residual_deficit_bits values "
                f"{sorted(deficits)}; pass residual_deficit_bits explicitly. "
                "Omega is defined against one known deficit, so averaging "
                "several would silently produce a meaningless overhead."
            )
        residual_deficit_bits = deficits.pop()

    losses_before = [float(e.initial_loss) for e in episodes]
    losses_after = [float(e.final_loss) for e in episodes]
    bits = [float(e.external_raw_bits) for e in episodes]
    costs = [float(e.external_cost) for e in episodes]
    asked = [e.query is not None for e in episodes]
    correct = [1.0 - l for l in losses_after]

    mean_before = float(np.mean(losses_before))
    mean_after = float(np.mean(losses_after))
    mean_bits = float(np.mean(bits))
    delta = mean_before - mean_after

    informative = [
        bool((e.external_response or {}).get("informative", False))
        for e in episodes
        if e.external_response is not None
    ]

    acc_ci: tuple[float, float] | None = None
    bits_ci: tuple[float, float] | None = None
    if n_boot > 0:
        a = bootstrap_mean(correct, name="accuracy", n_boot=n_boot, alpha=alpha, seed=seed)
        b = bootstrap_mean(bits, name="external_bits", n_boot=n_boot, alpha=alpha, seed=seed + 1)
        acc_ci = (a.ci_low, a.ci_high)
        bits_ci = (b.ci_low, b.ci_high)

    n_bits_values = {e.ai_private_state.get("n_bits") for e in episodes}
    if len(n_bits_values) != 1:
        raise ValueError(f"arm mixes task sizes n_bits={sorted(map(str, n_bits_values))}")
    raw_n_bits = n_bits_values.pop()
    n_bits = int(raw_n_bits) if raw_n_bits is not None else None

    accuracy = float(np.mean(correct))
    communicated = any(asked)
    sufficient = accuracy >= 1.0 - sufficiency_tol
    status = overhead_status(communicated=communicated, sufficient=sufficient)

    return ArmMetrics(
        policy=policies.pop(),
        task=tasks.pop(),
        n_items=len(episodes),
        n_bits=n_bits,
        residual_deficit_bits=float(residual_deficit_bits),
        accuracy=accuracy,
        mean_loss_before=mean_before,
        mean_loss_after=mean_after,
        delta_distortion=delta,
        external_query_rate=float(np.mean(asked)),
        mean_external_bits=mean_bits,
        mean_external_cost=float(np.mean(costs)),
        coordination_overhead=(
            coordination_overhead(mean_bits, residual_deficit_bits)
            if status == OVERHEAD_DEFINED
            else None
        ),
        overhead_status=status,
        sufficient=sufficient,
        information_efficiency=information_efficiency(delta, mean_bits),
        empirical_response_entropy_bits=empirical_response_entropy(episodes),
        informative_response_rate=(
            float(np.mean(informative)) if informative else None
        ),
        accuracy_ci=acc_ci,
        external_bits_ci=bits_ci,
    )


def aggregate_all(
    episodes: Sequence[GroundingEpisode],
    *,
    n_boot: int = 0,
    alpha: float = 0.05,
    seed: int = 0,
    sufficiency_tol: float = 1e-9,
) -> list[ArmMetrics]:
    """Aggregate every (policy, n) arm, sorted by n then policy for reporting."""
    arms = [
        aggregate_arm(
            group, n_boot=n_boot, alpha=alpha, seed=seed,
            sufficiency_tol=sufficiency_tol,
        )
        for group in group_by_arm(episodes).values()
    ]
    return sorted(arms, key=lambda a: (a.n_bits if a.n_bits is not None else -1, a.policy))


# ------------------------------------------------------- route accounting -----
# Phase 1 metrics (plan section 10). The unit is still one episode, but the
# question changes: not "what did this query cost?" but "was routing worth it?".
def total_route_cost(loss: float, bits: float, lambda_bit: float) -> float:
    """C_r = L_r + lambda * B_r (plan section 9).

    The single scalar the router is optimizing. Reported alongside its two
    components everywhere, because a policy can lower C by trading decision
    quality for communication and that trade is the thing under study, not a
    detail.
    """
    return float(loss) + float(lambda_bit) * float(bits)


def _oracle_cost(episode: GroundingEpisode) -> float:
    """min over routes of C_r, from the counterfactuals stored on the episode."""
    md = episode.metadata
    lam = float(md["lambda_bit"])
    return min(
        total_route_cost(float(md["loss_act"]), float(md["bits_act"]), lam),
        total_route_cost(float(md["loss_ask"]), float(md["bits_ask"]), lam),
    )


def _oracle_route(episode: GroundingEpisode) -> str:
    md = episode.metadata
    lam = float(md["lambda_bit"])
    cost_act = total_route_cost(float(md["loss_act"]), float(md["bits_act"]), lam)
    cost_ask = total_route_cost(float(md["loss_ask"]), float(md["bits_ask"]), lam)
    # Ties to ACT, matching hg.routing.grounding_router.oracle_route: at
    # indifference, not spending the source's attention is the better default.
    return "act" if cost_act <= cost_ask else "ask_targeted"


@dataclass(slots=True)
class RouteMetrics:
    """Everything Phase 1 reports for one (policy, lambda) arm."""

    policy: str
    task: str
    n_items: int
    lambda_bit: float

    # Fraction of items where the chosen route equals the oracle route.
    route_accuracy: float
    # Final decision distortion, and the loss the AI would have taken by acting.
    mean_final_loss: float
    mean_loss_before: float
    delta_distortion: float

    external_query_rate: float
    mean_external_bits: float
    mean_total_cost: float
    mean_oracle_cost: float
    # Regret = C_chosen - C_oracle, averaged over items. Zero only for a
    # clairvoyant router.
    routing_regret: float

    information_efficiency: float | None
    total_cost_ci: tuple[float, float] | None = None
    regret_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "task": self.task,
            "n_items": self.n_items,
            "lambda_bit": self.lambda_bit,
            "route_accuracy": self.route_accuracy,
            "mean_final_loss": self.mean_final_loss,
            "mean_loss_before": self.mean_loss_before,
            "delta_distortion": self.delta_distortion,
            "external_query_rate": self.external_query_rate,
            "mean_external_bits": self.mean_external_bits,
            "mean_total_cost": self.mean_total_cost,
            "total_cost_ci": list(self.total_cost_ci) if self.total_cost_ci else None,
            "mean_oracle_cost": self.mean_oracle_cost,
            "routing_regret": self.routing_regret,
            "regret_ci": list(self.regret_ci) if self.regret_ci else None,
            "information_efficiency": self.information_efficiency,
            "efficiency_numerator": self.delta_distortion,
            "efficiency_denominator": self.mean_external_bits,
        }


def episode_costs(episodes: Sequence[GroundingEpisode]) -> np.ndarray:
    """Per-episode realized C_r, in episode order. The paired-bootstrap input."""
    return np.asarray(
        [
            total_route_cost(
                e.final_loss, e.external_raw_bits, float(e.metadata["lambda_bit"])
            )
            for e in episodes
        ],
        dtype=float,
    )


def aggregate_route_arm(
    episodes: Sequence[GroundingEpisode],
    *,
    n_boot: int = 0,
    alpha: float = 0.05,
    seed: int = 0,
) -> RouteMetrics:
    """Aggregate one (policy, lambda) arm into the Phase 1 metric set."""
    if not episodes:
        raise ValueError("cannot aggregate an empty arm")
    policies = {e.policy for e in episodes}
    if len(policies) != 1:
        raise ValueError(f"arm mixes policies {sorted(policies)}")
    lambdas = {float(e.metadata["lambda_bit"]) for e in episodes}
    if len(lambdas) != 1:
        raise ValueError(
            f"arm mixes lambda values {sorted(lambdas)}; the route objective is "
            "defined at one price of external information"
        )

    costs = episode_costs(episodes)
    oracle = np.asarray([_oracle_cost(e) for e in episodes], dtype=float)
    regret = costs - oracle
    losses_before = np.asarray([e.initial_loss for e in episodes], dtype=float)
    losses_after = np.asarray([e.final_loss for e in episodes], dtype=float)
    bits = np.asarray([e.external_raw_bits for e in episodes], dtype=float)
    matched = np.asarray(
        [e.route == _oracle_route(e) for e in episodes], dtype=float
    )

    delta = float(losses_before.mean() - losses_after.mean())
    mean_bits = float(bits.mean())

    cost_ci = regret_ci = None
    if n_boot > 0:
        c = bootstrap_mean(costs, name="total_cost", n_boot=n_boot, alpha=alpha, seed=seed)
        r = bootstrap_mean(regret, name="regret", n_boot=n_boot, alpha=alpha, seed=seed + 1)
        cost_ci = (c.ci_low, c.ci_high)
        regret_ci = (r.ci_low, r.ci_high)

    return RouteMetrics(
        policy=policies.pop(),
        task=episodes[0].task,
        n_items=len(episodes),
        lambda_bit=lambdas.pop(),
        route_accuracy=float(matched.mean()),
        mean_final_loss=float(losses_after.mean()),
        mean_loss_before=float(losses_before.mean()),
        delta_distortion=delta,
        external_query_rate=float(np.mean([e.query is not None for e in episodes])),
        mean_external_bits=mean_bits,
        mean_total_cost=float(costs.mean()),
        mean_oracle_cost=float(oracle.mean()),
        routing_regret=float(regret.mean()),
        information_efficiency=information_efficiency(delta, mean_bits),
        total_cost_ci=cost_ci,
        regret_ci=regret_ci,
    )


def paired_cost_advantage(
    challenger: Sequence[GroundingEpisode],
    baseline: Sequence[GroundingEpisode],
    *,
    name: str = "cost_advantage",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """Item-paired bootstrap of C_baseline - C_challenger. Positive = better.

    Paired, because every policy is evaluated on the same items in the same
    order: an unpaired comparison would carry the between-item variance of the
    task, which is far larger than the difference between two routing policies
    and would hide a real effect behind it.
    """
    a = list(challenger)
    b = list(baseline)
    if len(a) != len(b):
        raise ValueError(f"unpaired arms: {len(a)} vs {len(b)} episodes")
    mismatched = [
        (x.item_id, y.item_id) for x, y in zip(a, b) if x.item_id != y.item_id
    ]
    if mismatched:
        raise ValueError(
            f"arms are not item-aligned; first mismatch {mismatched[0]}. "
            "A paired comparison over misaligned arms is meaningless."
        )
    diff = episode_costs(b) - episode_costs(a)
    return bootstrap_mean(
        diff, name=name, n_boot=n_boot, alpha=alpha, seed=seed, null_value=0.0
    )
