"""Phase 2 orchestration: a frozen primary LM and the query-compression claim.

Plan sections 12 to 17. The baselines are the plan's, and the comparison that
matters is B4 against B2:

    B0  frozen LM, no grounding
    B1  frozen LM + more internal sampling (self-consistency)
    B2  frozen LM + generic ask-human escalation
    B3  frozen LM + oracle targeted query
    B4  frozen LM + a synthesized query

    "At equal final distortion, does HG-v2 reduce external response cost?"

Gate G2 is therefore a *conditional* comparison and is written as one: a query
policy that reaches lower cost by answering worse has not compressed anything.

Two deliberate design choices, both of which change how the numbers read:

1. **Every ask baseline asks on every item.** The task has items whose prompt
   already determines the answer, so a router would skip them -- but routing is
   Phase 1's contribution and gate G1 already measured it. Holding the ask
   policy fixed across B2, B3 and B4 leaves the *query* as the only difference,
   which is what isolates gate G2. Excess response cost is reported both over
   all items and over the items that genuinely need grounding, so the cost of
   not routing stays visible rather than being folded into the query result.

2. **B1 exists to fail.** Self-consistency spends internal computation, and on
   an item whose decision-relevant variable is absent from the prompt there is
   nothing internal to converge on. B1 costs zero external bits and should not
   beat B0 by much. That is the empirical content of `internal_reducibility = 0`
   in the decision state, and it is the reason "just sample more" is not an
   answer to a grounding deficit.

Running this on `grounding_simulator` validates the pipeline; it says nothing
about an LLM. Gate G2 is only meaningful once `backend` names a real model.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..inference.base import GenerationSettings, get_backend
from ..io import git_commit, run_id, write_json, write_yaml
from ..queries.lm_query_synthesizer import LMQuerySynthesizer
from ..queries.synthetic import GoldQuerySynthesizer, NoisyQuerySynthesizer
from ..report.figures import plot_grounding_sweep
from .channels import QuerySpec, Route
from .config import GroundingConfig
from .costs import bootstrap_mean, information_efficiency, zero_one_loss
from .episodes import GroundingEpisode, GroundingRunDir, read_episodes, write_episodes
from .phase0 import environment
from .route_choice import (
    ACTIONS,
    NO_CONSTRAINTS,
    PROMPT_VERSION,
    ROUTE_CHOICE_TASK_VERSION,
    TASK_NAME,
    RouteChoiceItem,
    RouteChoiceSpec,
    TravellerSource,
    constraints_text,
    excess_response_cost,
    generate_route_choice_items,
    generic_query,
    resolves_decision,
)

log = logging.getLogger(__name__)

BASELINES = (
    "b0_no_grounding",
    "b1_self_consistency",
    "b2_generic_ask",
    "b3_oracle_targeted",
    "b4_synthesized_query",
)

# The pair gate G2 is about (plan section 17: "the important comparison is
# HG-v2 query versus generic human escalation").
G2_CHALLENGER = "b4_synthesized_query"
G2_BASELINE = "b2_generic_ask"
# The oracle query against the same baseline. This isolates the *mechanism* --
# can a targeted query compress at equal quality at all? -- from the quality of
# whatever produced the query. A run where the mechanism check fails is broken;
# a run where only the G2 comparison fails has a bad synthesizer.
MECHANISM_CHALLENGER = "b3_oracle_targeted"


@dataclass(slots=True)
class ArmQueryMetrics:
    """Per-arm metrics, including the query-quality set from plan section 16."""

    baseline: str
    n_items: int
    accuracy: float
    mean_final_loss: float
    mean_loss_before: float
    delta_distortion: float
    mean_external_bits: float
    # Resolution accuracy: did the answer allow the correct final action?
    # Distinct from `accuracy`, which also depends on the model applying the
    # rule correctly once it has the information.
    resolution_accuracy: float
    # Omega-hat = B_query - B_optimal (plan section 16), over all items and
    # over just the items with a nonzero residual deficit.
    mean_excess_response_cost: float
    mean_excess_where_grounding_needed: float | None
    query_efficiency: float | None
    # Share of asks whose variable set equals the gold query's.
    gold_match_rate: float | None
    # Internal compute, for the baselines that spend it instead of bits.
    mean_samples: float
    accuracy_ci: tuple[float, float] | None = None
    bits_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "n_items": self.n_items,
            "accuracy": self.accuracy,
            "accuracy_ci": list(self.accuracy_ci) if self.accuracy_ci else None,
            "mean_final_loss": self.mean_final_loss,
            "mean_loss_before": self.mean_loss_before,
            "delta_distortion": self.delta_distortion,
            "mean_external_bits": self.mean_external_bits,
            "bits_ci": list(self.bits_ci) if self.bits_ci else None,
            "resolution_accuracy": self.resolution_accuracy,
            "mean_excess_response_cost": self.mean_excess_response_cost,
            "mean_excess_where_grounding_needed": self.mean_excess_where_grounding_needed,
            "query_efficiency": self.query_efficiency,
            "efficiency_numerator": self.delta_distortion,
            "efficiency_denominator": self.mean_external_bits,
            "gold_match_rate": self.gold_match_rate,
            "mean_samples": self.mean_samples,
        }


# ------------------------------------------------------------------ running ---
@contextlib.contextmanager
def _generate_mode(backend):
    """Temporarily put a backend into sampled-generation mode.

    Baseline B1 needs *real* samples. The transformers backend defaults to
    `scoring_mode: rank`, which scores each label's continuation log-probability
    and takes the argmax -- a deterministic function of the prompt, so drawing
    "k samples" from it would return the same answer k times and B1 would be B0
    with a different name. Ranking is still the right default for the decision
    arms, since it removes the parser as a source of variance.

    Mutating and restoring the attribute rather than constructing a second
    backend keeps the weights loaded once; a second instance would mean a second
    copy of the model on the device for no scientific gain. A backend without the
    attribute (the simulator) is left alone.
    """
    previous = getattr(backend, "scoring_mode", None)
    if previous is not None:
        backend.scoring_mode = "generate"
    try:
        yield backend
    finally:
        if previous is not None:
            backend.scoring_mode = previous


def _settings(cfg: GroundingConfig, *, temperature: float, seed: int) -> GenerationSettings:
    return GenerationSettings(
        temperature=temperature,
        max_new_tokens=int(cfg.raw.get("max_new_tokens", 4)),
        seed=seed,
    )


def _episode(
    item: RouteChoiceItem,
    baseline: str,
    route: Route,
    *,
    action_before: str,
    action_after: str,
    query: QuerySpec | None,
    response: Any,
    lambda_bit: float,
    n_samples: int,
    backend_name: str,
) -> GroundingEpisode:
    reference = item.reference_action
    episode = GroundingEpisode(
        item_id=item.item_id,
        task=TASK_NAME,
        policy=baseline,
        route=route.value,
        ai_private_state={
            "requirement": item.requirement,
            "a_faster": item.a_faster,
            "time_a": item.time_a,
            "time_b": item.time_b,
            "prompt_version": PROMPT_VERSION,
        },
        hidden_reference_state={
            "acceptances": dict(item.acceptances),
            "decision_variable": reference,
            "needs_grounding": item.needs_grounding,
        },
        decision_state={
            "candidate_actions": list(ACTIONS),
            "predicted_action": action_before,
            # Exact for this construction: 1 bit when the withheld variable
            # decides, 0 when the prompt already closes the rule.
            "exact": True,
            "residual_deficit_bits": item.residual_deficit_bits,
            "internal_reducibility": 0.0,
            "contestedness": 0.0,
            "consequence": 1.0,
        },
        initial_action=action_before,
        initial_loss=zero_one_loss(action_before, reference),
        final_action=action_after,
        final_loss=zero_one_loss(action_after, reference),
        metadata={
            "task_version": ROUTE_CHOICE_TASK_VERSION,
            "backend": backend_name,
            "n_samples": n_samples,
            "needs_grounding": item.needs_grounding,
            "residual_deficit_bits": item.residual_deficit_bits,
            "lambda_bit": lambda_bit,
        },
    )
    if query is not None and response is not None:
        episode.attach_grounding(query, response, lambda_bit=lambda_bit)
        episode.metadata["resolves_decision"] = resolves_decision(item, query)
        episode.metadata["excess_response_cost"] = excess_response_cost(item, query)
        episode.metadata["query_variables"] = list(query.variable_names)
        episode.metadata["gold_variables"] = [item.requirement]
    else:
        # An ungrounded arm still resolves the items whose prompt sufficed.
        episode.metadata["resolves_decision"] = not item.needs_grounding
        episode.metadata["excess_response_cost"] = -item.residual_deficit_bits
    episode.validate()
    return episode


def ungrounded_pass(
    items: Sequence[RouteChoiceItem], cfg: GroundingConfig, backend
) -> dict[str, str]:
    """What the model does with the prompt as given, computed once.

    Every baseline needs this for `initial_loss`, and the prompt and decoding
    are identical across them, so it is computed once and shared. Two reasons,
    and the second is the important one: it removes four fifths of the forward
    passes in a five-arm run, and it makes `initial_action` *identical* across
    arms by construction rather than by coincidence -- which is what lets the
    distortion reduction attributable to grounding be compared between them.
    """
    base_items = [item.inference_item() for item in items]
    preds = backend.predict(base_items, _settings(cfg, temperature=0.0, seed=cfg.seed))
    return {p.item_id: (p.label or ACTIONS[0]) for p in preds}


def run_baseline(
    baseline: str,
    items: Sequence[RouteChoiceItem],
    cfg: GroundingConfig,
    backend,
    diagnostics: dict[str, Any] | None = None,
    before: Mapping[str, str] | None = None,
) -> list[GroundingEpisode]:
    """Run one baseline over every item."""
    lambda_bit = cfg.lambda_bit
    backend_name = getattr(backend, "name", "unknown")
    n_consistency = int(cfg.raw.get("self_consistency_samples", 5))

    if before is None:
        before = ungrounded_pass(items, cfg, backend)
    base_items = [item.inference_item() for item in items]

    if baseline == "b0_no_grounding":
        return [
            _episode(
                item, baseline, Route.ACT,
                action_before=before[item.item_id],
                action_after=before[item.item_id],
                query=None, response=None, lambda_bit=lambda_bit,
                n_samples=1, backend_name=backend_name,
            )
            for item in items
        ]

    if baseline == "b1_self_consistency":
        votes: dict[str, Counter] = {item.item_id: Counter() for item in items}
        # k independent draws, taken by varying the decoding seed. That is the
        # only way to get real samples through the Backend interface, and it
        # keeps the run reproducible.
        with _generate_mode(backend):
            for k in range(n_consistency):
                preds = backend.predict(
                    base_items,
                    _settings(
                        cfg,
                        temperature=float(cfg.raw.get("temperature", 0.7)),
                        seed=cfg.seed + 1000 + k,
                    ),
                )
                for p in preds:
                    # An unparseable sample is a real outcome of sampling, and
                    # dropping it would flatter self-consistency. It votes for
                    # nothing and the majority is taken over what parsed.
                    if p.label is not None:
                        votes[p.item_id][p.label] += 1
        return [
            _episode(
                item, baseline, Route.ACT,
                action_before=before[item.item_id],
                action_after=(
                    votes[item.item_id].most_common(1)[0][0]
                    if votes[item.item_id]
                    else before[item.item_id]
                ),
                query=None, response=None, lambda_bit=lambda_bit,
                n_samples=n_consistency, backend_name=backend_name,
            )
            for item in items
        ]

    # The three ask baselines differ only in which query they send.
    if baseline == "b2_generic_ask":
        queries: list[QuerySpec | None] = [generic_query(item.variables) for item in items]
    else:
        synthesizer = _synthesizer_for(baseline, cfg, backend)
        batch = getattr(synthesizer, "synthesize_batch", None)
        if diagnostics is not None and hasattr(synthesizer, "summary"):
            diagnostics.setdefault("synthesizer_runs", {})[baseline] = synthesizer
        if batch is not None:
            # The LM synthesizer needs one backend call for many items; a
            # per-item loop on a real model wastes most of the GPU.
            with _generate_mode(backend):
                queries = list(batch(items))
        else:
            queries = [synthesizer.synthesize(item) for item in items]

    # A synthesizer may return None -- the LM answering "NONE", i.e. deciding no
    # query is needed. That item is then an ACT, and it must cost zero bits
    # rather than being quietly charged for a query that was never sent.
    responses = [
        TravellerSource(item).respond(query) if query is not None else None
        for item, query in zip(items, queries)
    ]
    grounded_items = [
        item.inference_item(
            constraints_text(response.content) if response is not None else NO_CONSTRAINTS,
            revealed=tuple(response.content) if response is not None else (),
        )
        for item, response in zip(items, responses)
    ]
    grounded_preds = backend.predict(
        grounded_items, _settings(cfg, temperature=0.0, seed=cfg.seed)
    )
    after = {p.item_id: (p.label or ACTIONS[0]) for p in grounded_preds}

    asked_route = Route.ASK_GENERIC if baseline == "b2_generic_ask" else Route.ASK_TARGETED
    return [
        _episode(
            item, baseline,
            asked_route if query is not None else Route.ACT,
            action_before=before[item.item_id],
            action_after=after[item.item_id],
            query=query, response=response, lambda_bit=lambda_bit,
            n_samples=1, backend_name=backend_name,
        )
        for item, query, response in zip(items, queries, responses)
    ]


def _synthesizer_for(baseline: str, cfg: GroundingConfig, backend=None):
    if baseline == "b3_oracle_targeted":
        return GoldQuerySynthesizer()
    if baseline == "b4_synthesized_query":
        opts = dict(cfg.raw.get("synthesizer") or {})
        kind = str(opts.pop("kind", "noisy"))
        if kind == "gold":
            return GoldQuerySynthesizer()
        if kind == "noisy":
            opts.setdefault("seed", cfg.seed)
            return NoisyQuerySynthesizer(**opts)
        if kind == "lm":
            if backend is None:
                raise ValueError("the lm synthesizer needs the primary backend")
            opts.setdefault("seed", cfg.seed)
            return LMQuerySynthesizer(backend, **opts)
        raise ValueError(
            f"unknown synthesizer kind {kind!r}; expected 'gold', 'noisy' or 'lm'"
        )
    raise ValueError(f"baseline {baseline!r} does not use a synthesizer")


def run_sweep(cfg: GroundingConfig) -> tuple[list[GroundingEpisode], dict[str, Any]]:
    task = cfg.phase2_task
    if task is None:
        raise ValueError("phase 2 requires a `task` section with kind: route_choice")

    items = generate_route_choice_items(task)
    backend_name = str(cfg.raw.get("backend", "grounding_simulator"))
    backend_options = dict(cfg.raw.get("backend_options") or {})
    backend_options.setdefault("seed", cfg.seed)
    backend = get_backend(backend_name, **backend_options)
    backend.load()
    synth_holder: dict[str, Any] = {}
    try:
        before = ungrounded_pass(items, cfg, backend)
        episodes: list[GroundingEpisode] = []
        for baseline in cfg.policies:
            episodes.extend(
                run_baseline(
                    baseline, items, cfg, backend,
                    diagnostics=synth_holder, before=before,
                )
            )
            log.info("%s: %d episodes", baseline, len(items))
    finally:
        backend.unload()

    diagnostics = {
        "n_items": len(items),
        "backend": backend_name,
        "backend_options": backend_options,
        "prompt_version": PROMPT_VERSION,
        "preference_variables": list(task.variables),
        "generic_ask_bits": task.generic_bits,
        "share_needing_grounding": float(
            sum(i.needs_grounding for i in items) / len(items)
        ),
        "synthesizer": dict(cfg.raw.get("synthesizer") or {}),
        "example_prompt": items[0].prompt(),
        "is_simulated": backend_name == "grounding_simulator",
        # How the synthesized queries were actually arrived at: parse rate,
        # fallback rate, how many variables were named. Without this a B4 result
        # cannot be attributed to targeting rather than to a parser.
        "synthesizer_summary": {
            name: obj.summary()
            for name, obj in (synth_holder.get("synthesizer_runs") or {}).items()
        },
    }
    return episodes, diagnostics


# ----------------------------------------------------------------- analysis ---
def aggregate_baseline(
    episodes: Sequence[GroundingEpisode],
    *,
    n_boot: int = 0,
    alpha: float = 0.05,
    seed: int = 0,
) -> ArmQueryMetrics:
    if not episodes:
        raise ValueError("cannot aggregate an empty baseline")
    baselines = {e.policy for e in episodes}
    if len(baselines) != 1:
        raise ValueError(f"arm mixes baselines {sorted(baselines)}")

    correct = [1.0 - e.final_loss for e in episodes]
    losses_before = [e.initial_loss for e in episodes]
    losses_after = [e.final_loss for e in episodes]
    bits = [e.external_raw_bits for e in episodes]
    resolves = [bool(e.metadata.get("resolves_decision", False)) for e in episodes]
    excess = [float(e.metadata.get("excess_response_cost", 0.0)) for e in episodes]
    needs = [bool(e.metadata.get("needs_grounding", False)) for e in episodes]
    samples = [float(e.metadata.get("n_samples", 1)) for e in episodes]

    excess_needed = [x for x, need in zip(excess, needs) if need]
    asked = [e for e in episodes if e.query is not None]
    gold_match = None
    if asked:
        gold_match = sum(
            sorted(e.metadata.get("query_variables") or [])
            == sorted(e.metadata.get("gold_variables") or [])
            for e in asked
        ) / len(asked)

    mean = lambda xs: float(sum(xs) / len(xs))  # noqa: E731
    delta = mean(losses_before) - mean(losses_after)
    mean_bits = mean(bits)

    acc_ci = bits_ci = None
    if n_boot > 0:
        a = bootstrap_mean(correct, n_boot=n_boot, alpha=alpha, seed=seed)
        b = bootstrap_mean(bits, n_boot=n_boot, alpha=alpha, seed=seed + 1)
        acc_ci = (a.ci_low, a.ci_high)
        bits_ci = (b.ci_low, b.ci_high)

    return ArmQueryMetrics(
        baseline=baselines.pop(),
        n_items=len(episodes),
        accuracy=mean(correct),
        mean_final_loss=mean(losses_after),
        mean_loss_before=mean(losses_before),
        delta_distortion=delta,
        mean_external_bits=mean_bits,
        resolution_accuracy=mean([float(r) for r in resolves]),
        mean_excess_response_cost=mean(excess),
        mean_excess_where_grounding_needed=(
            mean(excess_needed) if excess_needed else None
        ),
        query_efficiency=information_efficiency(delta, mean_bits),
        gold_match_rate=gold_match,
        mean_samples=mean(samples),
        accuracy_ci=acc_ci,
        bits_ci=bits_ci,
    )


def build_analysis(
    cfg: GroundingConfig, episodes: Sequence[GroundingEpisode]
) -> dict[str, Any]:
    grouped: dict[str, list[GroundingEpisode]] = {}
    for e in episodes:
        grouped.setdefault(e.policy, []).append(e)

    arms = {
        name: aggregate_baseline(
            eps, n_boot=cfg.n_bootstrap, alpha=cfg.alpha, seed=cfg.seed
        )
        for name, eps in grouped.items()
    }
    order = [b for b in BASELINES if b in arms] + sorted(set(arms) - set(BASELINES))
    gate = evaluate_gate(cfg, grouped, arms)
    return {
        "experiment": cfg.name,
        "phase": "phase2",
        "task": cfg.task_kind,
        "task_version": ROUTE_CHOICE_TASK_VERSION,
        "n_episodes": len(episodes),
        "arms": [arms[name].to_dict() for name in order],
        "internal_compute_check": internal_compute_check(arms),
        "mechanism_check": compression_comparison(
            cfg, grouped, arms, MECHANISM_CHALLENGER, G2_BASELINE
        ),
        "gate": gate,
    }


def internal_compute_check(arms: Mapping[str, ArmQueryMetrics]) -> dict[str, Any]:
    """Does spending more internal computation substitute for grounding?

    The empirical content of `internal_reducibility = 0`. If self-consistency
    closed the gap, the whole premise -- that the missing information is not in
    M_A at all -- would be wrong, and that is worth checking rather than
    assuming.
    """
    b0 = arms.get("b0_no_grounding")
    b1 = arms.get("b1_self_consistency")
    if b0 is None or b1 is None:
        return {"available": False, "reason": "needs both B0 and B1"}
    gain = b1.accuracy - b0.accuracy
    return {
        "available": True,
        "b0_accuracy": b0.accuracy,
        "b1_accuracy": b1.accuracy,
        "b1_samples": b1.mean_samples,
        "accuracy_gain": gain,
        "external_bits_spent": b1.mean_external_bits,
        "note": (
            f"Self-consistency over {b1.mean_samples:.0f} samples moved accuracy "
            f"by {gain:+.4f} while spending no external bits. The withheld "
            "variable is not recoverable from the prompt, so internal "
            "computation cannot substitute for grounding -- which is what "
            "internal_reducibility = 0 asserts."
        ),
    }


def compression_comparison(
    cfg: GroundingConfig,
    grouped: Mapping[str, Sequence[GroundingEpisode]],
    arms: Mapping[str, ArmQueryMetrics],
    challenger_name: str,
    baseline_name: str,
) -> dict[str, Any]:
    """Does `challenger` cost fewer external bits at no worse decision quality?

    The conditional structure is the whole point: a query policy that reaches
    lower cost by deciding worse has compressed nothing. So decision quality is
    checked first, and the bit saving is only meaningful if it holds.
    """
    tol = cfg.gate.distortion_tolerance
    challenger = arms.get(challenger_name)
    baseline = arms.get(baseline_name)
    if challenger is None or baseline is None:
        return {
            "available": False,
            "reason": f"needs both {challenger_name} and {baseline_name}",
            "passed": False,
        }

    distortion_gap = challenger.mean_final_loss - baseline.mean_final_loss
    equal_quality = distortion_gap <= tol
    advantage = bootstrap_mean(
        [
            b.external_raw_bits - c.external_raw_bits
            for c, b in zip(grouped[challenger_name], grouped[baseline_name])
        ],
        name="bits_saved",
        n_boot=max(cfg.n_bootstrap, 1),
        alpha=cfg.alpha,
        seed=cfg.seed,
        null_value=0.0,
    )
    significant = advantage.point > 0 and advantage.ci_low > 0

    failures: list[str] = []
    if not equal_quality:
        failures.append(
            f"{challenger_name} final loss {challenger.mean_final_loss:.4f} "
            f"exceeds {baseline_name}'s {baseline.mean_final_loss:.4f} by "
            f"{distortion_gap:+.4f}, above the tolerance {tol}. A cost saving "
            "bought by deciding worse is not compression."
        )
    if not significant:
        failures.append(
            f"bits saved {advantage.point:+.4f} "
            f"[{advantage.ci_low:+.4f}, {advantage.ci_high:+.4f}] does not "
            "exclude zero"
        )

    return {
        "available": True,
        "challenger": challenger_name,
        "baseline": baseline_name,
        "passed": bool(equal_quality and significant),
        "equal_decision_quality": bool(equal_quality),
        "distortion_gap": distortion_gap,
        "distortion_tolerance": tol,
        "bits_saved": baseline.mean_external_bits - challenger.mean_external_bits,
        "bits_saved_ci": [advantage.ci_low, advantage.ci_high],
        "compression_ratio": (
            baseline.mean_external_bits / challenger.mean_external_bits
            if challenger.mean_external_bits > 0
            else None
        ),
        "failures": failures,
    }


def evaluate_gate(
    cfg: GroundingConfig,
    grouped: Mapping[str, Sequence[GroundingEpisode]],
    arms: Mapping[str, ArmQueryMetrics],
) -> dict[str, Any]:
    """Gate G2 (plan section 27): query compression at equal decision quality.

    `evaluated` is False on a simulated backend. G2 is a claim about a frozen
    off-the-shelf model, and a simulator cannot bear it either way -- so the
    comparison is still computed and reported, but it does not open or close the
    gate. What a simulated run *can* establish is that the metrics detect the
    effect when it is present, which is `mechanism_check`.
    """
    tol = cfg.gate.distortion_tolerance
    challenger = arms.get(G2_CHALLENGER)
    baseline = arms.get(G2_BASELINE)

    simulated = str(cfg.raw.get("backend", "grounding_simulator")) == "grounding_simulator"

    if challenger is None or baseline is None:
        return {
            "gate": "G2",
            "title": "Query compression: a targeted query costs less than generic escalation",
            "passed": False,
            "evaluated": not simulated,
            "backend_is_simulated": simulated,
            "failures": [
                f"needs both {G2_CHALLENGER} and {G2_BASELINE} in the run"
            ],
            "verdict": "Gate G2 could not be evaluated.",
        }

    distortion_gap = challenger.mean_final_loss - baseline.mean_final_loss
    comparison = compression_comparison(
        cfg, grouped, arms, G2_CHALLENGER, G2_BASELINE
    )
    simulated = str(cfg.raw.get("backend", "grounding_simulator")) == "grounding_simulator"
    return {
        "gate": "G2",
        "title": "Query compression: a targeted query costs less than generic escalation",
        "passed": comparison["passed"],
        "evaluated": not simulated,
        "backend_is_simulated": simulated,
        "equal_decision_quality": comparison["equal_decision_quality"],
        "distortion_gap": comparison["distortion_gap"],
        "distortion_tolerance": tol,
        "bits_saved": comparison["bits_saved"],
        "bits_saved_ci": comparison["bits_saved_ci"],
        "compression_ratio": comparison["compression_ratio"],
        "failures": comparison["failures"],
        "verdict": _g2_verdict(comparison["passed"], simulated, comparison["failures"]),
    }


def _g2_verdict(passed: bool, simulated: bool, failures: Sequence[str]) -> str:
    if not passed and simulated:
        return (
            "The B4-vs-B2 comparison does not hold for the *stand-in* "
            "synthesizer configured here, and on a simulated backend that is a "
            "statement about the stand-in rather than about the architecture. "
            "Read `mechanism_check` first: if the oracle query compresses at "
            "equal quality, the task and the metrics are sound and what failed "
            "is the query producer. " + " ".join(failures)
        )
    if not passed:
        return (
            "Gate G2 not met. Plan section 27: investigate whether the "
            "theory-to-architecture mapping is wrong before building further. "
            + " ".join(failures)
        )
    if simulated:
        return (
            "The synthesized query reaches the same decision quality as generic "
            "escalation for fewer external bits. **This run used the simulated "
            "backend, so gate G2 is NOT satisfied by it.** What it establishes "
            "is that the task, the query representation, the cost accounting and "
            "the metrics can detect the effect when it is present -- the "
            "precondition for testing it on a frozen LM, not the test itself. "
            "Set `backend: transformers` with a real checkpoint to evaluate G2."
        )
    return (
        "Gate G2 met on a frozen off-the-shelf model: at equal final distortion, "
        "a synthesized targeted query costs fewer external bits than generic "
        "human escalation. This is the first HG-v2 architectural result."
    )


# ------------------------------------------------------------------ figures ---
def write_figures(run: GroundingRunDir, analysis: Mapping[str, Any]) -> dict[str, bool]:
    arms = analysis["arms"]
    order = [a["baseline"] for a in arms]
    index = {name: float(i) for i, name in enumerate(order)}

    def series(field: str) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for a in arms:
            if a[field] is None:
                continue
            out[a["baseline"]] = [(index[a["baseline"]], float(a[field]))]
        return out

    wrote: dict[str, bool] = {}
    wrote["cost_vs_quality"] = plot_grounding_sweep(
        run.figure("phase2_bits_vs_accuracy.png"),
        {
            a["baseline"]: [(float(a["mean_external_bits"]), float(a["accuracy"]))]
            for a in arms
        },
        xlabel="mean external response cost (bits)",
        ylabel="decision accuracy",
        title="Phase 2: decision quality against what the source had to send",
        ylim_bottom=0.0,
    )
    wrote["bits"] = plot_grounding_sweep(
        run.figure("phase2_response_bits.png"),
        series("mean_external_bits"),
        xlabel="baseline (see report for names)",
        ylabel="mean external response cost (bits)",
        title="Phase 2: external information cost per baseline",
    )
    return wrote


# ------------------------------------------------------------------- report ---
def render_report(
    cfg: GroundingConfig, analysis: Mapping[str, Any], run: GroundingRunDir
) -> str:
    gate = analysis["gate"]
    simulated = gate.get("backend_is_simulated", True)
    lines: list[str] = [
        f"# {analysis['experiment']} — Phase 2 (gate G2)",
        "",
        f"Task `{analysis['task']}` version `{analysis['task_version']}`, "
        f"{analysis['n_episodes']} episodes, seed {cfg.seed}.",
        "",
    ]
    if simulated:
        lines += [
            "> **This run used the simulated backend `grounding_simulator`, so it "
            "does not evaluate gate G2.** It validates that the task, the query "
            "representation, the cost accounting and the query-quality metrics "
            "can detect query compression when it is present. Nothing below is "
            "evidence about any language model. Set `backend: transformers` with "
            "a real checkpoint to test G2.",
            "",
        ]
    lines += [
        "## The task",
        "",
        "Two routes; the rule travels in the prompt so the correct action never "
        "depends on the model's own preferences. Route A requires one traveller "
        "concession, and whether that concession is acceptable is withheld from "
        "the prompt and held by a simulated source. When route A is slower the "
        "rule closes without it (`H(J | M_A) = 0`); when A is faster the "
        "withheld bit decides (`H(J | M_A) = 1`). The source holds an opinion on "
        "every concession, so generic escalation costs one bit per variable "
        "while the gold query costs one bit total.",
        "",
        "## Baselines (plan section 17)",
        "",
        "| arm | grounding | external bits |",
        "|---|---|---|",
        "| `b0_no_grounding` | none | 0 |",
        "| `b1_self_consistency` | none, more internal sampling | 0 |",
        "| `b2_generic_ask` | source dumps every preference | one per variable |",
        "| `b3_oracle_targeted` | the gold query | 1 |",
        "| `b4_synthesized_query` | a synthesized query | depends on the query |",
        "",
        "Every ask arm asks on **every** item, so the query is the only thing "
        "that differs between them. Routing -- asking only where it pays -- is "
        "Phase 1's contribution and gate G1 already measured it.",
        "",
        "## Results",
        "",
        "| arm | accuracy | final loss | bits | resolution | Ω̂ (all) | Ω̂ (needs grounding) | ΔD/bit | gold match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in analysis["arms"]:
        def fmt(value: Any, spec: str = ".4f") -> str:
            return "—" if value is None else format(value, spec)

        lines.append(
            f"| `{a['baseline']}` | {a['accuracy']:.4f} | "
            f"{a['mean_final_loss']:.4f} | {a['mean_external_bits']:.2f} | "
            f"{a['resolution_accuracy']:.4f} | "
            f"{a['mean_excess_response_cost']:+.4f} | "
            f"{fmt(a['mean_excess_where_grounding_needed'], '+.4f')} | "
            f"{fmt(a['query_efficiency'])} | {fmt(a['gold_match_rate'])} |"
        )

    check = analysis.get("internal_compute_check") or {}
    if check.get("available"):
        lines += [
            "",
            "## Can internal computation substitute for grounding?",
            "",
            check["note"],
        ]

    mech = analysis.get("mechanism_check") or {}
    if mech.get("available"):
        lines += [
            "",
            "## Mechanism check: does a targeted query compress at all?",
            "",
            f"`{mech['challenger']}` vs `{mech['baseline']}`, which isolates the "
            "mechanism from the quality of whatever produced the query.",
            "",
            f"- equal decision quality: {mech['equal_decision_quality']} "
            f"(final-loss gap {mech['distortion_gap']:+.4f})",
            f"- bits saved: {mech['bits_saved']:+.4f} "
            f"[{mech['bits_saved_ci'][0]:+.4f}, {mech['bits_saved_ci'][1]:+.4f}]"
            + (
                f", a {mech['compression_ratio']:.2f}x reduction"
                if mech.get("compression_ratio")
                else ""
            ),
            "",
            (
                "**Detected.** The cost accounting and the query-quality metrics "
                "register targeted compression when it is present, which is the "
                "precondition for testing gate G2 on a real model."
                if mech["passed"]
                else "**Not detected.** Something is wrong with the task or the "
                "metrics, not with the synthesizer: " + " ".join(mech["failures"])
            ),
        ]

    lines += [
        "",
        "## Gate G2",
        "",
        f"**{'PASSED' if gate['passed'] else 'NOT MET'}**"
        + ("" if gate.get("evaluated") else " (not evaluated: simulated backend)")
        + f" — {gate['title']}.",
        "",
        f"- equal decision quality: {gate['equal_decision_quality']} "
        f"(final-loss gap {gate['distortion_gap']:+.4f}, tolerance "
        f"{gate['distortion_tolerance']})",
        f"- bits saved vs generic escalation: {gate['bits_saved']:+.4f} "
        f"[{gate['bits_saved_ci'][0]:+.4f}, {gate['bits_saved_ci'][1]:+.4f}]",
    ]
    if gate.get("compression_ratio"):
        lines.append(f"- compression ratio: {gate['compression_ratio']:.2f}x")
    lines += ["", gate["verdict"]]
    if gate["failures"]:
        lines += ["", "Failures:", ""] + [f"- {f}" for f in gate["failures"]]

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python experiments/hg_v2/run_phase2.py --config "
        + (str(cfg.source_path) if cfg.source_path else "<config>"),
        f"uv run python experiments/hg_v2/analyze.py --run-dir {run.root}",
        "```",
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------- entry points ---
def write_run(
    cfg: GroundingConfig,
    episodes: Sequence[GroundingEpisode],
    diagnostics: Mapping[str, Any],
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> GroundingRunDir:
    root = Path(output_root or cfg.output_root) / run_id(tag or cfg.name)
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")
    run = GroundingRunDir(root).create()
    write_yaml(run.config_path, cfg.resolved_dict())

    by_policy: dict[str, list[GroundingEpisode]] = {}
    for e in episodes:
        by_policy.setdefault(e.policy, []).append(e)
    counts = {
        policy: write_episodes(run.episodes(policy), eps)
        for policy, eps in sorted(by_policy.items())
    }

    write_json(run.root / "diagnostics.json", dict(diagnostics))
    write_json(
        run.manifest_path,
        {
            "run_id": root.name,
            "experiment": cfg.name,
            "phase": "phase2",
            "task": cfg.task_kind,
            "task_version": ROUTE_CHOICE_TASK_VERSION,
            "prompt_version": PROMPT_VERSION,
            "seed": cfg.seed,
            "policies": list(cfg.policies),
            "episode_counts": counts,
            "config_path": str(cfg.source_path) if cfg.source_path else None,
            "backend": diagnostics.get("backend"),
            "model_checkpoint": (cfg.raw.get("backend_options") or {}).get("checkpoint"),
            "git_commit": git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": environment(),
        },
    )
    return run


def analyze_run_dir(
    run_root: str | Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    from .config import load_grounding_config

    run = GroundingRunDir(run_root)
    cfg = load_grounding_config(config_path or run.config_path)
    policies = run.available_policies()
    if not policies:
        raise FileNotFoundError(f"no episode files under {run.root}/episodes")
    episodes: list[GroundingEpisode] = []
    for policy in policies:
        episodes.extend(read_episodes(run.episodes(policy), strict=True))

    analysis = build_analysis(cfg, episodes)
    write_json(run.analysis_path, analysis)
    write_figures(run, analysis)
    run.report_path.write_text(render_report(cfg, analysis, run), encoding="utf-8")
    return analysis


def run_phase2(
    cfg: GroundingConfig,
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> tuple[GroundingRunDir, dict[str, Any], list[str]]:
    episodes, diagnostics = run_sweep(cfg)
    run = write_run(cfg, episodes, diagnostics, output_root=output_root, tag=tag)
    live = build_analysis(cfg, episodes)
    from_disk = analyze_run_dir(run.root)
    return run, from_disk, compare_analyses(live, from_disk)


def compare_analyses(
    live: Mapping[str, Any], from_disk: Mapping[str, Any]
) -> list[str]:
    problems: list[str] = []
    live_arms = {a["baseline"]: a for a in live["arms"]}
    disk_arms = {a["baseline"]: a for a in from_disk["arms"]}
    if set(live_arms) != set(disk_arms):
        problems.append("arms differ across the round-trip")
        return problems
    for name, la in live_arms.items():
        for field, value in la.items():
            other = disk_arms[name][field]
            if isinstance(value, float) and isinstance(other, float):
                if math.isnan(value) and math.isnan(other):
                    continue
            if value != other:
                problems.append(
                    f"{name}: {field} = {value!r} live but {other!r} recomputed "
                    "from saved episodes"
                )
    if live["gate"]["passed"] != from_disk["gate"]["passed"]:
        problems.append("the gate verdict changed across the round-trip")
    return problems
