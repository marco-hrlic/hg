"""E2 and E3: routing and query synthesis in one system, on a frozen LM.

The improved plan's central experiment (sections 6 to 10). Phase 1 showed that
a learned router beats fixed ACT/ASK policies on a synthetic task. Phase 2
showed that a frozen LM can synthesize a query that costs less than generic
escalation. The two have never been run together, and the most informative
finding of the Phase 2 run is why that matters: the model asked a perfectly
targeted question 58% of the time and *never once* decided not to ask -- it
said NONE on none of 1000 items, and asked on all 490 whose prompt already
settled the answer. Every one of those bits was waste that a router removes.

The architecture under test is therefore

    route -> query -> external response -> decision

with the three measured separately.

The factorial (plan section 7)
------------------------------
    B0  no grounding
    B1  more internal compute (self-consistency)
    B2  generic ask on every item
    B3  oracle route + oracle targeted query
    B4  always ask + synthesized query          (reproduces Phase 2's B4)
    B5  learned route + synthesized query       (the integrated HG-v2 arm)
    B6  learned route + oracle targeted query   (isolates routing error)
    B7  oracle route + synthesized query        (isolates synthesis error)

B5 against B4 is gate H1. B6 and B7 are what make the gap decomposable: B5's
shortfall against B3 is the sum of a routing error B7 measures and a synthesis
error B6 measures, and without both, a single disappointing number would be
uninterpretable.

Why counterfactuals are computed once and arms assembled from them
-------------------------------------------------------------------
Every arm is a different *policy* over the same three model behaviours: what
the model answers ungrounded, what it answers given a generic dump, and what it
answers given a targeted answer. Running each arm as its own pass over the
model would multiply GPU time by the arm count and, worse, would make the arms
paired only by coincidence. So each behaviour is measured once per item and the
eight arms are assembled from the cache. Two consequences worth stating: the
arms are exactly item-paired by construction, and a lambda sweep costs nothing,
because repricing a saved counterfactual is arithmetic.

E3 (plan section 9) rides along
-------------------------------
Each configured query variant produces one extra always-ask arm, so the
minimality interventions are measured on the *same items* as the factorial and
against the same B2 and B3 references. Q0 is the unmodified synthesizer and is
the comparator; Q1 adds a minimality instruction, Q2 states a one-variable
budget, Q3 enforces that budget structurally.
"""

from __future__ import annotations

import contextlib
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..inference.base import GenerationSettings, get_backend
from ..io import git_commit, run_id, write_json, write_yaml
from ..metrics.calibration import calibration_report
from ..queries.lm_query_synthesizer import LMQuerySynthesizer, QUERY_VARIANTS
from ..queries.quality import QueryOutcome, oracle_gap_closure, summarize_queries
from ..queries.synthetic import GoldQuerySynthesizer, NoisyQuerySynthesizer
from ..report.figures import plot_grounding_sweep
from ..routing.decision_state import (
    LM_FEATURE_NAMES,
    DecisionState,
    agreement_features,
)
from ..routing.grounding_router import LogisticRouter, route_from_expected_cost
from .channels import QuerySpec, Response, Route
from .config import GroundingConfig
from .costs import (
    aggregate_route_arm,
    bootstrap_mean,
    total_route_cost,
    zero_one_loss,
)
from .episodes import GroundingEpisode, GroundingRunDir, read_episodes, write_episodes
from .phase0 import environment
from .route_choice import (
    ACTIONS,
    NO_CONSTRAINTS,
    PROMPT_VERSION,
    ROUTE_CHOICE_TASK_VERSION,
    RouteChoiceItem,
    TravellerSource,
    constraints_text,
    generate_route_choice_items,
    generic_query,
    gold_query,
    resolves_decision,
)

log = logging.getLogger(__name__)

TASK_NAME = "route_choice_integrated"

# --- the factorial ----------------------------------------------------------
ARM_B0 = "b0_no_grounding"
ARM_B1 = "b1_self_consistency"
ARM_B2 = "b2_generic_ask"
ARM_B3 = "b3_oracle_route_oracle_query"
ARM_B4 = "b4_always_ask_synth"
ARM_B5 = "b5_learned_route_synth"
ARM_B6 = "b6_learned_route_oracle_query"
ARM_B7 = "b7_oracle_route_synth"

FACTORIAL_ARMS = (ARM_B0, ARM_B1, ARM_B2, ARM_B3, ARM_B4, ARM_B5, ARM_B6, ARM_B7)
# E3 arms are named for the variant they run, so the arm name in a saved
# episode says which intervention produced it without consulting the config.
E3_ARM_PREFIX = "e3_"
E3_ARMS = tuple(f"{E3_ARM_PREFIX}{name}" for name in sorted(QUERY_VARIANTS))
ARMS = FACTORIAL_ARMS + E3_ARMS

# Where an arm's query comes from.
SOURCE_NONE = "none"
SOURCE_GENERIC = "generic"
SOURCE_ORACLE = "oracle"
SOURCE_SYNTH = "synth"

# How an arm decides whether to ask.
ROUTE_NEVER = "never"
ROUTE_ALWAYS = "always"
ROUTE_LEARNED = "learned"
ROUTE_ORACLE = "oracle"

# Gate H1's pair (plan section 18): routing must buy something on top of
# synthesis, at matched final distortion.
H1_CHALLENGER = ARM_B5
H1_BASELINE = ARM_B4
# E3's comparator (plan section 9).
E3_BASELINE_VARIANT = "q0_baseline"


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """An arm is a routing policy crossed with a query source."""

    name: str
    route_policy: str
    query_source: str
    # For E3 arms and for B4/B5/B7: which synthesizer variant supplies the query.
    variant: str | None = None
    # B1 replaces the ungrounded action with a self-consistency vote.
    uses_self_consistency: bool = False


def arm_specs(cfg: GroundingConfig) -> list[ArmSpec]:
    """Resolve the configured arm names into specs.

    The primary variant is the one B4, B5 and B7 use. It defaults to the
    unmodified synthesizer so the factorial reproduces Phase 2's B4 rather than
    quietly measuring an intervention.
    """
    primary = str(cfg.raw.get("query_variant") or E3_BASELINE_VARIANT)
    if primary not in QUERY_VARIANTS:
        raise ValueError(
            f"unknown query_variant {primary!r}; expected one of "
            f"{sorted(QUERY_VARIANTS)}"
        )
    table = {
        ARM_B0: ArmSpec(ARM_B0, ROUTE_NEVER, SOURCE_NONE),
        ARM_B1: ArmSpec(ARM_B1, ROUTE_NEVER, SOURCE_NONE, uses_self_consistency=True),
        ARM_B2: ArmSpec(ARM_B2, ROUTE_ALWAYS, SOURCE_GENERIC),
        ARM_B3: ArmSpec(ARM_B3, ROUTE_ORACLE, SOURCE_ORACLE),
        ARM_B4: ArmSpec(ARM_B4, ROUTE_ALWAYS, SOURCE_SYNTH, variant=primary),
        ARM_B5: ArmSpec(ARM_B5, ROUTE_LEARNED, SOURCE_SYNTH, variant=primary),
        ARM_B6: ArmSpec(ARM_B6, ROUTE_LEARNED, SOURCE_ORACLE),
        ARM_B7: ArmSpec(ARM_B7, ROUTE_ORACLE, SOURCE_SYNTH, variant=primary),
    }
    specs: list[ArmSpec] = []
    for name in cfg.policies:
        if name in table:
            specs.append(table[name])
            continue
        variant = name[len(E3_ARM_PREFIX):]
        if not name.startswith(E3_ARM_PREFIX) or variant not in QUERY_VARIANTS:
            raise ValueError(f"unknown arm {name!r}; expected one of {list(ARMS)}")
        specs.append(ArmSpec(name, ROUTE_ALWAYS, SOURCE_SYNTH, variant=variant))
    return specs


def required_variants(specs: Sequence[ArmSpec]) -> list[str]:
    """Which synthesizers actually have to be run for these arms."""
    return sorted({s.variant for s in specs if s.variant is not None})


# ------------------------------------------------------------ the item cache ---
@dataclass(slots=True)
class Grounded:
    """One counterfactual: what the model does given one query's answer."""

    query: QuerySpec | None
    response: Response | None
    action: str
    loss: float
    bits: float
    sound: bool
    exact_gold: bool

    @property
    def n_variables(self) -> int:
        return len(self.query.variable_names) if self.query else 0


@dataclass(slots=True)
class ItemCache:
    """Every model behaviour needed to assemble any arm, for one item."""

    item: RouteChoiceItem
    action_act: str
    loss_act: float
    features: dict[str, float]
    sc_action: str
    sc_loss: float
    grounded: dict[str, Grounded] = field(default_factory=dict)

    @property
    def needs_grounding(self) -> bool:
        return self.item.needs_grounding


@contextlib.contextmanager
def _generate_mode(backend):
    """Temporarily switch a backend from label ranking to sampled generation.

    Same reason as Phase 2's: `scoring_mode: rank` is a deterministic function
    of the prompt, so "k samples" from it would be one sample k times and the
    self-consistency features would be constant by construction.
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


def _label_features(probs: Mapping[str, float] | None, chosen: str) -> dict[str, float]:
    """The frozen model's own confidence, as router features.

    A model that has to guess a withheld variable should be less decided than
    one applying a rule it can see. Whether it actually is, is the empirical
    question behind falsification F1 -- so the features are recorded even when
    the backend supplies nothing and they degrade to constants, which shows up
    as a router that cannot separate anything rather than as a missing column.
    """
    if not probs:
        return {"label_margin": 0.0, "top_label_prob": 0.5}
    values = sorted(probs.values(), reverse=True)
    margin = float(values[0] - values[1]) if len(values) > 1 else float(values[0])
    return {
        "label_margin": abs(margin),
        "top_label_prob": float(probs.get(chosen, values[0])),
    }


def build_cache(
    items: Sequence[RouteChoiceItem],
    cfg: GroundingConfig,
    backend,
    variants: Sequence[str],
    diagnostics: dict[str, Any],
) -> list[ItemCache]:
    """Measure every model behaviour once. One pass per behaviour, not per arm."""
    base_items = [item.inference_item() for item in items]
    n_consistency = int(cfg.raw.get("self_consistency_samples", 5))

    # --- what the model does with the prompt as given ----------------------
    log.info("ungrounded pass over %d items", len(items))
    preds = backend.predict(base_items, _settings(cfg, temperature=0.0, seed=cfg.seed))
    by_id = {p.item_id: p for p in preds}

    # --- k sampled draws, for the internal-agreement features and for B1 ---
    votes: dict[str, Counter] = {item.item_id: Counter() for item in items}
    with _generate_mode(backend):
        for k in range(n_consistency):
            log.info("self-consistency sample %d/%d", k + 1, n_consistency)
            sampled = backend.predict(
                base_items,
                _settings(
                    cfg,
                    temperature=float(cfg.raw.get("temperature", 0.7)),
                    seed=cfg.seed + 1000 + k,
                ),
            )
            for p in sampled:
                if p.label is not None:
                    votes[p.item_id][p.label] += 1

    cache: list[ItemCache] = []
    for item in items:
        prediction = by_id[item.item_id]
        action = prediction.label or ACTIONS[0]
        counted = votes[item.item_id]
        # ACTIONS[0] is the "1" side of the binary encoding; the features are
        # symmetric in the labels, so which side is which does not matter as
        # long as it is fixed.
        observations = [1] * counted.get(ACTIONS[0], 0) + [0] * counted.get(ACTIONS[1], 0)
        features = {
            **agreement_features(observations or [1]),
            **_label_features(prediction.probs, action),
        }
        sc_action = counted.most_common(1)[0][0] if counted else action
        cache.append(
            ItemCache(
                item=item,
                action_act=action,
                loss_act=zero_one_loss(action, item.reference_action),
                features=features,
                sc_action=sc_action,
                sc_loss=zero_one_loss(sc_action, item.reference_action),
            )
        )

    # --- the grounded counterfactuals, one pass each -----------------------
    _add_grounded(cache, SOURCE_GENERIC, [generic_query(i.variables) for i in items], cfg, backend)
    _add_grounded(cache, SOURCE_ORACLE, [gold_query(i) for i in items], cfg, backend)
    for variant in variants:
        log.info("synthesizing queries: %s", variant)
        synthesizer = _synthesizer_for(cfg, backend, variant)
        batch = getattr(synthesizer, "synthesize_batch", None)
        if batch is not None:
            with _generate_mode(backend):
                queries = list(batch(items))
        else:
            queries = [synthesizer.synthesize(item) for item in items]
        if hasattr(synthesizer, "summary"):
            diagnostics.setdefault("synthesizer_summary", {})[variant] = (
                synthesizer.summary()
            )
        _add_grounded(cache, variant, queries, cfg, backend)
    return cache


def _add_grounded(
    cache: Sequence[ItemCache],
    key: str,
    queries: Sequence[QuerySpec | None],
    cfg: GroundingConfig,
    backend,
) -> None:
    """Run one query set over every item and record the counterfactual.

    A `None` query is the synthesizer answering NONE: the item is grounded by
    nothing, costs nothing, and keeps the ungrounded action. It must not be
    charged for a query that was never sent.
    """
    responses = [
        TravellerSource(entry.item).respond(query) if query is not None else None
        for entry, query in zip(cache, queries)
    ]
    grounded_items = [
        entry.item.inference_item(
            constraints_text(response.content) if response is not None else NO_CONSTRAINTS,
            revealed=tuple(response.content) if response is not None else (),
        )
        for entry, response in zip(cache, responses)
    ]
    log.info("grounded pass: %s", key)
    preds = backend.predict(grounded_items, _settings(cfg, temperature=0.0, seed=cfg.seed))
    by_id = {p.item_id: p for p in preds}

    for entry, query, response in zip(cache, queries, responses):
        item = entry.item
        action = (by_id[item.item_id].label or ACTIONS[0]) if query is not None else entry.action_act
        entry.grounded[key] = Grounded(
            query=query,
            response=response,
            action=action,
            loss=zero_one_loss(action, item.reference_action),
            bits=float(query.response_bits) if query is not None else 0.0,
            # Soundness is a property of the query, not of the answer the model
            # then gave: a sound query that the model misapplied is a capability
            # failure, and merging the two would make both unreadable.
            sound=resolves_decision(item, query) if query is not None
            else not item.needs_grounding,
            exact_gold=bool(
                query is not None
                and sorted(query.variable_names) == [item.requirement]
            ),
        )


def _synthesizer_for(cfg: GroundingConfig, backend, variant: str):
    opts = dict(cfg.raw.get("synthesizer") or {})
    kind = str(opts.pop("kind", "lm"))
    opts.setdefault("seed", cfg.seed)
    if kind == "gold":
        return GoldQuerySynthesizer()
    if kind == "noisy":
        opts.pop("on_parse_failure", None)
        opts.pop("max_new_tokens", None)
        opts.pop("temperature", None)
        return NoisyQuerySynthesizer(**opts)
    if kind == "lm":
        if backend is None:
            raise ValueError("the lm synthesizer needs the primary backend")
        return LMQuerySynthesizer(backend, variant=variant, **opts)
    raise ValueError(
        f"unknown synthesizer kind {kind!r}; expected 'gold', 'noisy' or 'lm'"
    )


# --------------------------------------------------------------- the router ---
def split_cache(
    cache: Sequence[ItemCache], *, train_fraction: float, seed: int
) -> tuple[list[ItemCache], list[ItemCache]]:
    """Fit split and evaluation split, keyed on the item id.

    Keyed rather than sliced so the split does not move when the item count
    changes, and so a re-run at a different n reports on a superset of the same
    evaluation items rather than on a differently-shuffled sample.
    """
    rng = np.random.default_rng([seed, 3301])
    keys = rng.random(len(cache))
    train = [c for c, k in zip(cache, keys) if k < train_fraction]
    evaluate = [c for c, k in zip(cache, keys) if k >= train_fraction]
    if not train or not evaluate:
        raise ValueError(
            f"degenerate split: {len(train)} train, {len(evaluate)} eval items"
        )
    return train, evaluate


def fit_router(train: Sequence[ItemCache], cfg: GroundingConfig) -> LogisticRouter:
    """P(acting now is wrong | the model's own internal signals).

    Supervised against the realized act-now error, exactly as in Phase 1, and
    on features that are all AI-private: k sampled votes and the frozen model's
    label distribution. Nothing about the withheld variable, the source, or the
    reference action enters -- a router that could see those would report its
    own leak as routing skill.
    """
    opts = dict(cfg.raw.get("router") or {})
    router = LogisticRouter(
        learning_rate=float(opts.get("learning_rate", 0.5)),
        n_steps=int(opts.get("n_steps", 2000)),
        l2=float(opts.get("l2", 1e-4)),
        feature_names=LM_FEATURE_NAMES,
    )
    return router.fit([c.features for c in train], [c.loss_act for c in train])


def expected_ask_bits(train: Sequence[ItemCache], source: str) -> float:
    """What a route decision should assume an ask will cost.

    Estimated on the training split rather than read off the item in hand: the
    router chooses *before* the query is synthesized, so using this item's
    actual query cost would give it information it cannot have. For the oracle
    query the cost is known to be one bit and no estimate is needed.
    """
    if source == SOURCE_ORACLE:
        return 1.0
    if source == SOURCE_GENERIC:
        return float(np.mean([c.grounded[SOURCE_GENERIC].bits for c in train]))
    bits = [c.grounded[source].bits for c in train]
    return float(np.mean(bits)) if bits else 1.0


# --------------------------------------------------------- assembling an arm ---
def _query_key(spec: ArmSpec) -> str:
    if spec.query_source == SOURCE_SYNTH:
        assert spec.variant is not None
        return spec.variant
    return spec.query_source


def choose_route(
    spec: ArmSpec,
    entry: ItemCache,
    *,
    lambda_bit: float,
    router: LogisticRouter | None,
    ask_bits: float,
) -> tuple[bool, float | None]:
    """(ask?, the router's score if it produced one)."""
    if spec.route_policy == ROUTE_NEVER:
        return False, None
    if spec.route_policy == ROUTE_ALWAYS:
        return True, None
    if spec.route_policy == ROUTE_LEARNED:
        if router is None:
            raise ValueError("a learned route needs a fitted router")
        p_wrong = float(router.predict_proba([entry.features])[0])
        route = route_from_expected_cost(
            p_wrong, lambda_bit=lambda_bit, ask_bits=ask_bits
        )
        return route is not Route.ACT, p_wrong
    if spec.route_policy == ROUTE_ORACLE:
        grounded = entry.grounded[_query_key(spec)]
        cost_act = entry.loss_act
        cost_ask = grounded.loss + lambda_bit * grounded.bits
        # Ties to ACT, matching every other oracle route in this repository: at
        # indifference, not spending the source's attention is the better
        # default.
        return cost_ask < cost_act, None
    raise ValueError(f"unknown route policy {spec.route_policy!r}")


def oracle_route_label(entry: ItemCache, grounded: Grounded, lambda_bit: float) -> str:
    """The clairvoyant route for this item and this query, as a Route value."""
    cost_act = total_route_cost(entry.loss_act, 0.0, lambda_bit)
    cost_ask = total_route_cost(grounded.loss, grounded.bits, lambda_bit)
    return Route.ACT.value if cost_act <= cost_ask else Route.ASK_TARGETED.value


def build_episode(
    spec: ArmSpec,
    entry: ItemCache,
    *,
    lambda_bit: float,
    router: LogisticRouter | None,
    ask_bits: float,
) -> GroundingEpisode:
    """One item under one arm at one price. The artifact of plan section 15."""
    item = entry.item
    reference = item.reference_action
    # `initial_action` is the ungrounded greedy answer for *every* arm,
    # including B1. That is what makes the distortion reduction comparable
    # between arms: B1's contribution is the move from that action to the
    # self-consistency vote, and recording the vote as the starting point
    # instead would report B1 as having reduced nothing.
    action_before, loss_before = entry.action_act, entry.loss_act

    asked, router_score = choose_route(
        spec, entry, lambda_bit=lambda_bit, router=router, ask_bits=ask_bits
    )
    key = _query_key(spec)
    grounded = entry.grounded.get(key)
    # An arm with no query source still needs an ask counterfactual, or its
    # routing regret is zero by construction: `aggregate_route_arm` takes the
    # clairvoyant route to be the cheaper of acting and asking, and an arm
    # whose "ask" costs nothing and achieves nothing makes never-asking
    # trivially optimal. The oracle query is the right stand-in -- the regret
    # then reads as "what a perfect router with a perfect query would have
    # achieved", which is the question B0 and B1 exist to lose.
    reference_ask = grounded if grounded is not None else entry.grounded.get(SOURCE_ORACLE)
    if grounded is None or grounded.query is None:
        # Either the arm has no query source at all, or the synthesizer said
        # NONE. Both are ACT, and both must cost zero bits.
        asked = False
    if asked:
        assert grounded is not None and grounded.query is not None
        route = Route.ASK_GENERIC if spec.query_source == SOURCE_GENERIC else Route.ASK_TARGETED
        action_after, loss_after = grounded.action, grounded.loss
        query, response = grounded.query, grounded.response
        sound, exact_gold = grounded.sound, grounded.exact_gold
        bits = grounded.bits
    elif spec.uses_self_consistency:
        route = Route.ACT
        action_after, loss_after = entry.sc_action, entry.sc_loss
        query = response = None
        sound, exact_gold = (not item.needs_grounding), False
        bits = 0.0
    else:
        route = Route.ACT
        action_after, loss_after = action_before, loss_before
        query = response = None
        # Acting is sound exactly when the prompt already determined the answer.
        sound, exact_gold = (not item.needs_grounding), False
        bits = 0.0

    deficit = item.residual_deficit_bits
    # Plan section 2.1: coordination overhead is defined only for a protocol
    # that met the requirement. An arm that left the deficit open reports null,
    # never a negative number that would read as beating the source-coding bound.
    protocol_sufficient = bool(sound)
    overhead = float(bits - deficit) if protocol_sufficient else None

    state = DecisionState(
        item_id=item.item_id,
        candidate_actions=list(ACTIONS),
        predicted_action=action_before,
        expected_loss_now=router_score,
        internal_reducibility=0.0,
        external_value=router_score,
        human_value=router_score,
        contestedness=0.0,
        consequence=1.0,
        # An estimator produced this, so it is not exact -- the one field that
        # keeps Phase 0's analytic decision states distinguishable from a
        # frozen model's.
        exact=False,
        features=dict(entry.features),
    )

    episode = GroundingEpisode(
        item_id=item.item_id,
        task=TASK_NAME,
        policy=spec.name,
        route=route.value,
        ai_private_state={
            "requirement": item.requirement,
            "concession_route": item.concession_route,
            "concession_faster": item.concession_faster,
            "time_a": item.time_a,
            "time_b": item.time_b,
            "prompt_version": PROMPT_VERSION,
            "ai_state_features": dict(entry.features),
        },
        hidden_reference_state={
            "acceptances": dict(item.acceptances),
            "decision_variable": reference,
            "grounding_required_gold": item.needs_grounding,
        },
        decision_state=state.to_dict(),
        initial_action=action_before,
        initial_loss=loss_before,
        final_action=action_after,
        final_loss=loss_after,
        metadata={
            "task_version": ROUTE_CHOICE_TASK_VERSION,
            "lambda_bit": lambda_bit,
            "arm": spec.name,
            "route_policy": spec.route_policy,
            "query_source": spec.query_source,
            "query_variant": spec.variant,
            # --- plan section 15's episode schema, under its own names -----
            "grounding_required_gold": item.needs_grounding,
            "initial_correct": loss_before == 0.0,
            "router_score": router_score,
            "router_route": route.value,
            "oracle_route": (
                oracle_route_label(entry, reference_ask, lambda_bit)
                if reference_ask is not None
                else Route.ACT.value
            ),
            "query_text": query.natural_language_query if query else None,
            "query_variables": list(query.variable_names) if query else [],
            "query_sound": sound,
            "query_exact_gold": exact_gold,
            "query_response_alphabet_size": (
                query.response_alphabet_size if query else None
            ),
            "external_response_bits": bits,
            "final_correct": loss_after == 0.0,
            "protocol_sufficient": protocol_sufficient,
            "coordination_overhead": overhead,
            # --- counterfactuals, so any lambda can be repriced and the
            # oracle route recomputed from the artifact alone ---------------
            "needs_grounding": item.needs_grounding,
            "residual_deficit_bits": deficit,
            "loss_act": entry.loss_act,
            "bits_act": 0.0,
            "loss_ask": reference_ask.loss if reference_ask else entry.loss_act,
            "bits_ask": reference_ask.bits if reference_ask else 0.0,
            "ask_bits_assumed": ask_bits,
            "excess_response_cost": bits - deficit,
            "n_samples": 1,
        },
    )
    if query is not None and response is not None:
        episode.attach_grounding(query, response, lambda_bit=lambda_bit)
    episode.validate()
    _check_json_finite(episode)
    return episode


def _check_json_finite(episode: GroundingEpisode) -> None:
    """No NaN anywhere in the artifact (plan section 15).

    "Undefined quantities must serialize as JSON null, never NaN." Checked at
    construction rather than trusted, because `json.dumps` writes a bare `NaN`
    token that most other readers reject, and by the time anyone notices, the
    run has finished.
    """
    for name, value in episode.metadata.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(
                f"{episode.item_id}: metadata[{name!r}] is {value}; undefined "
                "quantities must be None so they serialize as JSON null"
            )


def query_outcome(episode: GroundingEpisode) -> QueryOutcome:
    """The query-quality view of one episode (hg.queries.quality)."""
    md = episode.metadata
    return QueryOutcome(
        item_id=episode.item_id,
        needs_grounding=bool(md["grounding_required_gold"]),
        asked=episode.query is not None,
        n_variables=len(md.get("query_variables") or []),
        response_bits=float(episode.external_raw_bits),
        sound=bool(md["query_sound"]),
        exact_gold=bool(md["query_exact_gold"]),
        response_alphabet_size=md.get("query_response_alphabet_size"),
    )


# ------------------------------------------------------------------ running ---
def run_sweep(cfg: GroundingConfig) -> tuple[list[GroundingEpisode], dict[str, Any]]:
    """Measure the counterfactuals once, then assemble every arm at every price."""
    task = cfg.phase2_task
    if task is None:
        raise ValueError("E2 requires a `task` section with kind: route_choice_integrated")

    specs = arm_specs(cfg)
    variants = required_variants(specs)
    items = generate_route_choice_items(task)
    backend_name = str(cfg.raw.get("backend", "grounding_simulator"))
    backend_options = dict(cfg.raw.get("backend_options") or {})
    backend_options.setdefault("seed", cfg.seed)
    backend = get_backend(backend_name, **backend_options)
    backend.load()
    diagnostics: dict[str, Any] = {}
    try:
        cache = build_cache(items, cfg, backend, variants, diagnostics)
    finally:
        backend.unload()

    train, evaluate = split_cache(
        cache, train_fraction=cfg.train_fraction, seed=cfg.seed
    )
    router = fit_router(train, cfg)
    ask_bits = {
        source: expected_ask_bits(train, source)
        for source in (SOURCE_GENERIC, SOURCE_ORACLE, *variants)
    }
    log.info(
        "router fitted on %d items, evaluating %d; assumed ask cost %s",
        len(train), len(evaluate), ask_bits,
    )

    episodes: list[GroundingEpisode] = []
    for lambda_bit in cfg.lambda_grid:
        for spec in specs:
            bits = ask_bits.get(_query_key(spec), 1.0)
            for entry in evaluate:
                episodes.append(
                    build_episode(
                        spec, entry,
                        lambda_bit=lambda_bit, router=router, ask_bits=bits,
                    )
                )

    diagnostics.update(
        {
            "n_items": len(items),
            "n_train": len(train),
            "n_eval": len(evaluate),
            "train_fraction": cfg.train_fraction,
            "backend": backend_name,
            "backend_options": backend_options,
            "prompt_version": PROMPT_VERSION,
            "preference_variables": list(task.variables),
            "generic_ask_bits": task.generic_bits,
            "share_needing_grounding": float(
                sum(i.needs_grounding for i in items) / len(items)
            ),
            "router": router.to_dict(),
            "assumed_ask_bits": ask_bits,
            "query_variants": {v: QUERY_VARIANTS[v] for v in variants},
            "arms": [s.name for s in specs],
            "is_simulated": backend_name == "grounding_simulator",
            "example_prompt": items[0].prompt(),
            "calibration": router_calibration(evaluate, router, cfg),
        }
    )
    return episodes, diagnostics


# ------------------------------------------------- decision-state evaluation ---
def router_calibration(
    evaluate: Sequence[ItemCache], router: LogisticRouter, cfg: GroundingConfig
) -> dict[str, Any]:
    """Score the estimator itself, not the policy it feeds (plan sections 2.3, 13).

    Two things are measured and they answer different questions.

    `p_wrong` asks whether the estimate is *calibrated*: does 0.3 mean 0.3?
    That is the quantity a threshold is compared against, and routing regret
    cannot see it -- Phase 1's router and the exact posterior produced identical
    routes while disagreeing about the numbers.

    `value_of_asking` asks whether the estimate *orders* items correctly at a
    given price, against the empirical V_ask = L_ACT - L_ASK - lambda*B of plan
    section 13. Ordering is what a threshold policy actually consumes, and a
    poorly calibrated estimator with the right ordering is a repairable
    problem, while the reverse is not.
    """
    if not evaluate:
        return {"available": False, "reason": "no evaluation items"}
    primary = str(cfg.raw.get("query_variant") or E3_BASELINE_VARIANT)
    key = primary if primary in evaluate[0].grounded else SOURCE_ORACLE

    predicted = [float(router.predict_proba([c.features])[0]) for c in evaluate]
    outcome = [c.loss_act for c in evaluate]
    assumed = expected_ask_bits(evaluate, key)

    rows: list[dict[str, Any]] = []
    for lam in cfg.lambda_grid:
        actual = [
            c.loss_act - c.grounded[key].loss - lam * c.grounded[key].bits
            for c in evaluate
        ]
        estimated = [p - lam * assumed for p in predicted]
        report = calibration_report(predicted, outcome, against=actual)
        rows.append(
            {
                "lambda_bit": lam,
                "spearman_predicted_vs_actual_value": report.spearman,
                "mean_absolute_error_value": float(
                    np.mean(np.abs(np.asarray(estimated) - np.asarray(actual)))
                ),
                "mean_actual_value_of_asking": float(np.mean(actual)),
                "mean_estimated_value_of_asking": float(np.mean(estimated)),
                # Share of items where asking was actually worth it. When this
                # is 0 or 1 the routing problem is degenerate at this price and
                # no estimator can be distinguished from a constant.
                "share_where_asking_paid": float(np.mean([v > 0 for v in actual])),
            }
        )

    base = calibration_report(predicted, outcome)
    return {
        "available": True,
        "n": len(evaluate),
        "features": list(LM_FEATURE_NAMES),
        "query_source_scored": key,
        "assumed_ask_bits": assumed,
        "p_wrong": base.to_dict(),
        "value_of_asking": rows,
        "note": (
            f"The estimator's Brier score is {base.brier:.4f} against a base "
            f"error rate of {base.mean_observed:.4f}, with expected calibration "
            f"error {base.expected_calibration_error:.4f}. A router whose "
            "features carry nothing would score the base rate's variance, "
            f"{base.mean_observed * (1 - base.mean_observed):.4f}, so the gap "
            "between those two is what the internal signal is worth here."
        ),
    }


# ----------------------------------------------------------------- analysis ---
def aggregate_arm(
    episodes: Sequence[GroundingEpisode],
    *,
    n_available: int,
    n_boot: int = 0,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Route metrics, query-quality metrics and class-conditional accuracy."""
    route = aggregate_route_arm(episodes, n_boot=n_boot, alpha=alpha, seed=seed)
    quality = summarize_queries(
        [query_outcome(e) for e in episodes], n_available=n_available
    )

    references = [
        str(e.hidden_reference_state.get("decision_variable", "")) for e in episodes
    ]
    by_class = {
        target: [1.0 - e.final_loss for e, ref in zip(episodes, references) if ref == target]
        for target in ACTIONS
    }
    # Coordination overhead, averaged over the episodes where it is defined at
    # all. Plan section 2.1: an insufficient protocol contributes nothing here
    # rather than contributing a negative number.
    overheads = [
        e.metadata["coordination_overhead"]
        for e in episodes
        if e.metadata.get("coordination_overhead") is not None
    ]
    return {
        **route.to_dict(),
        "arm": episodes[0].policy,
        "accuracy": float(np.mean([1.0 - e.final_loss for e in episodes])),
        "accuracy_when_a_correct": float(np.mean(by_class["A"])) if by_class["A"] else None,
        "accuracy_when_b_correct": float(np.mean(by_class["B"])) if by_class["B"] else None,
        "share_answered_a": float(np.mean([e.final_action == "A" for e in episodes])),
        "protocol_sufficient_rate": float(
            np.mean([bool(e.metadata["protocol_sufficient"]) for e in episodes])
        ),
        "mean_coordination_overhead": float(np.mean(overheads)) if overheads else None,
        "coordination_overhead_defined_share": len(overheads) / len(episodes),
        "query": quality,
    }


def build_analysis(
    cfg: GroundingConfig,
    episodes: Sequence[GroundingEpisode],
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    task = cfg.phase2_task
    n_available = len(task.variables) if task is not None else 5

    grouped: dict[tuple[str, float], list[GroundingEpisode]] = {}
    for e in episodes:
        grouped.setdefault((e.policy, float(e.metadata["lambda_bit"])), []).append(e)

    arms = {
        key: aggregate_arm(
            eps, n_available=n_available,
            n_boot=cfg.n_bootstrap, alpha=cfg.alpha, seed=cfg.seed,
        )
        for key, eps in grouped.items()
    }
    order = sorted(arms, key=lambda k: (k[1], list(ARMS).index(k[0]) if k[0] in ARMS else 99))

    lambdas = sorted({lam for _, lam in grouped})
    h1 = [
        compare_cost(cfg, grouped, arms, H1_CHALLENGER, H1_BASELINE, lam)
        for lam in lambdas
    ]
    decomposition = [factorial_decomposition(arms, lam) for lam in lambdas]
    e3 = minimality_comparison(cfg, grouped, arms, lambdas)
    simulated = str(cfg.raw.get("backend", "grounding_simulator")) == "grounding_simulator"
    return {
        "experiment": cfg.name,
        "phase": "e2",
        "task": cfg.task_kind,
        "task_version": ROUTE_CHOICE_TASK_VERSION,
        "n_episodes": len(episodes),
        "lambda_grid": lambdas,
        "backend_is_simulated": simulated,
        "arms": [arms[key] | {"lambda_bit": key[1]} for key in order],
        "h1_routing_benefit": [row for row in h1 if row.get("available")],
        "factorial_decomposition": decomposition,
        "minimality": e3,
        "router_calibration": (diagnostics or {}).get("calibration"),
        "gates": evaluate_gates(cfg, h1, e3, simulated),
    }


def compare_cost(
    cfg: GroundingConfig,
    grouped: Mapping[tuple[str, float], Sequence[GroundingEpisode]],
    arms: Mapping[tuple[str, float], Mapping[str, Any]],
    challenger_name: str,
    baseline_name: str,
    lambda_bit: float,
) -> dict[str, Any]:
    """Does `challenger` send fewer external bits at no worse decision quality?

    Structured exactly like gate G2's comparison, and for the same reason: an
    arm that reaches lower cost by deciding worse has not compressed anything,
    so decision quality is checked first and the saving is only read if it holds.
    """
    challenger = arms.get((challenger_name, lambda_bit))
    baseline = arms.get((baseline_name, lambda_bit))
    if challenger is None or baseline is None:
        return {
            "available": False,
            "lambda_bit": lambda_bit,
            "reason": f"needs both {challenger_name} and {baseline_name}",
            "passed": False,
        }

    tol = cfg.gate.distortion_tolerance
    gap = challenger["mean_final_loss"] - baseline["mean_final_loss"]
    matched = gap <= tol
    # The paired interval on the distortion difference (plan section 22, item
    # 11). The gate criterion stays on the point estimate, matching gate G2 so
    # the two runs are comparable, but a point estimate that ignores its own
    # uncertainty is a weak reading of "matched" -- so `matched_within_ci` is
    # reported beside it, and it is the stricter statement.
    distortion = bootstrap_mean(
        [
            c.final_loss - b.final_loss
            for c, b in zip(
                grouped[(challenger_name, lambda_bit)],
                grouped[(baseline_name, lambda_bit)],
            )
        ],
        name="distortion_gap",
        n_boot=max(cfg.n_bootstrap, 1),
        alpha=cfg.alpha,
        seed=cfg.seed,
        null_value=0.0,
    )
    saved = bootstrap_mean(
        [
            b.external_raw_bits - c.external_raw_bits
            for c, b in zip(
                grouped[(challenger_name, lambda_bit)],
                grouped[(baseline_name, lambda_bit)],
            )
        ],
        name="bits_saved",
        n_boot=max(cfg.n_bootstrap, 1),
        alpha=cfg.alpha,
        seed=cfg.seed,
        null_value=0.0,
    )
    significant = saved.point > 0 and saved.ci_low > 0

    failures: list[str] = []
    if not matched:
        failures.append(
            f"{challenger_name} final loss {challenger['mean_final_loss']:.4f} "
            f"exceeds {baseline_name}'s {baseline['mean_final_loss']:.4f} by "
            f"{gap:+.4f}, above the tolerance {tol}"
        )
    if not significant:
        failures.append(
            f"bits saved {saved.point:+.4f} [{saved.ci_low:+.4f}, "
            f"{saved.ci_high:+.4f}] does not exclude zero"
        )
    return {
        "available": True,
        "lambda_bit": lambda_bit,
        "challenger": challenger_name,
        "baseline": baseline_name,
        "passed": bool(matched and significant),
        "equal_decision_quality": bool(matched),
        "distortion_gap": gap,
        "distortion_gap_ci": [distortion.ci_low, distortion.ci_high],
        "distortion_gap_p_value": distortion.p_value,
        "matched_within_ci": bool(distortion.ci_high <= tol),
        "distortion_tolerance": tol,
        "challenger_bits": challenger["mean_external_bits"],
        "baseline_bits": baseline["mean_external_bits"],
        "bits_saved": saved.point,
        "bits_saved_ci": [saved.ci_low, saved.ci_high],
        "challenger_ask_rate": challenger["query"]["ask_rate"],
        "baseline_ask_rate": baseline["query"]["ask_rate"],
        "failures": failures,
    }


def factorial_decomposition(
    arms: Mapping[tuple[str, float], Mapping[str, Any]], lambda_bit: float
) -> dict[str, Any]:
    """B4-B7 read as what they were designed to isolate (plan section 7).

    B5 is learned routing on a synthesized query. Its two ingredients can each
    be replaced by an oracle: B7 fixes the routing, B6 fixes the query. So

        routing_gap   = B5 - B7   what imperfect routing costs
        synthesis_gap = B5 - B6   what an imperfect query costs

    in mean external bits at matched quality. Reported together with final loss,
    because a "gap" that also moved decision quality is not a cost gap.
    """
    def cell(name: str) -> Mapping[str, Any] | None:
        return arms.get((name, lambda_bit))

    row: dict[str, Any] = {"lambda_bit": lambda_bit}
    for name in FACTORIAL_ARMS:
        arm = cell(name)
        if arm is None:
            continue
        row[name] = {
            "mean_external_bits": arm["mean_external_bits"],
            "mean_final_loss": arm["mean_final_loss"],
            "ask_rate": arm["query"]["ask_rate"],
            "soundness": arm["query"]["soundness"],
            "waste_bits_per_item": arm["query"]["waste_bits_per_item"],
        }
    b5, b6, b7 = cell(ARM_B5), cell(ARM_B6), cell(ARM_B7)
    if b5 and b7:
        row["routing_gap_bits"] = b5["mean_external_bits"] - b7["mean_external_bits"]
    if b5 and b6:
        row["synthesis_gap_bits"] = b5["mean_external_bits"] - b6["mean_external_bits"]

    generic, oracle = cell(ARM_B2), cell(ARM_B3)
    if generic and oracle:
        row["eta_q"] = {
            name: oracle_gap_closure(
                generic["mean_external_bits"],
                cell(name)["mean_external_bits"],
                oracle["mean_external_bits"],
            )
            for name in FACTORIAL_ARMS
            if cell(name) is not None
        }
        row["eta_q_note"] = (
            "eta_Q = (B_generic - B_arm) / (B_generic - B_oracle), with B_oracle "
            f"= {oracle['mean_external_bits']:.4f} taken from B3 -- an oracle "
            "*route* as well as an oracle query, so it is below one bit per "
            "item and eta_Q can exceed 1 for an arm that also routes. Read only "
            "at matched final distortion."
        )
    return row


def minimality_comparison(
    cfg: GroundingConfig,
    grouped: Mapping[tuple[str, float], Sequence[GroundingEpisode]],
    arms: Mapping[tuple[str, float], Mapping[str, Any]],
    lambdas: Sequence[float],
) -> dict[str, Any]:
    """E3: each minimality intervention against the unmodified synthesizer.

    Evaluated at one price and only one. The E3 arms all ask on every item, so
    their bits, their soundness and their final loss do not depend on lambda at
    all -- repeating the comparison across the grid would report the same
    numbers many times and invite reading a coincidence as a trend.
    """
    if not lambdas:
        return {"available": False, "reason": "no lambda grid"}
    lam = lambdas[0]
    baseline_name = f"{E3_ARM_PREFIX}{E3_BASELINE_VARIANT}"
    baseline = arms.get((baseline_name, lam))
    if baseline is None:
        return {
            "available": False,
            "reason": f"needs the {baseline_name} arm as the Q0 comparator",
        }

    generic = arms.get((ARM_B2, lam))
    oracle = arms.get((ARM_B3, lam))
    tol = cfg.gate.distortion_tolerance
    max_loss = cfg.gate.max_soundness_loss

    rows: list[dict[str, Any]] = []
    for name, _lam in sorted(arms):
        if _lam != lam or not name.startswith(E3_ARM_PREFIX) or name == baseline_name:
            continue
        arm = arms[(name, lam)]
        saved = bootstrap_mean(
            [
                b.external_raw_bits - c.external_raw_bits
                for c, b in zip(grouped[(name, lam)], grouped[(baseline_name, lam)])
            ],
            name="bits_saved",
            n_boot=max(cfg.n_bootstrap, 1),
            alpha=cfg.alpha,
            seed=cfg.seed,
            null_value=0.0,
        )
        soundness_loss = baseline["query"]["soundness"] - arm["query"]["soundness"]
        distortion_gap = arm["mean_final_loss"] - baseline["mean_final_loss"]
        distortion = bootstrap_mean(
            [
                c.final_loss - b.final_loss
                for c, b in zip(grouped[(name, lam)], grouped[(baseline_name, lam)])
            ],
            name="distortion_gap",
            n_boot=max(cfg.n_bootstrap, 1),
            alpha=cfg.alpha,
            seed=cfg.seed,
            null_value=0.0,
        )
        eta = (
            oracle_gap_closure(
                generic["mean_external_bits"],
                arm["mean_external_bits"],
                oracle["mean_external_bits"],
            )
            if generic and oracle
            else None
        )
        rows.append(
            {
                "arm": name,
                "variant": name[len(E3_ARM_PREFIX):],
                "description": QUERY_VARIANTS[name[len(E3_ARM_PREFIX):]]["description"],
                "mean_external_bits": arm["mean_external_bits"],
                "bits_saved_vs_q0": saved.point,
                "bits_saved_ci": [saved.ci_low, saved.ci_high],
                "significant": bool(saved.point > 0 and saved.ci_low > 0),
                "soundness": arm["query"]["soundness"],
                "soundness_loss_vs_q0": soundness_loss,
                "soundness_preserved": bool(soundness_loss <= max_loss),
                "accuracy": arm["accuracy"],
                "mean_final_loss": arm["mean_final_loss"],
                "distortion_gap_vs_q0": distortion_gap,
                "distortion_gap_ci": [distortion.ci_low, distortion.ci_high],
                "quality_preserved": bool(distortion_gap <= tol),
                "quality_preserved_within_ci": bool(distortion.ci_high <= tol),
                "eta_q": eta,
                "alpha_minimal": arm["query"]["alpha_minimal"],
                "share_full_escalation": arm["query"]["share_full_escalation"],
                "variables_distribution": arm["query"]["variables_distribution"],
                # H2's conjunction, spelled out: cheaper, still sound, still
                # deciding as well. Any one of the three failing means the
                # intervention traded rather than improved.
                "improves": bool(
                    saved.point > 0
                    and saved.ci_low > 0
                    and soundness_loss <= max_loss
                    and distortion_gap <= tol
                ),
            }
        )

    return {
        "available": True,
        "lambda_bit": lam,
        "baseline_arm": baseline_name,
        "baseline": {
            "mean_external_bits": baseline["mean_external_bits"],
            "soundness": baseline["query"]["soundness"],
            "accuracy": baseline["accuracy"],
            "alpha_minimal": baseline["query"]["alpha_minimal"],
            "share_full_escalation": baseline["query"]["share_full_escalation"],
            "variables_distribution": baseline["query"]["variables_distribution"],
            "eta_q": (
                oracle_gap_closure(
                    generic["mean_external_bits"],
                    baseline["mean_external_bits"],
                    oracle["mean_external_bits"],
                )
                if generic and oracle
                else None
            ),
        },
        "interventions": rows,
    }


def evaluate_gates(
    cfg: GroundingConfig,
    h1_rows: Sequence[Mapping[str, Any]],
    e3: Mapping[str, Any],
    simulated: bool,
) -> dict[str, Any]:
    """Gates H1 and H2 (improved plan section 18).

    `evaluated` is False on a simulated backend, for the same reason gate G2
    carries that flag: these are claims about a frozen off-the-shelf model, and
    a stand-in cannot bear them either way. The comparisons are still computed
    and reported -- a simulated run establishes that the metrics can detect the
    effect when it is present, which is the precondition for the test rather
    than the test.
    """
    winning = [row["lambda_bit"] for row in h1_rows if row.get("passed")]
    required = cfg.gate.min_winning_lambdas
    h1_passed = len(winning) >= required
    h1_failures: list[str] = []
    if not h1_passed:
        h1_failures.append(
            f"{H1_CHALLENGER} sent fewer external bits than {H1_BASELINE} at "
            f"matched distortion at {len(winning)} of {len(h1_rows)} prices "
            f"({winning}); {required} are required"
        )
        h1_failures.extend(
            f"lambda={row['lambda_bit']}: {f}"
            for row in h1_rows
            for f in row.get("failures", [])
        )

    improving = [r["arm"] for r in (e3.get("interventions") or []) if r["improves"]]
    h2_available = bool(e3.get("available")) and bool(e3.get("interventions"))
    h2_passed = bool(improving)
    h2_failures: list[str] = []
    if not h2_available:
        h2_failures.append(
            "no minimality interventions were run; H2 needs at least the "
            f"{E3_ARM_PREFIX}{E3_BASELINE_VARIANT} arm and one other variant"
        )
    elif not h2_passed:
        h2_failures.append(
            "no intervention reduced response cost without losing soundness or "
            "decision quality: "
            + "; ".join(
                f"{r['variant']} saved {r['bits_saved_vs_q0']:+.3f} bits "
                f"(sound {r['soundness']:.3f}, loss gap "
                f"{r['distortion_gap_vs_q0']:+.4f})"
                for r in e3.get("interventions") or []
            )
        )

    return {
        "H1": {
            "gate": "H1",
            "title": "Integrated routing benefit: routing on top of synthesis",
            "passed": h1_passed,
            "evaluated": not simulated,
            "backend_is_simulated": simulated,
            "winning_lambdas": winning,
            "n_lambdas": len(h1_rows),
            "min_winning_lambdas": required,
            "failures": h1_failures,
            "verdict": (
                "At matched final distortion the learned router plus a "
                "synthesized query costs the source fewer bits than always "
                "asking with the same query, over a range of prices. Routing "
                "and query synthesis compose (hypothesis H-A)."
                if h1_passed
                else "Gate H1 not met. Falsification F1 is the reading to check "
                "first: routing that worked on the synthetic task may not "
                "transfer to the LM's own internal signal. Read the router "
                "calibration before the arm table -- an estimator with no "
                "discrimination cannot be rescued by a different threshold."
            ),
        },
        "H2": {
            "gate": "H2",
            "title": "Minimality improvement: a cheaper query at equal soundness",
            "passed": h2_passed,
            "evaluated": (not simulated) and h2_available,
            "backend_is_simulated": simulated,
            "improving_arms": improving,
            "max_soundness_loss": cfg.gate.max_soundness_loss,
            "failures": h2_failures,
            "verdict": (
                f"Interventions {improving} reduced external response cost "
                "without materially reducing soundness or decision quality "
                "(hypothesis H-B)."
                if h2_passed
                else "Gate H2 not met. Check whether the interventions failed "
                "to move the cost distribution at all, or moved it by breaking "
                "soundness -- falsification F2. The mixture weights "
                "(alpha_minimal, share_full_escalation) separate those two."
            ),
        },
    }


# ------------------------------------------------------------------- writing ---
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
            "phase": "e2",
            "task": cfg.task_kind,
            "task_version": ROUTE_CHOICE_TASK_VERSION,
            "prompt_version": PROMPT_VERSION,
            "seed": cfg.seed,
            "lambda_grid": list(cfg.lambda_grid),
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


def read_run_episodes(run: GroundingRunDir) -> list[GroundingEpisode]:
    policies = run.available_policies()
    if not policies:
        raise FileNotFoundError(f"no episode files under {run.root}/episodes")
    episodes: list[GroundingEpisode] = []
    for policy in policies:
        episodes.extend(read_episodes(run.episodes(policy), strict=True))
    return episodes


# ------------------------------------------------------------------ figures ---
def write_figures(run: GroundingRunDir, analysis: Mapping[str, Any]) -> dict[str, bool]:
    arms = analysis["arms"]
    names = sorted({a["arm"] for a in arms})

    def series(field: str, *, inner: str | None = None) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for name in names:
            points = []
            for a in arms:
                if a["arm"] != name:
                    continue
                value = a[inner][field] if inner else a[field]
                if value is None:
                    continue
                points.append((float(a["lambda_bit"]), float(value)))
            if points:
                out[name] = sorted(points)
        return out

    wrote: dict[str, bool] = {}
    wrote["bits"] = plot_grounding_sweep(
        run.figure("e2_response_bits.png"),
        series("mean_external_bits"),
        xlabel="lambda = price of one external bit",
        ylabel="mean external response cost (bits)",
        title="E2: what the source had to send, per arm",
    )
    wrote["cost"] = plot_grounding_sweep(
        run.figure("e2_total_cost.png"),
        series("mean_total_cost"),
        xlabel="lambda = price of one external bit",
        ylabel="mean total cost  C = L + lambda*B",
        title="E2: total cost of routing and asking",
    )
    wrote["ask_rate"] = plot_grounding_sweep(
        run.figure("e2_ask_rate.png"),
        series("ask_rate", inner="query"),
        xlabel="lambda = price of one external bit",
        ylabel="fraction of items routed to ASK",
        title="E2: who asks, and how the price changes it",
    )
    wrote["waste"] = plot_grounding_sweep(
        run.figure("e2_waste.png"),
        series("waste_bits_per_item", inner="query"),
        xlabel="lambda = price of one external bit",
        ylabel="bits spent on items the prompt already settled",
        title="E2: waste rate -- bits bought for decisions that were already made",
    )
    calibration = analysis.get("router_calibration") or {}
    if calibration.get("available"):
        curve = [
            (row["mean_predicted"], row["observed_frequency"])
            for row in calibration["p_wrong"]["curve"]
            if row["n"]
        ]
        wrote["calibration"] = plot_grounding_sweep(
            run.figure("e2_router_calibration.png"),
            {
                "router": sorted(curve),
                "perfect": [(0.0, 0.0), (1.0, 1.0)],
            },
            xlabel="predicted P(acting now is wrong)",
            ylabel="observed frequency",
            title="E2: is the decision-state estimator calibrated?",
        )
    return wrote


# ------------------------------------------------------------------- report ---
def _fmt(value: Any, spec: str = ".4f") -> str:
    return "—" if value is None else format(value, spec)


def render_report(
    cfg: GroundingConfig, analysis: Mapping[str, Any], run: GroundingRunDir
) -> str:
    gates = analysis["gates"]
    h1, h2 = gates["H1"], gates["H2"]
    simulated = analysis["backend_is_simulated"]
    lambdas = analysis["lambda_grid"]

    lines: list[str] = [
        f"# {analysis['experiment']} — E2/E3 (gates H1, H2)",
        "",
        f"Task `{analysis['task']}` version `{analysis['task_version']}`, "
        f"{analysis['n_episodes']} episodes, seed {cfg.seed}, "
        f"{len(lambdas)} prices.",
        "",
    ]
    if simulated:
        lines += [
            "> **This run used the simulated backend `grounding_simulator`, so "
            "it does not evaluate gates H1 or H2.** It checks that the "
            "factorial, the cost accounting, the router calibration and the "
            "minimality metrics detect what they are meant to detect. Nothing "
            "below is evidence about a language model.",
            "",
        ]
    lines += [
        "## The architecture under test",
        "",
        "```text",
        "route -> query -> external response -> decision",
        "```",
        "",
        "Phase 1 measured the first arrow on a synthetic task and Phase 2 the "
        "second on a frozen LM. This run is the first time they are in one "
        "system. The Phase 2 result that motivates it: the model asked a "
        "perfectly targeted question on 58% of items and never once declined "
        "to ask, including on the 49% of items whose prompt already determined "
        "the answer.",
        "",
        "## Arms (plan section 7)",
        "",
        "| arm | route | query |",
        "|---|---|---|",
        "| `b0_no_grounding` | never | — |",
        "| `b1_self_consistency` | never | — (more internal compute) |",
        "| `b2_generic_ask` | always | every preference the source holds |",
        "| `b3_oracle_route_oracle_query` | oracle | the gold one-bit query |",
        "| `b4_always_ask_synth` | always | synthesized |",
        "| `b5_learned_route_synth` | learned | synthesized (**the claim**) |",
        "| `b6_learned_route_oracle_query` | learned | gold (isolates routing) |",
        "| `b7_oracle_route_synth` | oracle | synthesized (isolates synthesis) |",
        "",
        "## Results",
        "",
        "`Ω̂` is mean excess response cost given soundness; `waste` is bits "
        "spent on items the prompt already settled.",
        "",
        "| λ | arm | acc | acc\\|A | acc\\|B | bits | ask rate | sound | Ω̂\\|sound | waste | C = L+λB | regret |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in analysis["arms"]:
        q = a["query"]
        lines.append(
            f"| {a['lambda_bit']:.2f} | `{a['arm']}` | {a['accuracy']:.4f} | "
            f"{_fmt(a['accuracy_when_a_correct'], '.3f')} | "
            f"{_fmt(a['accuracy_when_b_correct'], '.3f')} | "
            f"{a['mean_external_bits']:.3f} | {q['ask_rate']:.3f} | "
            f"{q['soundness']:.3f} | "
            f"{_fmt(q['mean_excess_bits_given_sound'], '+.3f')} | "
            f"{q['waste_bits_per_item']:.3f} | {a['mean_total_cost']:.4f} | "
            f"{a['routing_regret']:.4f} |"
        )

    lines += [
        "",
        "## Gate H1 — does routing buy anything on top of synthesis?",
        "",
        f"`{H1_CHALLENGER}` against `{H1_BASELINE}`: the same synthesized query, "
        "asked everywhere versus asked where the router says it pays. Positive "
        "means the router is cheaper for the source.",
        "",
        "Final distortion is compared as an item-paired difference with its own "
        "interval, not only as a point gap: `ΔD` is "
        "`L(learned route) − L(always ask)`, so a positive value means the "
        "routed arm decided worse.",
        "",
        "| λ | B(learned route) | B(always ask) | bits saved | 95% CI | ΔD | ΔD 95% CI | matched | wins |",
        "|---:|---:|---:|---:|---|---:|---|:---:|:---:|",
    ]
    for row in analysis["h1_routing_benefit"]:
        lines.append(
            f"| {row['lambda_bit']:.2f} | {row['challenger_bits']:.3f} | "
            f"{row['baseline_bits']:.3f} | {row['bits_saved']:+.3f} | "
            f"[{row['bits_saved_ci'][0]:+.3f}, {row['bits_saved_ci'][1]:+.3f}] | "
            f"{row['distortion_gap']:+.4f} | "
            f"[{row['distortion_gap_ci'][0]:+.4f}, {row['distortion_gap_ci'][1]:+.4f}] | "
            + ("yes" if row["equal_decision_quality"] else "no")
            + ("" if row["matched_within_ci"] else " (point only)")
            + f" | {'**yes**' if row['passed'] else 'no'} |"
        )

    decomposition = analysis.get("factorial_decomposition") or []
    if decomposition:
        lines += [
            "",
            "## Factorial decomposition (B4–B7)",
            "",
            "B5's shortfall against the oracle arm splits into a routing error "
            "and a synthesis error, and the point of running B6 and B7 is that "
            "the split is measured rather than argued. `routing_gap` is "
            "B5 − B7 (what imperfect routing costs, holding the query fixed); "
            "`synthesis_gap` is B5 − B6 (what an imperfect query costs, holding "
            "the route fixed).",
            "",
            "| λ | routing gap (bits) | synthesis gap (bits) | η_Q(B5) | η_Q(B4) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in decomposition:
            eta = row.get("eta_q") or {}
            lines.append(
                f"| {row['lambda_bit']:.2f} | "
                f"{_fmt(row.get('routing_gap_bits'), '+.3f')} | "
                f"{_fmt(row.get('synthesis_gap_bits'), '+.3f')} | "
                f"{_fmt(eta.get(ARM_B5), '.3f')} | {_fmt(eta.get(ARM_B4), '.3f')} |"
            )
        note = next((r.get("eta_q_note") for r in decomposition if r.get("eta_q_note")), None)
        if note:
            lines += ["", note]

    calibration = analysis.get("router_calibration") or {}
    if calibration.get("available"):
        p = calibration["p_wrong"]
        lines += [
            "",
            "## Decision-state estimator (plan sections 2.3 and 13)",
            "",
            "Routing regret alone cannot see estimator error: a thresholded "
            "policy is unchanged by any estimate that stays on the same side of "
            "the threshold. So the estimate is scored directly.",
            "",
            f"- features: `{'`, `'.join(calibration['features'])}`",
            f"- Brier {p['brier']:.4f}, ECE {p['expected_calibration_error']:.4f}, "
            f"MCE {p['max_calibration_error']:.4f}, MAE {p['mean_absolute_error']:.4f}",
            f"- mean predicted {p['mean_predicted']:.4f} against an observed "
            f"act-now error rate of {p['mean_observed']:.4f}",
            "",
            calibration["note"],
            "",
            "| λ | Spearman(p̂, V_ask) | MAE(V̂_ask, V_ask) | mean V_ask | share where asking paid |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in calibration["value_of_asking"]:
            lines.append(
                f"| {row['lambda_bit']:.2f} | "
                f"{_fmt(row['spearman_predicted_vs_actual_value'], '.3f')} | "
                f"{row['mean_absolute_error_value']:.4f} | "
                f"{row['mean_actual_value_of_asking']:+.4f} | "
                f"{row['share_where_asking_paid']:.3f} |"
            )

    e3 = analysis.get("minimality") or {}
    if e3.get("available"):
        base = e3["baseline"]
        lines += [
            "",
            "## E3 — minimality interventions (plan sections 9 and 10)",
            "",
            f"All arms ask on every item, so this table isolates the *query* "
            f"from the route. Comparator `{e3['baseline_arm']}` is the "
            "unmodified synthesizer.",
            "",
            "| arm | bits | saved vs Q0 | 95% CI | sound | Δsound | acc | ΔD | α minimal | full escalation | η_Q | improves |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
            f"| `{e3['baseline_arm']}` (Q0) | {base['mean_external_bits']:.3f} | — | — | "
            f"{base['soundness']:.3f} | — | {base['accuracy']:.4f} | — | "
            f"{_fmt(base['alpha_minimal'], '.3f')} | "
            f"{_fmt(base['share_full_escalation'], '.3f')} | "
            f"{_fmt(base['eta_q'], '.3f')} | — |",
        ]
        for row in e3["interventions"]:
            lines.append(
                f"| `{row['arm']}` | {row['mean_external_bits']:.3f} | "
                f"{row['bits_saved_vs_q0']:+.3f} | "
                f"[{row['bits_saved_ci'][0]:+.3f}, {row['bits_saved_ci'][1]:+.3f}] | "
                f"{row['soundness']:.3f} | {row['soundness_loss_vs_q0']:+.3f} | "
                f"{row['accuracy']:.4f} | {row['distortion_gap_vs_q0']:+.4f} | "
                f"{_fmt(row['alpha_minimal'], '.3f')} | "
                f"{_fmt(row['share_full_escalation'], '.3f')} | "
                f"{_fmt(row['eta_q'], '.3f')} | "
                f"{'**yes**' if row['improves'] else 'no'} |"
            )
        lines += [
            "",
            "Variables named per query, which is the distribution plan section "
            "10 asks to be reported rather than summarized:",
            "",
            "| arm | " + " | ".join(f"{k} var" for k in ("1", "2", "3", "4", "5")) + " |",
            "|---|" + "---:|" * 5,
        ]
        for name, dist in [
            (e3["baseline_arm"], base["variables_distribution"]),
            *[(r["arm"], r["variables_distribution"]) for r in e3["interventions"]],
        ]:
            total = sum(dist.values()) or 1
            cells = " | ".join(
                f"{dist.get(str(k), 0) / total:.3f}" for k in range(1, 6)
            )
            lines.append(f"| `{name}` | {cells} |")

    for gate in (h1, h2):
        lines += [
            "",
            f"## Gate {gate['gate']}",
            "",
            f"**{'PASSED' if gate['passed'] else 'NOT MET'}**"
            + ("" if gate.get("evaluated") else " (not evaluated: simulated backend)")
            + f" — {gate['title']}.",
            "",
            gate["verdict"],
        ]
        if gate["failures"]:
            lines += ["", "Failures:", ""] + [f"- {f}" for f in gate["failures"][:10]]

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python experiments/hg_v2/run_e2.py --config "
        + (str(cfg.source_path) if cfg.source_path else "<config>"),
        f"uv run python experiments/hg_v2/analyze.py --run-dir {run.root}",
        "```",
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------- entry points ---
def analyze_run_dir(
    run_root: str | Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    from ..io import read_json
    from .config import load_grounding_config

    run = GroundingRunDir(run_root)
    cfg = load_grounding_config(config_path or run.config_path)
    episodes = read_run_episodes(run)
    diagnostics_path = run.root / "diagnostics.json"
    diagnostics = read_json(diagnostics_path) if diagnostics_path.exists() else {}
    analysis = build_analysis(cfg, episodes, diagnostics)
    write_json(run.analysis_path, analysis)
    write_figures(run, analysis)
    run.report_path.write_text(render_report(cfg, analysis, run), encoding="utf-8")
    return analysis


def run_e2(
    cfg: GroundingConfig,
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> tuple[GroundingRunDir, dict[str, Any], list[str]]:
    """Run the factorial, save it, and analyse it from disk.

    Same discipline as every other phase here: the reported analysis is the one
    recomputed from the written episodes, and it is compared against the
    in-memory result rather than trusted.
    """
    episodes, diagnostics = run_sweep(cfg)
    run = write_run(cfg, episodes, diagnostics, output_root=output_root, tag=tag)
    live = build_analysis(cfg, episodes, diagnostics)
    from_disk = analyze_run_dir(run.root)
    return run, from_disk, compare_analyses(live, from_disk)


def compare_analyses(
    live: Mapping[str, Any], from_disk: Mapping[str, Any]
) -> list[str]:
    problems: list[str] = []
    live_arms = {(a["arm"], a["lambda_bit"]): a for a in live["arms"]}
    disk_arms = {(a["arm"], a["lambda_bit"]): a for a in from_disk["arms"]}
    if set(live_arms) != set(disk_arms):
        problems.append("arms differ across the round-trip")
        return problems
    for key, la in live_arms.items():
        for field_name, value in la.items():
            other = disk_arms[key][field_name]
            if isinstance(value, float) and isinstance(other, float):
                if math.isnan(value) and math.isnan(other):
                    continue
            if value != other:
                problems.append(
                    f"{key[0]}@lambda={key[1]}: {field_name} = {value!r} live "
                    f"but {other!r} recomputed from saved episodes"
                )
    for gate in ("H1", "H2"):
        if live["gates"][gate]["passed"] != from_disk["gates"][gate]["passed"]:
            problems.append(f"the {gate} verdict changed across the round-trip")
    return problems
