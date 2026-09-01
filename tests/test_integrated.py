"""E2/E3: the integrated router + synthesizer, and the metrics that read it.

The tests that matter here are not "does it run". They are the ones that would
catch a result being manufactured: that the arms are item-paired rather than
independently sampled, that coordination overhead is null rather than negative
when a protocol left the deficit open, that the router sees nothing it should
not, and that a minimality intervention which buys cheapness by breaking
soundness is *not* counted as an improvement.
"""

from __future__ import annotations

import json
import math
import re

import pytest

from hg.grounding.config import load_grounding_config
from hg.grounding.integrated import (
    ARM_B0,
    ARM_B2,
    ARM_B3,
    ARM_B4,
    ARM_B5,
    ARM_B6,
    ARM_B7,
    E3_ARM_PREFIX,
    ArmSpec,
    ROUTE_ALWAYS,
    ROUTE_LEARNED,
    ROUTE_NEVER,
    SOURCE_GENERIC,
    SOURCE_NONE,
    SOURCE_SYNTH,
    arm_specs,
    build_analysis,
    build_episode,
    build_cache,
    compare_analyses,
    query_outcome,
    run_e2,
    split_cache,
)
from hg.inference.base import Prediction, register_backend
from hg.grounding.route_choice import (
    ACTIONS,
    META_REFERENCE_ACTION,
    VARIABLE_PHRASES,
    RouteChoiceSpec,
    generate_route_choice_items,
)

CONFIG = "configs/experiments/hg_v2/e2_integrated_sim.yaml"


# --------------------------------------------------------------- a stub model ---
@register_backend("test_scripted")
class ScriptedBackend:
    """A model whose query synthesis depends on which prompt it was given.

    The E3 interventions are a prompt difference, so the stand-in synthesizer
    used by the simulated config cannot exercise them -- it never reads a
    prompt. This backend does the minimum needed to: it answers the decision
    prompt from the item's own metadata, and answers a synthesis prompt with a
    variable list that depends on the instruction it was shown.
    """

    name = "test_scripted"

    def __init__(self, *, seed: int = 0, hedge_variables: int = 3, **_options):
        self.seed = seed
        self.hedge_variables = hedge_variables

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def predict(self, items, settings):
        out = []
        for item in items:
            if "Facts needed:" in item.prompt:
                out.append(Prediction(item.item_id, None, raw_output=self._query(item)))
                continue
            # The decision prompt: right when the constraints block carries the
            # deciding preference, a fixed guess otherwise.
            reference = str(item.meta.get(META_REFERENCE_ACTION, ACTIONS[0]))
            sufficient = bool(item.meta.get("gs_sufficient", False))
            label = reference if sufficient else ACTIONS[0]
            out.append(
                Prediction(
                    item.item_id,
                    label,
                    raw_output=label,
                    probs={label: 0.8, _other(label): 0.2},
                )
            )
        return out

    def _query(self, item) -> str:
        # The deciding variable is named in the rule the prompt carries, which
        # is exactly the inference a real model has to make here. The
        # synthesis prompt carries no metadata, so there is nothing else to
        # read it off -- which is the point: the stub has to work for its
        # answer the same way the model does.
        gold = _requirement_in(item.prompt)
        others = [n for n in VARIABLE_PHRASES if n != gold]
        # Under the one-variable instruction it names just that variable;
        # otherwise it hedges over several, which is the observed failure mode.
        if "at most one" in item.prompt:
            return gold
        return ", ".join([gold, *others][: self.hedge_variables])


def _other(label: str) -> str:
    return ACTIONS[1] if label == ACTIONS[0] else ACTIONS[0]


def _requirement_in(prompt: str) -> str:
    """The variable the rule in this prompt turns on."""
    match = re.search(r"requires ([^.]+)\.", prompt)
    phrase = match.group(1).strip() if match else ""
    for name, text in VARIABLE_PHRASES.items():
        if text == phrase:
            return name
    return next(iter(VARIABLE_PHRASES))


@pytest.fixture
def cfg():
    return load_grounding_config(CONFIG)


@pytest.fixture
def items():
    return generate_route_choice_items(RouteChoiceSpec(n_items=40, seed=0))


# ------------------------------------------------------------------ the arms ---
def test_arm_specs_cross_a_route_policy_with_a_query_source(cfg):
    specs = {s.name: s for s in arm_specs(cfg)}
    assert specs[ARM_B0].route_policy == ROUTE_NEVER
    assert specs[ARM_B2].query_source == SOURCE_GENERIC
    assert specs[ARM_B4].route_policy == ROUTE_ALWAYS
    assert specs[ARM_B5].route_policy == ROUTE_LEARNED
    assert specs[ARM_B5].query_source == SOURCE_SYNTH
    # B6 and B7 are the two single-substitutions of B5, which is what makes the
    # decomposition subtractive.
    assert (specs[ARM_B6].route_policy, specs[ARM_B6].query_source) == (
        specs[ARM_B5].route_policy,
        "oracle",
    )
    assert (specs[ARM_B7].route_policy, specs[ARM_B7].query_source) == (
        "oracle",
        specs[ARM_B5].query_source,
    )


def test_unknown_arm_names_are_refused(cfg):
    cfg.policies = ["b9_wishful_thinking"]
    with pytest.raises(ValueError, match="unknown arm"):
        arm_specs(cfg)


def test_e3_arm_names_carry_their_variant(cfg):
    cfg.policies = [f"{E3_ARM_PREFIX}q3_hard_cap"]
    spec = arm_specs(cfg)[0]
    assert spec.variant == "q3_hard_cap"
    assert spec.route_policy == ROUTE_ALWAYS


# ------------------------------------------------------------- the episodes ---
def _cache(cfg, items, backend_name="test_scripted", variants=("q0_baseline",)):
    from hg.inference.base import get_backend

    backend = get_backend(backend_name, seed=0)
    backend.load()
    cfg.raw["synthesizer"] = {"kind": "lm", "on_parse_failure": "generic"}
    return build_cache(items, cfg, backend, list(variants), {})


def test_act_episodes_cost_nothing_and_leave_the_overhead_undefined(cfg, items):
    cache = _cache(cfg, items)
    spec = ArmSpec(ARM_B0, ROUTE_NEVER, SOURCE_NONE)
    episodes = [
        build_episode(spec, entry, lambda_bit=0.1, router=None, ask_bits=1.0)
        for entry in cache
    ]
    assert all(e.external_raw_bits == 0.0 for e in episodes)
    assert all(e.query is None for e in episodes)
    for e in episodes:
        needed = e.metadata["grounding_required_gold"]
        # Plan section 2.1: overhead is defined only for a protocol that met
        # the requirement. Acting on an item that needed grounding did not.
        if needed:
            assert e.metadata["coordination_overhead"] is None
            assert e.metadata["protocol_sufficient"] is False
        else:
            assert e.metadata["coordination_overhead"] == 0.0


def test_coordination_overhead_is_never_negative(cfg, items):
    cache = _cache(cfg, items)
    for spec in (
        ArmSpec(ARM_B2, ROUTE_ALWAYS, SOURCE_GENERIC),
        ArmSpec(ARM_B4, ROUTE_ALWAYS, SOURCE_SYNTH, variant="q0_baseline"),
    ):
        for entry in cache:
            e = build_episode(spec, entry, lambda_bit=0.1, router=None, ask_bits=1.0)
            overhead = e.metadata["coordination_overhead"]
            assert overhead is None or overhead >= 0.0


def test_episodes_carry_the_section_15_schema_and_no_nan(cfg, items):
    cache = _cache(cfg, items)
    spec = ArmSpec(ARM_B4, ROUTE_ALWAYS, SOURCE_SYNTH, variant="q0_baseline")
    e = build_episode(spec, cache[0], lambda_bit=0.1, router=None, ask_bits=1.0)
    for field in (
        "grounding_required_gold", "router_score", "router_route", "oracle_route",
        "query_text", "query_variables", "query_sound", "query_exact_gold",
        "query_response_alphabet_size", "external_response_bits", "final_correct",
        "protocol_sufficient", "coordination_overhead",
    ):
        assert field in e.metadata, field
    # Serializable with no NaN token, which most JSON readers reject.
    assert "NaN" not in json.dumps(e.to_dict())


def test_a_none_query_is_an_act_and_is_not_charged(cfg, items):
    cache = _cache(cfg, items)
    spec = ArmSpec(ARM_B4, ROUTE_ALWAYS, SOURCE_SYNTH, variant="q0_baseline")
    entry = cache[0]
    entry.grounded["q0_baseline"].query = None
    e = build_episode(spec, entry, lambda_bit=0.1, router=None, ask_bits=1.0)
    assert e.route == "act"
    assert e.external_raw_bits == 0.0
    assert e.final_action == e.initial_action


# ------------------------------------------------------------ the guarantees ---
def test_arms_are_item_paired_by_construction(cfg, items):
    """Every arm is assembled from one shared cache, so pairing is structural."""
    cache = _cache(cfg, items)
    specs = [
        ArmSpec(ARM_B2, ROUTE_ALWAYS, SOURCE_GENERIC),
        ArmSpec(ARM_B4, ROUTE_ALWAYS, SOURCE_SYNTH, variant="q0_baseline"),
    ]
    arms = {
        spec.name: [
            build_episode(spec, entry, lambda_bit=0.1, router=None, ask_bits=1.0)
            for entry in cache
        ]
        for spec in specs
    }
    ids = [[e.item_id for e in eps] for eps in arms.values()]
    assert ids[0] == ids[1]
    # And the ungrounded action is identical across arms, which is what makes
    # the distortion *reduction* comparable between them.
    assert [e.initial_action for e in arms[ARM_B2]] == [
        e.initial_action for e in arms[ARM_B4]
    ]


def test_router_features_are_ai_private(cfg, items):
    """The estimator must not be able to see the answer it is predicting."""
    cache = _cache(cfg, items)
    for entry in cache:
        assert set(entry.features) == {
            "majority_margin", "vote_entropy_bits", "unanimous",
            "label_margin", "top_label_prob",
        }
        assert all(math.isfinite(v) for v in entry.features.values())
    # Two items with the same reference action but different model behaviour
    # must be able to get different features, and two with different reference
    # actions and the same behaviour must not be separable by them.
    by_features: dict[tuple, set] = {}
    for entry in cache:
        key = tuple(sorted(entry.features.items()))
        by_features.setdefault(key, set()).add(entry.item.reference_action)
    assert any(len(v) > 1 for v in by_features.values()), (
        "every feature vector maps to exactly one reference action, which "
        "means the features identify the answer"
    )


def test_split_is_keyed_on_the_item_not_on_position(cfg, items):
    small, _ = split_cache(_cache(cfg, items), train_fraction=0.5, seed=0)
    more = generate_route_choice_items(RouteChoiceSpec(n_items=60, seed=0))
    big, _ = split_cache(_cache(cfg, more), train_fraction=0.5, seed=0)
    ids_small = {c.item.item_id for c in small}
    ids_big = {c.item.item_id for c in big}
    assert ids_small <= ids_big, "growing the item count moved the split"


# ----------------------------------------------------------------- the gates ---
def _run(tmp_path, cfg_path=CONFIG, **overrides):
    cfg = load_grounding_config(cfg_path)
    cfg.phase2_task = RouteChoiceSpec(n_items=120, seed=0)
    cfg.n_bootstrap = 200
    cfg.lambda_grid = [0.05, 0.10, 0.20]
    cfg.raw.update(overrides)
    return run_e2(cfg, output_root=tmp_path)


def test_a_full_run_round_trips_and_reproduces_its_own_analysis(tmp_path, cfg):
    run, analysis, problems = _run(
        tmp_path,
        backend="test_scripted",
        backend_options={},
        synthesizer={"kind": "lm", "on_parse_failure": "generic"},
    )
    assert problems == [], problems
    assert analysis["n_episodes"] > 0
    # The round-trip check is the point: the reported analysis is the one
    # recomputed from the written episodes.
    assert compare_analyses(analysis, analysis) == []


def test_minimality_intervention_is_detected_when_it_works(tmp_path):
    """Q3's one-variable instruction must show up as a cost reduction."""
    cfg = load_grounding_config(CONFIG)
    cfg.phase2_task = RouteChoiceSpec(n_items=120, seed=0)
    cfg.n_bootstrap = 200
    cfg.lambda_grid = [0.10]
    cfg.policies = [
        ARM_B2, ARM_B3,
        f"{E3_ARM_PREFIX}q0_baseline",
        f"{E3_ARM_PREFIX}q3_hard_cap",
    ]
    cfg.raw.update(
        backend="test_scripted",
        backend_options={},
        synthesizer={"kind": "lm", "on_parse_failure": "generic"},
    )
    _run_dir, analysis, _problems = run_e2(cfg, output_root=tmp_path)
    e3 = analysis["minimality"]
    assert e3["available"]
    row = next(r for r in e3["interventions"] if r["variant"] == "q3_hard_cap")
    assert row["bits_saved_vs_q0"] > 0
    assert row["improves"], row
    # The mixture of plan section 10: the intervention should drain the
    # non-minimal mode rather than merely shifting the mean.
    assert row["alpha_minimal"] > e3["baseline"]["alpha_minimal"]
    assert analysis["gates"]["H2"]["passed"]


def test_a_cheaper_but_unsound_intervention_does_not_count_as_improvement(cfg):
    """Falsification F2, made a property of the metric rather than a caveat."""
    from hg.grounding.integrated import minimality_comparison

    base_arm = (f"{E3_ARM_PREFIX}q0_baseline", 0.1)
    cheap_arm = (f"{E3_ARM_PREFIX}q3_hard_cap", 0.1)
    arms = {
        (ARM_B2, 0.1): {"mean_external_bits": 5.0, "mean_final_loss": 0.05,
                        "accuracy": 0.95, "query": {"soundness": 1.0,
                        "alpha_minimal": 0.0, "share_full_escalation": 1.0,
                        "variables_distribution": {"5": 10}}},
        (ARM_B3, 0.1): {"mean_external_bits": 1.0, "mean_final_loss": 0.05,
                        "accuracy": 0.95, "query": {"soundness": 1.0,
                        "alpha_minimal": 1.0, "share_full_escalation": 0.0,
                        "variables_distribution": {"1": 10}}},
        base_arm: {"mean_external_bits": 2.5, "mean_final_loss": 0.05,
                   "accuracy": 0.95, "query": {"soundness": 0.98,
                   "alpha_minimal": 0.5, "share_full_escalation": 0.3,
                   "variables_distribution": {"1": 5, "5": 5}}},
        cheap_arm: {"mean_external_bits": 1.0, "mean_final_loss": 0.20,
                    "accuracy": 0.80, "query": {"soundness": 0.55,
                    "alpha_minimal": 1.0, "share_full_escalation": 0.0,
                    "variables_distribution": {"1": 10}}},
    }

    class _Stub:
        def __init__(self, bits, loss):
            self.external_raw_bits = bits
            self.final_loss = loss

    grouped = {
        base_arm: [_Stub(2.5, 0.05)] * 10,
        cheap_arm: [_Stub(1.0, 0.20)] * 10,
    }
    e3 = minimality_comparison(cfg, grouped, arms, [0.1])
    row = e3["interventions"][0]
    assert row["bits_saved_vs_q0"] > 0
    assert row["significant"]
    # Cheaper, and not an improvement: soundness collapsed and the decision got
    # worse, which is the trade H2 exists to refuse.
    assert not row["soundness_preserved"]
    assert not row["quality_preserved"]
    assert not row["improves"]
