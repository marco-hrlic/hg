"""HG-v2 Phase 2: the controlled NL task, query synthesis, and gate G2.

Plan sections 12 to 17. The tests that matter most here are the ones that would
catch a *flattering* failure: a withheld variable leaking into the prompt, a
grounded baseline silently not receiving its answer, or a cost metric that
cannot tell a bloated query from a targeted one.
"""

from __future__ import annotations

import json

import pytest

from hg.config import ConfigError
from hg.grounding.channels import Route
from hg.grounding.config import load_grounding_config
from hg.grounding.phase2 import (
    BASELINES,
    G2_BASELINE,
    G2_CHALLENGER,
    MECHANISM_CHALLENGER,
    aggregate_baseline,
    build_analysis,
    compare_analyses,
    evaluate_gate,
    run_phase2,
    run_sweep,
)
from hg.grounding.route_choice import (
    ACTIONS,
    NO_CONSTRAINTS,
    PREFERENCE_VARIABLES,
    VARIABLE_PHRASES,
    RouteChoiceSpec,
    TravellerSource,
    constraints_text,
    excess_response_cost,
    generate_route_choice_items,
    generic_query,
    gold_query,
    resolves_decision,
    targeted_preference_query,
)
from hg.inference.base import GenerationSettings, get_backend
from hg.queries import GoldQuerySynthesizer, NoisyQuerySynthesizer


def items(n=200, **over):
    return generate_route_choice_items(RouteChoiceSpec(n_items=n, seed=0, **over))


# ------------------------------------------------------------------- the task ---
def test_the_rule_determines_the_reference_action():
    for item in items(500):
        if not item.a_faster:
            assert item.reference_action == "B"
        elif item.acceptances[item.requirement]:
            assert item.reference_action == "A"
        else:
            assert item.reference_action == "B"


def test_residual_deficit_is_one_bit_exactly_when_the_hidden_bit_decides():
    for item in items(300):
        assert item.needs_grounding == item.a_faster
        assert item.residual_deficit_bits == (1.0 if item.a_faster else 0.0)


def test_the_withheld_variable_never_appears_in_the_ungrounded_prompt():
    """The leak check. If the acceptance leaked into the prompt there would be
    no residual deficit and every Phase 2 number would be meaningless.

    Checked against the constraints block and against the exact verdict
    sentence, not against the whole prompt: the *rule* legitimately contains the
    words "is acceptable to the traveller", and asserting on a bare substring
    would either fail here or, if loosened, stop testing anything.
    """
    for item in items(300):
        prompt = item.prompt()
        assert NO_CONSTRAINTS in prompt

        # The verdict sentence `constraints_text` would render for this item's
        # decision-relevant variable, in both polarities.
        phrase = VARIABLE_PHRASES[item.requirement].capitalize()
        for verdict in (f"{phrase} is acceptable", f"{phrase} is not acceptable"):
            assert verdict not in prompt

        # The requirement itself is visible -- route A needs *something* -- and
        # the rule and both actions are present, so the task is well posed.
        assert VARIABLE_PHRASES[item.requirement] in prompt
        for action in ACTIONS:
            assert action in prompt
        assert "travel time" in prompt


def test_a_grounded_prompt_does_state_the_verdict():
    """The complement: the check above would pass trivially if the constraints
    block never said anything at all."""
    item = next(i for i in items(300) if i.needs_grounding)
    response = TravellerSource(item).respond(gold_query(item))
    prompt = item.prompt(constraints_text(response.content))
    phrase = VARIABLE_PHRASES[item.requirement].capitalize()
    expected = (
        f"{phrase} is acceptable"
        if item.acceptances[item.requirement]
        else f"{phrase} is not acceptable"
    )
    assert expected in prompt


def test_the_prompt_shows_the_travel_times_the_rule_needs():
    for item in items(50):
        prompt = item.prompt()
        assert f"{item.time_a} minutes" in prompt
        assert f"{item.time_b} minutes" in prompt


def test_a_faster_route_a_has_a_real_margin():
    """Not a rounding artifact a reader could dispute."""
    for item in items(300):
        assert abs(item.time_a - item.time_b) >= 5


def test_constraints_suffice_uses_explicit_variable_names():
    """Regression: this used to match phrases against rendered prose, which
    failed silently and in the flattering direction."""
    item = next(i for i in items(300) if i.needs_grounding)
    assert not item.constraints_suffice(())
    assert not item.constraints_suffice(("night_driving",)) or item.requirement == "night_driving"
    assert item.constraints_suffice((item.requirement,))
    # An item the prompt already settles needs nothing disclosed.
    easy = next(i for i in items(300) if not i.needs_grounding)
    assert easy.constraints_suffice(())


def test_the_grounded_item_is_marked_sufficient_only_with_the_right_variable():
    item = next(i for i in items(300) if i.needs_grounding)
    source = TravellerSource(item)

    gold = source.respond(gold_query(item))
    good = item.inference_item(constraints_text(gold.content), revealed=tuple(gold.content))
    assert good.meta["gs_sufficient"] is True

    wrong_var = next(v for v in item.variables if v != item.requirement)
    miss = source.respond(targeted_preference_query([wrong_var]))
    bad = item.inference_item(constraints_text(miss.content), revealed=tuple(miss.content))
    assert bad.meta["gs_sufficient"] is False


def test_a_single_preference_variable_is_rejected():
    """With one variable a generic ask and a targeted query cost the same, so
    gate G2 would be untestable by construction."""
    with pytest.raises(ValueError, match="at least two"):
        RouteChoiceSpec(variables=("mountain_roads",))


# ------------------------------------------------------------------ queries ----
def test_gold_query_costs_one_bit_and_resolves():
    for item in items(200):
        q = gold_query(item)
        assert q.response_bits == 1.0
        assert q.variable_names == (item.requirement,)
        assert resolves_decision(item, q)


def test_generic_query_costs_one_bit_per_variable():
    spec = RouteChoiceSpec(n_items=10, seed=0)
    q = generic_query(spec.variables)
    assert q.response_bits == float(len(spec.variables))
    assert q.response_alphabet_size == 2 ** len(spec.variables)
    assert spec.generic_bits == q.response_bits


def test_excess_response_cost_is_zero_for_the_gold_query_where_grounding_is_needed():
    for item in items(300):
        if item.needs_grounding:
            assert excess_response_cost(item, gold_query(item)) == 0.0
        else:
            # Nothing needed to be asked, so any ask is pure overhead.
            assert excess_response_cost(item, gold_query(item)) == 1.0


def test_a_bloated_query_resolves_but_overpays():
    """The case that separates cost metrics from decision metrics: a hedge is
    fully correct and fully wasteful, and an evaluation that looked only at
    decision quality would call it as good as gold."""
    item = next(i for i in items(200) if i.needs_grounding)
    decoys = [v for v in item.variables if v != item.requirement][:2]
    hedge = targeted_preference_query([item.requirement, *decoys])
    assert resolves_decision(item, hedge)
    assert hedge.response_bits == 3.0
    assert excess_response_cost(item, hedge) == 2.0


def test_a_wrong_query_is_cheap_and_useless():
    item = next(i for i in items(200) if i.needs_grounding)
    wrong = next(v for v in item.variables if v != item.requirement)
    q = targeted_preference_query([wrong])
    assert not resolves_decision(item, q)
    assert q.response_bits == 1.0


def test_query_rejects_repeats_and_unknown_variables():
    with pytest.raises(ValueError, match="repeats"):
        targeted_preference_query(["toll_roads", "toll_roads"])
    with pytest.raises(ValueError, match="unknown preference"):
        targeted_preference_query(["teleportation"])
    with pytest.raises(ValueError, match="at least one"):
        targeted_preference_query([])


def test_the_source_answers_truthfully_and_only_what_it_holds():
    item = items(10)[0]
    response = TravellerSource(item).respond(gold_query(item))
    assert response.content == {item.requirement: item.acceptances[item.requirement]}
    assert response.raw_bits == 1.0
    assert response.informative is True


def test_gold_synthesizer_always_produces_the_gold_query():
    syn = GoldQuerySynthesizer()
    for item in items(200):
        assert syn.synthesize(item).variable_names == (item.requirement,)


def test_noisy_synthesizer_hits_its_configured_rates():
    syn = NoisyQuerySynthesizer(p_correct=0.7, p_superset=0.2, n_extra=2, seed=0)
    pool = items(3000)
    gold = superset = miss = 0
    for item in pool:
        q = syn.synthesize(item)
        names = set(q.variable_names)
        if names == {item.requirement}:
            gold += 1
        elif item.requirement in names:
            superset += 1
        else:
            miss += 1
    n = len(pool)
    assert abs(gold / n - 0.7) < 0.04
    assert abs(superset / n - 0.2) < 0.04
    assert abs(miss / n - 0.1) < 0.04


def test_noisy_synthesizer_rejects_impossible_rates():
    with pytest.raises(ValueError, match="> 1"):
        NoisyQuerySynthesizer(p_correct=0.8, p_superset=0.5)


def test_noisy_synthesizer_is_deterministic():
    a = NoisyQuerySynthesizer(seed=1)
    b = NoisyQuerySynthesizer(seed=1)
    for item in items(100):
        assert a.synthesize(item).variable_names == b.synthesize(item).variable_names


# ---------------------------------------------------------------- the backend ---
def test_the_simulator_is_right_when_the_prompt_suffices_and_guesses_otherwise():
    backend = get_backend("grounding_simulator", seed=0, bias_strength=0.8)
    pool = items(2000)
    preds = backend.predict(
        [i.inference_item() for i in pool], GenerationSettings()
    )
    easy = [(p, i) for p, i in zip(preds, pool) if not i.needs_grounding]
    hard = [(p, i) for p, i in zip(preds, pool) if i.needs_grounding]

    assert all(p.label == i.reference_action for p, i in easy)
    hard_acc = sum(p.label == i.reference_action for p, i in hard) / len(hard)
    assert 0.4 < hard_acc < 0.6  # a guess, whatever the bias


def test_the_simulator_guesses_with_the_configured_bias():
    """A bias rather than a coin flip, because self-consistency cannot fix a
    bias and that is what baseline B1 is for."""
    backend = get_backend("grounding_simulator", seed=0, bias_strength=0.9)
    pool = [i for i in items(2000) if i.needs_grounding]
    preds = backend.predict(
        [i.inference_item() for i in pool], GenerationSettings()
    )
    # The hint on a grounding-needed item is always A (A is faster).
    share_a = sum(p.label == "A" for p in preds) / len(preds)
    assert abs(share_a - 0.9) < 0.05


def test_greedy_decoding_is_deterministic_and_sampling_varies():
    backend = get_backend("grounding_simulator", seed=0)
    pool = [i.inference_item() for i in items(400)]
    greedy = GenerationSettings(temperature=0.0)
    a = [p.label for p in backend.predict(pool, greedy)]
    b = [p.label for p in backend.predict(pool, greedy)]
    assert a == b

    s1 = [p.label for p in backend.predict(pool, GenerationSettings(temperature=0.7, seed=1))]
    s2 = [p.label for p in backend.predict(pool, GenerationSettings(temperature=0.7, seed=2))]
    assert s1 != s2  # different seeds really are different draws


def test_the_simulator_respects_a_configured_error_rate():
    backend = get_backend("grounding_simulator", seed=0, error_rate_when_sufficient=0.2)
    pool = [i for i in items(2000) if not i.needs_grounding]
    preds = backend.predict(
        [i.inference_item() for i in pool], GenerationSettings()
    )
    acc = sum(p.label == i.reference_action for p, i in zip(preds, pool)) / len(pool)
    assert abs(acc - 0.8) < 0.05


def test_the_simulator_rejects_impossible_settings():
    with pytest.raises(ValueError, match="error_rate_when_sufficient"):
        get_backend("grounding_simulator", error_rate_when_sufficient=1.5)
    with pytest.raises(ValueError, match="bias_strength"):
        get_backend("grounding_simulator", bias_strength=-0.1)


# ------------------------------------------------------------------ baselines ---
def phase2_config(tmp_path, **over):
    import yaml

    body = {
        "name": "test_phase2",
        "task": {
            "kind": "route_choice",
            "n_items": 1500,
            "p_a_faster": 0.5,
            "p_accepts": 0.5,
            "variables": list(PREFERENCE_VARIABLES),
        },
        "policies": list(BASELINES),
        "backend": "grounding_simulator",
        "backend_options": {"bias_strength": 0.8},
        "self_consistency_samples": 5,
        "temperature": 0.7,
        "synthesizer": {"kind": "noisy", "p_correct": 0.7, "p_superset": 0.2},
        "seed": 0,
        "n_bootstrap": 100,
        "output_root": str(tmp_path / "results"),
    }
    body.update(over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_grounding_config(p)


def arms_of(analysis):
    return {a["baseline"]: a for a in analysis["arms"]}


def test_the_ask_baselines_pay_the_costs_the_plan_says(tmp_path):
    cfg = phase2_config(tmp_path)
    arms = arms_of(build_analysis(cfg, run_sweep(cfg)[0]))

    assert arms["b0_no_grounding"]["mean_external_bits"] == 0.0
    assert arms["b1_self_consistency"]["mean_external_bits"] == 0.0
    assert arms["b2_generic_ask"]["mean_external_bits"] == float(len(PREFERENCE_VARIABLES))
    assert arms["b3_oracle_targeted"]["mean_external_bits"] == 1.0
    # The noisy synthesizer hedges sometimes, so it costs more than gold.
    assert arms["b4_synthesized_query"]["mean_external_bits"] > 1.0


def test_grounding_actually_resolves_the_decision(tmp_path):
    """The regression that the phrase-matching bug caused: every grounded arm
    came out at exactly the ungrounded accuracy."""
    cfg = phase2_config(tmp_path)
    arms = arms_of(build_analysis(cfg, run_sweep(cfg)[0]))

    assert arms["b2_generic_ask"]["accuracy"] == 1.0
    assert arms["b3_oracle_targeted"]["accuracy"] == 1.0
    assert arms["b0_no_grounding"]["accuracy"] < 0.85
    assert arms["b3_oracle_targeted"]["accuracy"] > arms["b0_no_grounding"]["accuracy"]


def test_self_consistency_does_not_substitute_for_grounding(tmp_path):
    """Plan's premise that internal_reducibility = 0: no amount of internal
    sampling recovers a variable that is absent from the prompt."""
    cfg = phase2_config(tmp_path)
    analysis = build_analysis(cfg, run_sweep(cfg)[0])
    check = analysis["internal_compute_check"]

    assert check["available"]
    assert check["b1_samples"] == 5
    assert check["external_bits_spent"] == 0.0
    # It may wobble either way by a little; it must not close the grounding gap.
    assert abs(check["accuracy_gain"]) < 0.05
    arms = arms_of(analysis)
    assert arms["b1_self_consistency"]["accuracy"] < arms["b3_oracle_targeted"]["accuracy"]


def test_resolution_accuracy_is_not_the_same_as_decision_accuracy(tmp_path):
    """Resolution is about the query; accuracy also depends on the model
    applying the rule once it has the answer."""
    cfg = phase2_config(tmp_path, backend_options={"error_rate_when_sufficient": 0.2})
    arms = arms_of(build_analysis(cfg, run_sweep(cfg)[0]))
    gold = arms["b3_oracle_targeted"]
    assert gold["resolution_accuracy"] == 1.0
    assert gold["accuracy"] < 0.9  # the model, not the query, is the problem


def test_excess_cost_is_reported_both_overall_and_where_grounding_is_needed(tmp_path):
    """Otherwise the cost of not routing hides inside the query result."""
    cfg = phase2_config(tmp_path)
    arms = arms_of(build_analysis(cfg, run_sweep(cfg)[0]))
    gold = arms["b3_oracle_targeted"]
    # Zero on the items that actually needed a bit...
    assert gold["mean_excess_where_grounding_needed"] == pytest.approx(0.0)
    # ...but positive overall, because it also asked where nothing was needed.
    assert gold["mean_excess_response_cost"] > 0.0


def test_aggregate_refuses_to_mix_baselines(tmp_path):
    cfg = phase2_config(tmp_path)
    with pytest.raises(ValueError, match="mixes baselines"):
        aggregate_baseline(run_sweep(cfg)[0])


# ---------------------------------------------------------------- the G2 gate ---
def test_the_mechanism_check_detects_oracle_compression(tmp_path):
    cfg = phase2_config(tmp_path)
    mech = build_analysis(cfg, run_sweep(cfg)[0])["mechanism_check"]

    assert mech["available"]
    assert mech["challenger"] == MECHANISM_CHALLENGER
    assert mech["passed"]
    assert mech["equal_decision_quality"]
    assert mech["bits_saved"] == pytest.approx(len(PREFERENCE_VARIABLES) - 1.0)
    assert mech["compression_ratio"] == pytest.approx(len(PREFERENCE_VARIABLES))


def test_gate_g2_is_not_evaluated_on_a_simulated_backend(tmp_path):
    """G2 is a claim about a frozen off-the-shelf model. A simulator cannot bear
    it either way, so it must not open or close the gate."""
    cfg = phase2_config(tmp_path)
    gate = build_analysis(cfg, run_sweep(cfg)[0])["gate"]
    assert gate["backend_is_simulated"] is True
    assert gate["evaluated"] is False
    assert "NOT satisfied" in gate["verdict"] or "stand-in" in gate["verdict"]


def test_gate_g2_is_evaluated_once_the_backend_is_not_the_simulator(tmp_path):
    cfg = phase2_config(tmp_path)
    episodes = run_sweep(cfg)[0]
    grouped: dict[str, list] = {}
    for e in episodes:
        grouped.setdefault(e.policy, []).append(e)
    arms = {name: aggregate_baseline(eps) for name, eps in grouped.items()}

    cfg.raw["backend"] = "transformers"
    gate = evaluate_gate(cfg, grouped, arms)
    assert gate["evaluated"] is True
    assert gate["backend_is_simulated"] is False


def test_gate_g2_passes_with_a_gold_synthesizer(tmp_path):
    """A perfect synthesizer is the case the gate is supposed to open on."""
    cfg = phase2_config(tmp_path, synthesizer={"kind": "gold"})
    episodes = run_sweep(cfg)[0]
    grouped: dict[str, list] = {}
    for e in episodes:
        grouped.setdefault(e.policy, []).append(e)
    arms = {name: aggregate_baseline(eps) for name, eps in grouped.items()}

    cfg.raw["backend"] = "transformers"  # pretend it was real
    gate = evaluate_gate(cfg, grouped, arms)
    assert gate["passed"], gate["failures"]
    assert gate["equal_decision_quality"]
    assert gate["compression_ratio"] == pytest.approx(len(PREFERENCE_VARIABLES))


def test_gate_g2_refuses_a_cheap_query_that_decides_worse(tmp_path):
    """The conditional structure of the claim, which is the whole point: a
    synthesizer that mostly misses is cheap and must not pass."""
    cfg = phase2_config(
        tmp_path, synthesizer={"kind": "noisy", "p_correct": 0.2, "p_superset": 0.0}
    )
    episodes = run_sweep(cfg)[0]
    grouped: dict[str, list] = {}
    for e in episodes:
        grouped.setdefault(e.policy, []).append(e)
    arms = {name: aggregate_baseline(eps) for name, eps in grouped.items()}

    cfg.raw["backend"] = "transformers"
    gate = evaluate_gate(cfg, grouped, arms)
    assert not gate["passed"]
    assert not gate["equal_decision_quality"]
    assert gate["bits_saved"] > 0  # it was cheaper, and that is not enough
    assert any("deciding worse" in f for f in gate["failures"])


def test_an_unknown_synthesizer_kind_is_rejected(tmp_path):
    cfg = phase2_config(tmp_path, synthesizer={"kind": "telepathic"})
    with pytest.raises(ValueError, match="unknown synthesizer kind"):
        run_sweep(cfg)


# --------------------------------------------------------- artifacts / config ---
def test_phase2_analysis_from_saved_artifacts_matches_the_live_run(tmp_path):
    cfg = phase2_config(tmp_path, n_bootstrap=50)
    run, analysis, problems = run_phase2(cfg)
    assert problems == []

    from hg.grounding.phase2 import analyze_run_dir

    again = analyze_run_dir(run.root)
    assert compare_analyses(analysis, again) == []
    assert (run.root / "diagnostics.json").exists()
    assert run.report_path.exists()


def test_phase2_artifacts_are_strict_json(tmp_path):
    cfg = phase2_config(tmp_path, n_bootstrap=20)
    run, _, _ = run_phase2(cfg)
    paths = [run.analysis_path, run.manifest_path, run.root / "diagnostics.json"]
    paths += sorted((run.root / "episodes").glob("*.jsonl"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in ("NaN", "Infinity"):
            assert token not in text, f"{path.name} contains a bare {token}"
        if path.suffix == ".json":
            json.loads(text)


def test_episodes_record_the_route_and_the_query(tmp_path):
    cfg = phase2_config(tmp_path)
    episodes = run_sweep(cfg)[0]
    by_policy: dict[str, list] = {}
    for e in episodes:
        by_policy.setdefault(e.policy, []).append(e)

    assert all(e.route == Route.ACT.value for e in by_policy["b0_no_grounding"])
    assert all(e.query is None for e in by_policy["b0_no_grounding"])
    assert all(e.route == Route.ASK_GENERIC.value for e in by_policy["b2_generic_ask"])
    for e in by_policy["b3_oracle_targeted"]:
        assert e.route == Route.ASK_TARGETED.value
        assert e.query is not None
        assert e.metadata["query_variables"] == e.metadata["gold_variables"]


def test_the_diagnostics_flag_a_simulated_run(tmp_path):
    cfg = phase2_config(tmp_path)
    diagnostics = run_sweep(cfg)[1]
    assert diagnostics["is_simulated"] is True
    assert diagnostics["generic_ask_bits"] == float(len(PREFERENCE_VARIABLES))
    assert "example_prompt" in diagnostics


def test_the_shipped_phase2_config_loads(repo_root):
    cfg = load_grounding_config(
        repo_root / "configs" / "experiments" / "hg_v2" / "phase2_frozen_lm.yaml"
    )
    assert cfg.task_kind == "route_choice"
    assert cfg.phase2_task is not None
    assert set(cfg.policies) == set(BASELINES)
    assert cfg.gate.distortion_tolerance > 0
    # The backend must be replaceable from config (plan section 12).
    assert cfg.raw["backend"] == "grounding_simulator"


def test_phase1_policies_are_rejected_for_the_phase2_task(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("task:\n  kind: route_choice\npolicies: [always_act]\n")
    with pytest.raises(ConfigError, match="unknown policies"):
        load_grounding_config(p)


def test_a_phase2_gate_key_is_rejected_for_a_phase1_config(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "task:\n  kind: noisy_index\nlambda_grid: [0.1, 0.2]\n"
        "gate:\n  distortion_tolerance: 0.1\n"
    )
    with pytest.raises(ConfigError, match="unknown gate keys"):
        load_grounding_config(p)


def test_g2_pair_names_are_the_plan_s(tmp_path):
    assert G2_CHALLENGER == "b4_synthesized_query"
    assert G2_BASELINE == "b2_generic_ask"
