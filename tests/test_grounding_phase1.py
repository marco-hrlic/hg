"""HG-v2 Phase 1: the routing task, the router, and gate G1.

Plan sections 9 to 11 and 24 ("router oracle labels -- verify route-cost
calculations manually on tiny fixtures"). The route arithmetic is checked on
hand-computed fixtures rather than only end to end, because a sign error in
`C_r = L_r + lambda*B_r` would still produce a plausible-looking sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from hg.config import ConfigError
from hg.grounding.channels import Route
from hg.grounding.config import load_grounding_config
from hg.grounding.costs import (
    aggregate_route_arm,
    episode_costs,
    paired_cost_advantage,
    total_route_cost,
)
from hg.grounding.noisy_index import (
    ASK_BITS,
    NoisyIndexTaskSpec,
    bayes_act_error_probability,
    build_episode,
    generate_noisy_index_items,
    noise_mixture,
    split_items,
    true_act_error_probability,
)
from hg.grounding.phase1 import (
    POLICIES,
    build_analysis,
    check_validity,
    compare_analyses,
    decision_state_for,
    fit_router,
    route_for_policy,
    run_phase1,
    run_sweep,
)
from hg.routing.decision_state import FEATURE_NAMES, agreement_features
from hg.routing.grounding_router import (
    LogisticRouter,
    oracle_route,
    route_from_expected_cost,
)


# --------------------------------------------------- route-cost arithmetic ----
def test_total_route_cost_on_hand_computed_values():
    # Acting and being wrong costs 1 and communicates nothing.
    assert total_route_cost(1.0, 0.0, 0.2) == 1.0
    # Asking resolves the decision but pays lambda per bit.
    assert total_route_cost(0.0, 1.0, 0.2) == pytest.approx(0.2)
    assert total_route_cost(0.0, 8.0, 0.25) == pytest.approx(2.0)
    # lambda = 0 makes external information free, so asking is never worse.
    assert total_route_cost(0.0, 64.0, 0.0) == 0.0


def test_expected_cost_route_rule():
    """ACT iff E[L_act] <= lambda*B_ask."""
    assert route_from_expected_cost(0.05, lambda_bit=0.10) is Route.ACT
    assert route_from_expected_cost(0.30, lambda_bit=0.10) is Route.ASK_TARGETED
    # Exact indifference goes to ACT: spending the source's attention for no
    # expected gain is the worse tie-break.
    assert route_from_expected_cost(0.10, lambda_bit=0.10) is Route.ACT


def test_oracle_route_uses_realized_loss():
    # Acting would have been right, so asking can only add cost.
    assert oracle_route(0.0, lambda_bit=0.5) is Route.ACT
    # Acting would have been wrong: ask, as long as a bit costs less than
    # being wrong.
    assert oracle_route(1.0, lambda_bit=0.5) is Route.ASK_TARGETED
    # ...but if a bit costs more than the mistake, eat the mistake.
    assert oracle_route(1.0, lambda_bit=1.5) is Route.ACT
    assert oracle_route(1.0, lambda_bit=1.0) is Route.ACT  # tie -> ACT


def test_route_accuracy_and_regret_on_a_tiny_fixture():
    """Two items, one lambda, every number checkable by hand."""
    spec = NoisyIndexTaskSpec(n_bits=4, n_items=2, n_observations=1, p_self_sufficient=1.0)
    items = generate_noisy_index_items(spec)
    # p_self_sufficient = 1.0 means the single observation is never flipped, so
    # acting is always right and the oracle always acts.
    assert all(i.loss_act == 0.0 for i in items)

    lam = 0.25
    always_ask = [
        build_episode(
            i, Route.ASK_TARGETED, policy="always_ask",
            decision_state={}, lambda_bit=lam,
        )
        for i in items
    ]
    arm = aggregate_route_arm(always_ask)
    # It asked on both items where the oracle would have acted.
    assert arm.route_accuracy == 0.0
    assert arm.mean_final_loss == 0.0
    assert arm.mean_external_bits == ASK_BITS
    assert arm.mean_total_cost == pytest.approx(lam * ASK_BITS)
    assert arm.mean_oracle_cost == 0.0
    assert arm.routing_regret == pytest.approx(lam * ASK_BITS)


def test_regret_is_zero_for_the_oracle():
    cfg_items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=500))
    lam = 0.2
    episodes = []
    for item in cfg_items:
        route, p, exact = route_for_policy(
            "oracle", item, lambda_bit=lam, router=None
        )
        episodes.append(
            build_episode(
                item, route, policy="oracle",
                decision_state=decision_state_for(item, p_wrong=p, exact=exact),
                lambda_bit=lam,
            )
        )
    arm = aggregate_route_arm(episodes)
    assert arm.routing_regret == 0.0
    assert arm.route_accuracy == 1.0


def test_aggregate_refuses_to_mix_lambdas():
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=4))
    episodes = [
        build_episode(i, Route.ACT, policy="always_act", decision_state={}, lambda_bit=lam)
        for i, lam in zip(items, (0.1, 0.1, 0.2, 0.2))
    ]
    with pytest.raises(ValueError, match="mixes lambda"):
        aggregate_route_arm(episodes)


def test_paired_comparison_requires_item_alignment():
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=6))
    a = [build_episode(i, Route.ACT, policy="a", decision_state={}, lambda_bit=0.1) for i in items]
    b = [build_episode(i, Route.ACT, policy="b", decision_state={}, lambda_bit=0.1) for i in reversed(items)]
    with pytest.raises(ValueError, match="not item-aligned"):
        paired_cost_advantage(a, b, n_boot=10)


# --------------------------------------------------------------- the task ----
def test_self_sufficient_items_need_no_grounding():
    """Plan section 9: with probability p, J is directly observable from M_A."""
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=4000))
    self_sufficient = [i for i in items if i.self_sufficient]
    assert self_sufficient  # the construction produced some
    # Noiseless observations, so acting on them is never wrong.
    assert all(i.loss_act == 0.0 for i in self_sufficient)
    assert all(i.error_rate == 0.0 for i in self_sufficient)


def test_asking_always_resolves_the_decision():
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=200))
    for item in items:
        e = build_episode(
            item, Route.ASK_TARGETED, policy="p", decision_state={}, lambda_bit=0.1
        )
        assert e.final_loss == 0.0
        assert e.external_raw_bits == ASK_BITS


def test_acting_costs_no_external_bits():
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=100))
    for item in items:
        e = build_episode(item, Route.ACT, policy="p", decision_state={}, lambda_bit=0.1)
        assert e.external_raw_bits == 0.0
        assert e.query is None
        assert e.final_loss == e.initial_loss == item.loss_act


def test_counterfactuals_are_stored_on_every_episode():
    """Plan section 9 asks for both routes' outcomes, which is what makes the
    oracle route -- and therefore regret -- computable from the artifact."""
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=50))
    for item in items:
        e = build_episode(item, Route.ACT, policy="p", decision_state={}, lambda_bit=0.3)
        assert e.metadata["loss_act"] == item.loss_act
        assert e.metadata["loss_ask"] == 0.0
        assert e.metadata["bits_act"] == 0.0
        assert e.metadata["bits_ask"] == ASK_BITS


def test_even_observation_panels_are_rejected():
    """An even panel can tie, and a tie-break rule would be a second,
    undocumented policy inside the task."""
    with pytest.raises(ValueError, match="odd"):
        NoisyIndexTaskSpec(n_observations=4)


def test_error_rates_above_a_half_are_rejected():
    with pytest.raises(ValueError, match=r"\[0, 0.5\]"):
        NoisyIndexTaskSpec(error_rates=(0.7,))


def test_split_is_disjoint_and_deterministic():
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=500))
    a_train, a_eval = split_items(items, seed=0)
    b_train, b_eval = split_items(items, seed=0)
    assert [i.item_id for i in a_train] == [i.item_id for i in b_train]
    assert not ({i.item_id for i in a_train} & {i.item_id for i in a_eval})
    assert len(a_train) + len(a_eval) == len(items)


# ------------------------------------------------------------ probabilities ---
def test_noise_mixture_is_a_distribution():
    spec = NoisyIndexTaskSpec()
    assert sum(w for w, _ in noise_mixture(spec)) == pytest.approx(1.0)


def test_bayes_error_probability_is_calibrated():
    """The exact posterior must match the realized error rate per feature level.

    This is the check that makes `bayes_optimal` usable as a ceiling: if it were
    miscalibrated, the learned router could beat it for uninteresting reasons.
    """
    spec = NoisyIndexTaskSpec(n_items=20000)
    items = generate_noisy_index_items(spec)
    buckets: dict[float, list] = {}
    for item in items:
        p = bayes_act_error_probability(item.observations, spec)
        buckets.setdefault(round(p, 6), []).append(item)

    for p, group in buckets.items():
        realized = sum(i.loss_act for i in group) / len(group)
        # Binomial standard error on the group, times four.
        se = (max(p, 1e-6) * (1 - p) / len(group)) ** 0.5
        assert abs(realized - p) < max(4 * se, 0.01), (p, realized, len(group))


def test_unanimous_evidence_is_safer_than_a_split_vote():
    spec = NoisyIndexTaskSpec()
    unanimous = bayes_act_error_probability([1, 1, 1, 1, 1], spec)
    split = bayes_act_error_probability([1, 1, 1, 0, 0], spec)
    assert unanimous < split


def test_true_error_probability_conditions_on_the_noise_level():
    """Documented as *not* a ceiling: it conditions on e_i, which is neither
    more nor less informative than the observed votes."""
    spec = NoisyIndexTaskSpec(n_items=200)
    items = generate_noisy_index_items(spec)
    clean = [i for i in items if i.self_sufficient]
    assert all(true_act_error_probability(i) == 0.0 for i in clean)
    noisy = [i for i in items if not i.self_sufficient]
    assert all(0.0 < true_act_error_probability(i) < 0.5 for i in noisy)


# ---------------------------------------------------------------- features ----
def test_agreement_features_see_only_the_observations():
    """A router that could see the reference decision would report its own leak
    as routing skill, so the feature function takes nothing else."""
    assert agreement_features([1, 1, 1]) == {
        "majority_margin": 1.0,
        "vote_entropy_bits": 0.0,
        "unanimous": 1.0,
    }
    split = agreement_features([1, 1, 0])
    assert split["majority_margin"] == pytest.approx(1 / 3)
    assert split["vote_entropy_bits"] == pytest.approx(0.9182958340544896)
    assert split["unanimous"] == 0.0
    # Symmetric in the labels: 2-1 for zero looks exactly like 2-1 for one.
    assert agreement_features([0, 0, 1]) == split


def test_features_are_label_symmetric_and_ordered():
    spec = NoisyIndexTaskSpec()
    for obs in ([1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]):
        f = agreement_features(obs)
        assert set(f) == set(FEATURE_NAMES)
    margins = [
        agreement_features(o)["majority_margin"]
        for o in ([1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1])
    ]
    assert margins == sorted(margins)
    del spec


def test_empty_observations_are_rejected():
    with pytest.raises(ValueError, match="at least one observation"):
        agreement_features([])


# ------------------------------------------------------------------ router ----
def test_router_recovers_the_error_ordering():
    spec = NoisyIndexTaskSpec(n_items=6000)
    items = generate_noisy_index_items(spec)
    train, evaluate = split_items(items)
    router = LogisticRouter().fit(
        [agreement_features(i.observations) for i in train],
        [i.loss_act for i in train],
    )
    p_unanimous = float(router.predict_proba([agreement_features([1, 1, 1, 1, 1])])[0])
    p_split = float(router.predict_proba([agreement_features([1, 1, 1, 0, 0])])[0])
    assert p_unanimous < p_split
    # ...and it lands near the exact posterior it is estimating.
    assert abs(p_unanimous - bayes_act_error_probability([1, 1, 1, 1, 1], spec)) < 0.05
    assert abs(p_split - bayes_act_error_probability([1, 1, 1, 0, 0], spec)) < 0.05
    del evaluate


def test_router_is_deterministic():
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=500))
    f = [agreement_features(i.observations) for i in items]
    y = [i.loss_act for i in items]
    assert LogisticRouter().fit(f, y).to_dict() == LogisticRouter().fit(f, y).to_dict()


def test_router_refuses_to_predict_before_fitting():
    with pytest.raises(RuntimeError, match="not been fitted"):
        LogisticRouter().predict_proba([agreement_features([1, 1, 1])])


def test_router_refuses_an_empty_training_set():
    with pytest.raises(ValueError, match="empty training set"):
        LogisticRouter().fit([], [])


def test_router_survives_a_constant_feature():
    """A zero-variance feature must not become a division by zero that poisons
    every weight."""
    f = [agreement_features([1, 1, 1]) for _ in range(50)]
    router = LogisticRouter(n_steps=50).fit(f, [0.0] * 50)
    p = router.predict_proba(f)
    assert np.all(np.isfinite(p))


def test_sigmoid_does_not_overflow():
    from hg.routing.grounding_router import _sigmoid

    z = np.array([-1e4, -50.0, 0.0, 50.0, 1e4])
    p = _sigmoid(z)
    assert np.all(np.isfinite(p))
    assert p[0] == pytest.approx(0.0)
    assert p[2] == pytest.approx(0.5)
    assert p[-1] == pytest.approx(1.0)


# ------------------------------------------------------------- the G1 gate ----
def phase1_config(tmp_path, **over):
    import yaml

    body = {
        "name": "test_phase1",
        "task": {
            "kind": "noisy_index",
            "n_bits": 8,
            "n_items": 4000,
            "n_observations": 5,
            "p_self_sufficient": 0.4,
            "error_rates": [0.25, 0.45],
        },
        "policies": list(POLICIES),
        "lambda_grid": [0.05, 0.1, 0.2, 0.3],
        "train_fraction": 0.5,
        "seed": 0,
        "n_bootstrap": 100,
        "output_root": str(tmp_path / "results"),
    }
    body.update(over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_grounding_config(p)


def test_gate_g1_passes_and_the_router_beats_both_baselines(tmp_path):
    cfg = phase1_config(tmp_path)
    episodes, router, diagnostics = run_sweep(cfg)
    analysis = build_analysis(cfg, episodes)

    assert analysis["gate"]["passed"], analysis["gate"]["failures"]
    assert len(analysis["gate"]["winning_lambdas"]) >= 2
    assert diagnostics["n_train"] + diagnostics["n_eval"] == 4000
    assert router.weights


def test_the_learned_router_never_beats_its_own_ceiling(tmp_path):
    """bayes_optimal sees the same features plus exact knowledge of the noise
    distribution. Beating it would mean the router saw something it should not."""
    cfg = phase1_config(tmp_path)
    analysis = build_analysis(cfg, run_sweep(cfg)[0])
    assert analysis["validity"]["learned_beats_its_ceiling_at"] == []


def test_gate_g1_requires_a_range_not_one_lucky_price(tmp_path):
    cfg = phase1_config(tmp_path, lambda_grid=[0.1])
    analysis = build_analysis(cfg, run_sweep(cfg)[0])
    # Wins at its single price, but one price is not a range.
    assert analysis["comparisons"][0]["beats_both"]
    assert not analysis["gate"]["passed"]
    assert "nontrivial range" in analysis["gate"]["failures"][0]


def test_gate_g1_fails_when_the_signal_is_destroyed(tmp_path):
    """With no self-sufficient items and uniformly maximal noise, internal
    agreement carries no routing information and the gate must not open."""
    cfg = phase1_config(
        tmp_path,
        task={
            "kind": "noisy_index",
            "n_bits": 8,
            "n_items": 4000,
            "n_observations": 5,
            "p_self_sufficient": 0.0,
            "error_rates": [0.5],
        },
    )
    analysis = build_analysis(cfg, run_sweep(cfg)[0])
    assert not analysis["gate"]["passed"]


def test_always_ask_pays_a_bit_on_every_item(tmp_path):
    cfg = phase1_config(tmp_path)
    analysis = build_analysis(cfg, run_sweep(cfg)[0])
    for arm in analysis["arms"]:
        if arm["policy"] == "always_ask":
            assert arm["external_query_rate"] == 1.0
            assert arm["mean_external_bits"] == ASK_BITS
            assert arm["mean_final_loss"] == 0.0
        if arm["policy"] == "always_act":
            assert arm["external_query_rate"] == 0.0
            assert arm["mean_external_bits"] == 0.0


def test_the_router_asks_less_as_external_bits_get_dearer(tmp_path):
    """Monotonicity is the sanity check on the whole cost model: a higher price
    for external information must never increase the query rate."""
    cfg = phase1_config(tmp_path)
    analysis = build_analysis(cfg, run_sweep(cfg)[0])
    rates = [
        a["external_query_rate"] for a in analysis["arms"] if a["policy"] == "learned"
    ]
    assert rates == sorted(rates, reverse=True)


def test_decision_state_marks_estimates_as_inexact():
    """Plan sections 4.1 and 18: an estimator output must not be recorded as an
    exact quantity."""
    items = generate_noisy_index_items(NoisyIndexTaskSpec(n_items=10))
    train = items
    import types

    cfg = types.SimpleNamespace(raw={})
    router = fit_router(train, cfg)  # type: ignore[arg-type]
    _, p, exact = route_for_policy(
        "learned", items[0], lambda_bit=0.1, router=router
    )
    assert exact is False
    state = decision_state_for(items[0], p_wrong=p, exact=exact)
    assert state["exact"] is False
    assert state["expected_loss_now"] == p


def test_validity_check_flags_a_router_that_beats_its_ceiling():
    """The check has to be able to fire, or it is decoration."""
    from hg.grounding.costs import RouteMetrics

    def arm(policy: str, cost: float) -> RouteMetrics:
        return RouteMetrics(
            policy=policy, task="noisy_index", n_items=10, lambda_bit=0.1,
            route_accuracy=1.0, mean_final_loss=0.0, mean_loss_before=0.0,
            delta_distortion=0.0, external_query_rate=0.0,
            mean_external_bits=0.0, mean_total_cost=cost,
            mean_oracle_cost=0.0, routing_regret=0.0,
            information_efficiency=None,
        )

    flagged = check_validity(
        {("learned", 0.1): arm("learned", 0.01),
         ("bayes_optimal", 0.1): arm("bayes_optimal", 0.05)}
    )
    assert flagged["learned_beats_its_ceiling_at"]
    assert "leak" in flagged["note"]


# ------------------------------------------------------------------ config ----
def test_the_shipped_phase1_config_loads(repo_root):
    cfg = load_grounding_config(
        repo_root / "configs" / "experiments" / "hg_v2" / "phase1_router.yaml"
    )
    assert cfg.task_kind == "noisy_index"
    assert cfg.phase1_task is not None
    assert len(cfg.lambda_grid) >= 2
    assert cfg.gate.min_winning_lambdas >= 2


def test_a_phase1_config_needs_a_lambda_grid(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("task:\n  kind: noisy_index\n")
    with pytest.raises(ConfigError, match="lambda_grid"):
        load_grounding_config(p)


def test_phase0_policies_are_rejected_for_the_phase1_task(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "task:\n  kind: noisy_index\nlambda_grid: [0.1, 0.2]\n"
        "policies: [ask_generic]\n"
    )
    with pytest.raises(ConfigError, match="unknown policies"):
        load_grounding_config(p)


def test_phase1_gate_keys_are_rejected_for_a_phase0_config(tmp_path):
    """The two phases have different gates; a key from the wrong one must not
    silently leave a threshold at its default."""
    p = tmp_path / "bad.yaml"
    p.write_text("task:\n  kind: index\n  n_bits: [4]\ngate:\n  min_winning_lambdas: 3\n")
    with pytest.raises(ConfigError, match="unknown gate keys"):
        load_grounding_config(p)


def test_negative_lambda_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("task:\n  kind: noisy_index\nlambda_grid: [-0.1, 0.2]\n")
    with pytest.raises(ConfigError, match="non-negative"):
        load_grounding_config(p)


# ------------------------------------------------- artifacts and round-trip ---
def test_phase1_analysis_from_saved_artifacts_matches_the_live_run(tmp_path):
    cfg = phase1_config(tmp_path, n_bootstrap=50)
    run, analysis, problems = run_phase1(cfg)

    assert problems == []
    assert analysis["gate"]["passed"], analysis["gate"]["failures"]

    from hg.grounding.phase1 import analyze_run_dir

    again = analyze_run_dir(run.root)
    assert compare_analyses(analysis, again) == []
    assert (run.root / "diagnostics.json").exists()
    assert run.report_path.exists()


def test_phase1_artifacts_are_strict_json(tmp_path):
    import json

    cfg = phase1_config(tmp_path, n_bootstrap=20)
    run, _, _ = run_phase1(cfg)
    paths = [run.analysis_path, run.manifest_path, run.root / "diagnostics.json"]
    paths += sorted((run.root / "episodes").glob("*.jsonl"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in ("NaN", "Infinity"):
            assert token not in text, f"{path.name} contains a bare {token}"
        del text


def test_episode_costs_are_recomputable_from_the_artifact(tmp_path):
    """The whole point of the episode file: repricing without rerunning."""
    from hg.grounding.episodes import read_episodes, write_episodes

    cfg = phase1_config(tmp_path)
    episodes, _, _ = run_sweep(cfg)
    subset = [e for e in episodes if e.policy == "learned"][:200]
    path = tmp_path / "e.jsonl"
    write_episodes(path, subset)
    assert np.allclose(episode_costs(subset), episode_costs(read_episodes(path)))


def test_feature_level_table_enumerates_the_coarse_feature_space(tmp_path):
    """Documents why `learned` and `bayes_optimal` can be identical: with m
    observations there are only a few distinct agreement levels."""
    cfg = phase1_config(tmp_path)
    _, _, diagnostics = run_sweep(cfg)
    levels = diagnostics["feature_levels"]
    # m = 5 gives three distinct margins: 5-0, 4-1, 3-2.
    assert len(levels) == 3
    for level in levels:
        assert abs(level["learned_p_wrong"] - level["bayes_p_wrong"]) < 0.05
        assert abs(level["realized_error_rate"] - level["bayes_p_wrong"]) < 0.05


def test_a_phase1_run_directory_is_never_silently_overwritten(tmp_path, monkeypatch):
    """The guard is duplicated per phase, so it is tested per phase."""
    from hg.grounding import phase1 as phase1_module

    monkeypatch.setattr(phase1_module, "run_id", lambda *a, **k: "pinned_run_id")

    cfg = phase1_config(tmp_path, task={
        "kind": "noisy_index", "n_bits": 4, "n_items": 200,
    }, lambda_grid=[0.1, 0.2], n_bootstrap=0)
    episodes, _, diagnostics = run_sweep(cfg)
    run = phase1_module.write_run(cfg, episodes, diagnostics)
    assert run.root.name == "pinned_run_id"

    with pytest.raises(FileExistsError, match="already exists"):
        phase1_module.write_run(cfg, episodes, diagnostics)
