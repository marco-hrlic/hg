"""Phase 1 orchestration: learned routing on controlled tasks.

Plan sections 9 to 11. The task is `noisy_index`, where the AI sometimes
already holds enough to decide and sometimes does not, so the optimal route
genuinely varies per item and is known from the counterfactuals.

Gate G1 has a specific shape and it is not "the router looks reasonable":

    "Learned routing must beat BOTH always ACT and always ASK for some
     nontrivial range of external-information costs."   (plan section 11)

So the run sweeps lambda, evaluates every policy on the *same held-out items*,
and requires the learned router's mean total cost to be strictly below both
fixed policies with an item-paired bootstrap interval that excludes zero, at at
least `min_winning_lambdas` prices. A win at a single hand-picked lambda is not
a range.

Two reference policies are reported but excluded from the gate, because beating
them is not the claim:

    oracle         routes on the realized counterfactual loss. Clairvoyant, so
                   its cost is the floor and its regret is zero by definition.
    bayes_optimal  routes on the exact posterior P(acting now is wrong | the
                   AI's observations), marginalizing the unknown noise level.
                   This is the ceiling for any router built on those features:
                   same evidence, plus perfect knowledge of the noise
                   distribution that the learned router must infer. The gap
                   between it and the learned router is estimation error, not a
                   limit of the architecture.
"""

from __future__ import annotations

import logging
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..io import git_commit, run_id, write_json, write_yaml
from ..report.figures import plot_grounding_sweep
from ..routing.decision_state import FEATURE_NAMES, DecisionState, agreement_features
from ..routing.grounding_router import (
    FIXED_POLICIES,
    LogisticRouter,
    Route,
    oracle_route,
    route_from_expected_cost,
)
from .config import GroundingConfig
from .costs import (
    RouteMetrics,
    aggregate_route_arm,
    paired_cost_advantage,
)
from .episodes import GroundingEpisode, GroundingRunDir, read_episodes, write_episodes
from .noisy_index import (
    ASK_BITS,
    bayes_act_error_probability,
    NOISY_INDEX_TASK_VERSION,
    NoisyIndexItem,
    NoisyIndexTaskSpec,
    build_episode,
    generate_noisy_index_items,
    split_items,
    true_act_error_probability,
)
from .phase0 import environment

log = logging.getLogger(__name__)

POLICIES = ("always_act", "always_ask", "learned", "oracle", "bayes_optimal")


# ------------------------------------------------------------------ routing ---
def decision_state_for(
    item: NoisyIndexItem,
    *,
    p_wrong: float | None,
    exact: bool = False,
) -> dict[str, Any]:
    """The decision state recorded on an episode.

    `p_wrong` is an *estimate* for the learned router and the exact generative
    probability for the reference policies, and `exact` says which -- the
    distinction the plan insists on keeping visible (sections 4.1 and 18).
    """
    features = agreement_features(item.observations)
    state = DecisionState(
        item_id=item.item_id,
        candidate_actions=["0", "1"],
        predicted_action=item.act_now_action,
        expected_loss_now=p_wrong,
        # More internal computation cannot help: the AI already has all its
        # observations, and X_K itself is not in M_A.
        internal_reducibility=0.0,
        # A targeted ask resolves the decision, so the value of external
        # information is exactly the loss it avoids.
        external_value=p_wrong,
        human_value=p_wrong,
        contestedness=0.0,
        consequence=1.0,
        exact=exact,
        features=features,
    )
    return state.to_dict()


def route_for_policy(
    policy: str,
    item: NoisyIndexItem,
    *,
    lambda_bit: float,
    router: LogisticRouter | None,
    task: NoisyIndexTaskSpec | None = None,
) -> tuple[Route, float | None, bool]:
    """Choose a route. Returns (route, p_wrong estimate, whether it is exact)."""
    if policy == "always_act":
        return Route.ACT, None, False
    if policy == "always_ask":
        return Route.ASK_TARGETED, None, False
    if policy == "oracle":
        return (
            oracle_route(item.loss_act, lambda_bit=lambda_bit, ask_bits=ASK_BITS),
            item.loss_act,
            True,
        )
    if policy == "bayes_optimal":
        if task is None:
            raise ValueError("the bayes_optimal policy needs the task spec")
        p = bayes_act_error_probability(item.observations, task)
        return (
            route_from_expected_cost(p, lambda_bit=lambda_bit, ask_bits=ASK_BITS),
            p,
            True,
        )
    if policy == "learned":
        if router is None:
            raise ValueError("the learned policy needs a fitted router")
        features = agreement_features(item.observations)
        p = float(router.predict_proba([features])[0])
        return (
            route_from_expected_cost(p, lambda_bit=lambda_bit, ask_bits=ASK_BITS),
            p,
            False,
        )
    raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")


def fit_router(
    items: Sequence[NoisyIndexItem], cfg: GroundingConfig
) -> LogisticRouter:
    """Fit the router on the training split.

    Supervised against r*, per plan section 9. The label is the realized
    "acting now would be wrong" indicator, which for 0 < lambda*B < 1 *is* the
    oracle route -- see the module docstring of
    `hg.routing.grounding_router` for why fitting the probability instead of a
    per-lambda classifier is the same target with a repricing advantage.
    """
    router_opts = dict(cfg.raw.get("router") or {})
    router = LogisticRouter(
        learning_rate=float(router_opts.get("learning_rate", 0.5)),
        n_steps=int(router_opts.get("n_steps", 2000)),
        l2=float(router_opts.get("l2", 1e-4)),
    )
    return router.fit(
        [agreement_features(i.observations) for i in items],
        [i.loss_act for i in items],
    )


def run_sweep(
    cfg: GroundingConfig,
) -> tuple[list[GroundingEpisode], LogisticRouter, dict[str, Any]]:
    """Fit the router once, then run every policy at every lambda.

    The router is fitted on the training split only and every policy is
    evaluated on the same held-out items in the same order, which is what makes
    the paired cost comparison in the gate valid.
    """
    task = cfg.phase1_task
    if task is None:
        raise ValueError("phase 1 requires a `task` section with kind: noisy_index")

    items = generate_noisy_index_items(task)
    train, evaluate = split_items(
        items, train_fraction=cfg.train_fraction, seed=cfg.seed
    )
    router = fit_router(train, cfg)
    log.info(
        "router fitted on %d items, evaluating %d, weights=%s",
        len(train), len(evaluate), router.weights,
    )

    episodes: list[GroundingEpisode] = []
    for lambda_bit in cfg.lambda_grid:
        for policy in cfg.policies:
            for item in evaluate:
                route, p_wrong, exact = route_for_policy(
                    policy, item, lambda_bit=lambda_bit, router=router, task=task
                )
                episodes.append(
                    build_episode(
                        item,
                        route,
                        policy=policy,
                        decision_state=decision_state_for(
                            item, p_wrong=p_wrong, exact=exact
                        ),
                        lambda_bit=lambda_bit,
                        seed=cfg.seed,
                    )
                )

    diagnostics = {
        "feature_levels": feature_level_table(evaluate, router, task),
        "n_items_total": len(items),
        "n_train": len(train),
        "n_eval": len(evaluate),
        "train_fraction": cfg.train_fraction,
        "router": router.to_dict(),
        "base_act_error_rate": float(
            sum(i.loss_act for i in evaluate) / len(evaluate)
        ),
        "self_sufficient_share": float(
            sum(i.self_sufficient for i in evaluate) / len(evaluate)
        ),
    }
    return episodes, router, diagnostics


def feature_level_table(
    items: Sequence[NoisyIndexItem],
    router: LogisticRouter,
    task: NoisyIndexTaskSpec,
) -> list[dict[str, Any]]:
    """Every distinct feature vector, with the learned and exact error estimates.

    This table is why Phase 1's `learned` and `bayes_optimal` policies can come
    out identical, and it is worth writing down rather than leaving as a
    surprise. With m observations there are only a handful of distinct
    agreement levels, both policies are monotone thresholds on that ordering,
    and a threshold on a coarse ordering has few places to land. Identical
    routing is then the *correct* outcome, not a leak -- and this table lets a
    reader check that by comparing the two estimates level by level.

    The consequence for interpretation: Phase 1 can establish that routing has
    value (gate G1), but this task is too coarse to discriminate a good
    estimator from a perfect one.
    """
    buckets: dict[tuple[float, ...], list[NoisyIndexItem]] = {}
    for item in items:
        f = agreement_features(item.observations)
        buckets.setdefault(tuple(f[name] for name in FEATURE_NAMES), []).append(item)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items()):
        features = dict(zip(FEATURE_NAMES, key))
        rows.append(
            {
                **features,
                "n_items": len(group),
                "learned_p_wrong": float(router.predict_proba([features])[0]),
                "bayes_p_wrong": bayes_act_error_probability(
                    group[0].observations, task
                ),
                "realized_error_rate": float(
                    sum(i.loss_act for i in group) / len(group)
                ),
            }
        )
    return rows


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
            "phase": "phase1",
            "task": cfg.task_kind,
            "task_version": NOISY_INDEX_TASK_VERSION,
            "seed": cfg.seed,
            "lambda_grid": list(cfg.lambda_grid),
            "policies": list(cfg.policies),
            "episode_counts": counts,
            "config_path": str(cfg.source_path) if cfg.source_path else None,
            "model_checkpoint": None,
            "backend": "none (analytic simulator + logistic router)",
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


# ------------------------------------------------------------------ analysis ---
def build_analysis(
    cfg: GroundingConfig, episodes: Sequence[GroundingEpisode]
) -> dict[str, Any]:
    """Every Phase 1 quantity, from episodes alone."""
    grouped: dict[tuple[str, float], list[GroundingEpisode]] = {}
    for e in episodes:
        grouped.setdefault(
            (e.policy, float(e.metadata["lambda_bit"])), []
        ).append(e)

    arms = {
        key: aggregate_route_arm(
            eps, n_boot=cfg.n_bootstrap, alpha=cfg.alpha, seed=cfg.seed
        )
        for key, eps in grouped.items()
    }
    arm_rows = [
        arms[key].to_dict()
        for key in sorted(arms, key=lambda k: (k[1], k[0]))
    ]

    comparisons = build_comparisons(cfg, grouped, arms)
    gate = evaluate_gate(cfg, comparisons)
    validity = check_validity(arms)
    return {
        "experiment": cfg.name,
        "phase": "phase1",
        "task": cfg.task_kind,
        "task_version": NOISY_INDEX_TASK_VERSION,
        "n_episodes": len(episodes),
        "lambda_grid": sorted({float(e.metadata["lambda_bit"]) for e in episodes}),
        "arms": arm_rows,
        "comparisons": comparisons,
        "validity": validity,
        "gate": gate,
    }


def check_validity(
    arms: Mapping[tuple[str, float], RouteMetrics], *, tolerance: float = 1e-9
) -> dict[str, Any]:
    """Sanity checks that would indicate a leak rather than a result.

    `bayes_optimal` uses the same features as the learned router plus exact
    knowledge of the noise distribution, so it is the ceiling. A learned router
    materially below its cost is evidence that the router saw something it
    should not have -- reported loudly and separately from the gate, because it
    would invalidate the gate rather than fail it.

    Matching the ceiling exactly is expected here and is not a problem: see
    `feature_level_table` for why the feature space is coarse enough that both
    policies land on the same thresholds.
    """
    beaten: list[dict[str, Any]] = []
    matched: list[float] = []
    for (policy, lam), arm in arms.items():
        if policy != "learned":
            continue
        ceiling = arms.get(("bayes_optimal", lam))
        if ceiling is None:
            continue
        if arm.mean_total_cost < ceiling.mean_total_cost - tolerance:
            beaten.append(
                {
                    "lambda_bit": lam,
                    "learned_cost": arm.mean_total_cost,
                    "ceiling_cost": ceiling.mean_total_cost,
                }
            )
        elif abs(arm.mean_total_cost - ceiling.mean_total_cost) <= tolerance:
            matched.append(lam)
    return {
        "learned_beats_its_ceiling_at": beaten,
        "learned_matches_ceiling_at": sorted(matched),
        "note": (
            "The learned router matching bayes_optimal exactly is expected on "
            "this task: the agreement features take only a few distinct values, "
            "and both policies are monotone thresholds on that ordering. It "
            "means Phase 1 demonstrates routing value without discriminating a "
            "good estimator from a perfect one. Beating the ceiling would be a "
            "leak, not a result."
            if not beaten
            else "The learned router is cheaper than bayes_optimal, which uses "
            "the same features plus exact knowledge of the noise distribution. "
            "Check for information leaking into the router's features before "
            "reading anything else in this run."
        ),
    }


def build_comparisons(
    cfg: GroundingConfig,
    grouped: Mapping[tuple[str, float], Sequence[GroundingEpisode]],
    arms: Mapping[tuple[str, float], RouteMetrics],
) -> list[dict[str, Any]]:
    """Learned vs each fixed policy at every lambda, item-paired."""
    out: list[dict[str, Any]] = []
    lambdas = sorted({lam for _, lam in grouped})
    for lam in lambdas:
        learned_key = ("learned", lam)
        if learned_key not in grouped:
            continue
        row: dict[str, Any] = {
            "lambda_bit": lam,
            "learned_cost": arms[learned_key].mean_total_cost,
            "learned_regret": arms[learned_key].routing_regret,
            "learned_query_rate": arms[learned_key].external_query_rate,
            "beats": {},
        }
        for baseline in FIXED_POLICIES:
            key = (baseline, lam)
            if key not in grouped:
                continue
            adv = paired_cost_advantage(
                grouped[learned_key],
                grouped[key],
                name=f"learned_vs_{baseline}@lambda={lam}",
                n_boot=max(cfg.n_bootstrap, 1),
                alpha=cfg.alpha,
                seed=cfg.seed,
            )
            row["beats"][baseline] = {
                "baseline_cost": arms[key].mean_total_cost,
                "advantage": adv.point,
                "ci": [adv.ci_low, adv.ci_high],
                "p_value": adv.p_value,
                # Strictly better *and* the interval excludes zero. A point
                # advantage alone would let sampling noise open the gate.
                "significant": bool(adv.point > 0 and adv.ci_low > 0),
            }
        row["beats_both"] = bool(
            row["beats"]
            and len(row["beats"]) == len(FIXED_POLICIES)
            and all(v["significant"] for v in row["beats"].values())
        )
        for reference in ("oracle", "bayes_optimal"):
            key = (reference, lam)
            if key in arms:
                row[f"{reference}_cost"] = arms[key].mean_total_cost
        out.append(row)
    return out


def evaluate_gate(
    cfg: GroundingConfig, comparisons: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Gate G1 (plan section 11)."""
    winning = [c["lambda_bit"] for c in comparisons if c["beats_both"]]
    required = cfg.gate.min_winning_lambdas
    passed = len(winning) >= required

    failures: list[str] = []
    if not passed:
        failures.append(
            f"learned routing beat both fixed policies at {len(winning)} of "
            f"{len(comparisons)} lambda values ({winning}); "
            f"{required} are required for a nontrivial range"
        )

    return {
        "gate": "G1",
        "title": "Routing value: learned routing beats fixed ACT/ASK policies",
        "passed": passed,
        "winning_lambdas": winning,
        "n_lambdas": len(comparisons),
        "min_winning_lambdas": required,
        "failures": failures,
        "verdict": (
            "Learned routing beats both always-ACT and always-ASK over a range "
            "of external-information prices, with item-paired intervals that "
            "exclude zero. Routing on an internal signal has measurable value "
            "on this task. Note what this does not show: the task is synthetic "
            "and the query is already optimal, so gate G2 (query compression) "
            "is untouched by it."
            if passed
            else "Gate G1 not met. Plan section 11: investigate before moving "
            "on -- a router that cannot beat both fixed policies means either "
            "the internal signal carries no routing information, or the cost "
            "range swept contains no region where the tradeoff is live."
        ),
    }


# ------------------------------------------------------------------- figures ---
# Drawn dashed with hollow markers. The learned router can coincide exactly
# with its ceiling on this task, and a solid reference line under a solid
# learned line would be invisible -- see `feature_level_table`.
REFERENCE_STYLE = ("bayes_optimal", "oracle")


def write_figures(run: GroundingRunDir, analysis: Mapping[str, Any]) -> dict[str, bool]:
    arms = analysis["arms"]
    policies = sorted({a["policy"] for a in arms})

    def series(field: str) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for policy in policies:
            points = [
                (float(a["lambda_bit"]), float(a[field]))
                for a in arms
                if a["policy"] == policy and a[field] is not None
            ]
            if points:
                out[policy] = sorted(points)
        return out

    wrote: dict[str, bool] = {}
    wrote["cost"] = plot_grounding_sweep(
        run.figure("phase1_total_cost.png"),
        series("mean_total_cost"),
        xlabel="lambda = price of one external bit",
        ylabel="mean total cost  C = L + lambda*B",
        title="Phase 1: where learned routing beats both fixed policies",
        dashed=REFERENCE_STYLE,
    )
    wrote["regret"] = plot_grounding_sweep(
        run.figure("phase1_regret.png"),
        series("routing_regret"),
        xlabel="lambda = price of one external bit",
        ylabel="routing regret  C_chosen - C_oracle",
        title="Phase 1: routing regret against the clairvoyant oracle",
        dashed=REFERENCE_STYLE,
    )
    wrote["query_rate"] = plot_grounding_sweep(
        run.figure("phase1_query_rate.png"),
        series("external_query_rate"),
        xlabel="lambda = price of one external bit",
        ylabel="fraction of items routed to ASK",
        title="Phase 1: the learned router asks less as external bits get dearer",
        dashed=REFERENCE_STYLE,
    )
    wrote["route_accuracy"] = plot_grounding_sweep(
        run.figure("phase1_route_accuracy.png"),
        series("route_accuracy"),
        xlabel="lambda = price of one external bit",
        ylabel="fraction of items routed as the oracle would",
        title="Phase 1: route accuracy against the oracle route",
        dashed=REFERENCE_STYLE,
    )
    return wrote


# -------------------------------------------------------------------- report ---
def render_report(
    cfg: GroundingConfig, analysis: Mapping[str, Any], run: GroundingRunDir
) -> str:
    gate = analysis["gate"]
    diagnostics_path = run.root / "diagnostics.json"
    lines: list[str] = [
        f"# {analysis['experiment']} — Phase 1 (gate G1)",
        "",
        f"Task `{analysis['task']}` version `{analysis['task_version']}`, "
        f"{analysis['n_episodes']} episodes, seed {cfg.seed}.",
        "",
        "**No model was involved.** The AI's internal evidence is simulated and "
        "the router is a logistic regression on internal-agreement features. "
        "What this run can show is whether routing on an internal signal has "
        "value when the optimal route is known; it says nothing about an LLM.",
        "",
        "## The task",
        "",
        "The source holds `M_H = (X_0..X_{n-1})`, the AI holds the index `K`, "
        "and `J = X_K`. The AI also holds `m` noisy observations of `X_K`: on a "
        "share of items they are noiseless, so `J` is directly observable from "
        "`M_A` and asking is waste; elsewhere they are unreliable and a "
        "1-bit targeted ask resolves the decision. Route objective "
        "`C_r = L_r + λB_r`, and both counterfactual losses are stored on every "
        "episode so the oracle route and the regret are recomputable.",
        "",
        "## Policies",
        "",
        "| policy | routes on | in the gate? |",
        "|---|---|:---:|",
        "| `always_act` | never asks | baseline to beat |",
        "| `always_ask` | always asks | baseline to beat |",
        "| `learned` | estimated P(acting now is wrong) | **the claim** |",
        "| `bayes_optimal` | exact P(wrong \\| observations); the ceiling for "
        "these features | reference |",
        "| `oracle` | the realized counterfactual loss (clairvoyant) | reference |",
        "",
        "## Results",
        "",
        "| λ | policy | C = L + λB | route acc. | final loss | ask rate | regret |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for a in analysis["arms"]:
        lines.append(
            f"| {a['lambda_bit']:.2f} | `{a['policy']}` | "
            f"{a['mean_total_cost']:.4f} | {a['route_accuracy']:.4f} | "
            f"{a['mean_final_loss']:.4f} | {a['external_query_rate']:.4f} | "
            f"{a['routing_regret']:.4f} |"
        )

    lines += [
        "",
        "## Learned routing vs both fixed policies",
        "",
        "Advantage is `C_baseline − C_learned`, item-paired; positive means the "
        "router is cheaper. A λ counts as a win only if the router is strictly "
        "better than **both** baselines with a bootstrap interval excluding zero.",
        "",
        "| λ | vs always_act | vs always_ask | wins both |",
        "|---:|---|---|:---:|",
    ]
    for c in analysis["comparisons"]:
        cells = []
        for baseline in FIXED_POLICIES:
            b = c["beats"].get(baseline)
            if b is None:
                cells.append("—")
                continue
            mark = "✓" if b["significant"] else "✗"
            cells.append(
                f"{mark} {b['advantage']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]"
            )
        lines.append(
            f"| {c['lambda_bit']:.2f} | {cells[0]} | {cells[1]} | "
            f"{'**yes**' if c['beats_both'] else 'no'} |"
        )

    validity = analysis.get("validity") or {}
    if validity:
        lines += [
            "",
            "## Validity",
            "",
            validity["note"],
        ]
        if validity["learned_beats_its_ceiling_at"]:
            lines += [
                "",
                "**The learned router beat its own ceiling** at "
                f"λ = {[b['lambda_bit'] for b in validity['learned_beats_its_ceiling_at']]}. "
                "Treat this run as invalid until the feature path is audited.",
            ]

    lines += [
        "",
        "## Gate G1",
        "",
        f"**{'PASSED' if gate['passed'] else 'FAILED'}** — {gate['title']}.",
        "",
        f"Won at {len(gate['winning_lambdas'])} of {gate['n_lambdas']} prices "
        f"(λ = {gate['winning_lambdas']}); "
        f"{gate['min_winning_lambdas']} required.",
        "",
        gate["verdict"],
    ]
    if gate["failures"]:
        lines += ["", "Failures:", ""] + [f"- {f}" for f in gate["failures"]]

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python experiments/hg_v2/run_phase1.py --config "
        + (str(cfg.source_path) if cfg.source_path else "<config>"),
        f"uv run python experiments/hg_v2/analyze.py --run-dir {run.root}",
        "```",
        "",
        f"Router weights and split sizes: `{diagnostics_path}`.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- entry points ---
def analyze_run_dir(
    run_root: str | Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    from .config import load_grounding_config

    run = GroundingRunDir(run_root)
    cfg = load_grounding_config(config_path or run.config_path)
    episodes = read_run_episodes(run)
    analysis = build_analysis(cfg, episodes)
    write_json(run.analysis_path, analysis)
    write_figures(run, analysis)
    run.report_path.write_text(render_report(cfg, analysis, run), encoding="utf-8")
    return analysis


def run_phase1(
    cfg: GroundingConfig,
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> tuple[GroundingRunDir, dict[str, Any], list[str]]:
    """Run the sweep, save it, and analyse it from disk.

    Same discipline as Phase 0: the reported analysis is the one recomputed
    from the written episodes, compared against the in-memory result.
    """
    episodes, _router, diagnostics = run_sweep(cfg)
    run = write_run(cfg, episodes, diagnostics, output_root=output_root, tag=tag)

    live = build_analysis(cfg, episodes)
    from_disk = analyze_run_dir(run.root)
    return run, from_disk, compare_analyses(live, from_disk)


def compare_analyses(
    live: Mapping[str, Any], from_disk: Mapping[str, Any]
) -> list[str]:
    """Saved artifacts must reproduce the live metrics (plan section 8, item 6)."""
    problems: list[str] = []
    live_arms = {(a["policy"], a["lambda_bit"]): a for a in live["arms"]}
    disk_arms = {(a["policy"], a["lambda_bit"]): a for a in from_disk["arms"]}
    if set(live_arms) != set(disk_arms):
        problems.append("arms differ across the round-trip")
        return problems
    for key, la in live_arms.items():
        for field, value in la.items():
            other = disk_arms[key][field]
            if isinstance(value, float) and isinstance(other, float):
                if math.isnan(value) and math.isnan(other):
                    continue
            if value != other:
                problems.append(
                    f"{key[0]}@lambda={key[1]}: {field} = {value!r} live but "
                    f"{other!r} recomputed from saved episodes"
                )
    if live["gate"]["passed"] != from_disk["gate"]["passed"]:
        problems.append("the gate verdict changed across the round-trip")
    return problems
