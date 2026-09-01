"""Phase 0 orchestration: config in, run directory out, gate verdict.

One module so the runner script, the re-analysis script and the tests all drive
the same code path -- the same reason `hg.pipeline` exists for the M2-LLM
experiment.

The load-bearing structural property is that `build_analysis` takes episodes and
nothing else. The runner writes episodes to disk and then analyses them *by
reading them back*, so the numbers in `report.md` are by construction the
numbers a later re-analysis will produce. Acceptance criterion 6 (plan section
8) is not a promise about this code; it is a comparison the runner performs and
fails on.
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
from .config import GroundingConfig
from .costs import (
    OVERHEAD_DEFINED,
    ArmMetrics,
    aggregate_all,
)
from .episodes import GroundingEpisode, GroundingRunDir, read_episodes, write_episodes
from .synthetic import (
    INDEX_TASK_VERSION,
    IndexTaskSpec,
    generate_index_items,
    run_arm,
)

log = logging.getLogger(__name__)

# H(J | M_A) for the index problem, in bits. Exact by construction: the AI's
# index K says which bit decides, never its value.
RESIDUAL_DEFICIT_BITS = 1.0


# ------------------------------------------------------------------ running ---
def run_sweep(cfg: GroundingConfig) -> list[GroundingEpisode]:
    """Generate the task at every n and run every policy over it.

    Every policy sees the *same* items at a given n, which makes the arms paired
    and the cost comparison at equal decision quality meaningful.
    """
    episodes: list[GroundingEpisode] = []
    for n in cfg.sweep.n_bits:
        items = generate_index_items(
            IndexTaskSpec(n_bits=n, n_items=cfg.sweep.n_items, seed=cfg.seed)
        )
        for policy in cfg.policies:
            episodes.extend(
                run_arm(policy, items, seed=cfg.seed, lambda_bit=cfg.lambda_bit)
            )
        log.info("n=%d: %d items x %d policies", n, len(items), len(cfg.policies))
    return episodes


def environment() -> dict[str, Any]:
    """Package versions worth recording (plan section 25)."""
    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod in ("numpy", "yaml", "matplotlib"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - absence is the information here
            versions[mod] = None
    return versions


def write_run(
    cfg: GroundingConfig,
    episodes: Sequence[GroundingEpisode],
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> GroundingRunDir:
    """Create the run directory and write config, manifest and episodes."""
    root = Path(output_root or cfg.output_root) / run_id(tag or cfg.name)
    if root.exists():
        # Plan section 25: never overwrite a previous run silently.
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

    write_json(
        run.manifest_path,
        {
            "run_id": root.name,
            "experiment": cfg.name,
            "phase": "phase0",
            "task": cfg.task_kind,
            "task_version": INDEX_TASK_VERSION,
            "residual_deficit_bits": RESIDUAL_DEFICIT_BITS,
            "seed": cfg.seed,
            "lambda_bit": cfg.lambda_bit,
            "n_bits_sweep": list(cfg.sweep.n_bits),
            "n_items_per_arm": cfg.sweep.n_items,
            "policies": list(cfg.policies),
            "config_path": str(cfg.source_path) if cfg.source_path else None,
            "episode_counts": counts,
            # No model, on purpose: Phase 0 must not be quotable as evidence
            # about any system's behaviour.
            "model_checkpoint": None,
            "backend": "none (analytic simulator)",
            "git_commit": git_commit(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": environment(),
        },
    )
    return run


def read_run_episodes(run: GroundingRunDir) -> list[GroundingEpisode]:
    """Read every episode back off disk, validating as it goes."""
    policies = run.available_policies()
    if not policies:
        raise FileNotFoundError(f"no episode files under {run.root}/episodes")
    episodes: list[GroundingEpisode] = []
    for policy in policies:
        episodes.extend(read_episodes(run.episodes(policy), strict=True))
    return episodes


# ----------------------------------------------------------------- analysis ---
def theory_expectation(policy: str, n_bits: int) -> dict[str, Any]:
    """What the theorem says this arm must produce (plan sections 6 and 7).

    Kept as data rather than baked into the assertions so the report can print
    predicted and observed side by side -- the point of Phase 0 is the
    comparison, not the pass/fail bit.
    """
    if policy == "act":
        return {"accuracy": 0.5, "external_bits": 0.0, "omega": None}
    if policy == "ask_generic":
        return {"accuracy": 1.0, "external_bits": float(n_bits), "omega": n_bits - 1.0}
    if policy == "ask_targeted":
        return {"accuracy": 1.0, "external_bits": 1.0, "omega": 0.0}
    if policy == "ask_targeted_ablated":
        # The query and its cost are unchanged; only sufficiency is removed.
        return {"accuracy": 0.5, "external_bits": 1.0, "omega": None}
    raise ValueError(f"no theoretical expectation registered for policy {policy!r}")


def build_analysis(cfg: GroundingConfig, episodes: Sequence[GroundingEpisode]) -> dict[str, Any]:
    """Every computed quantity, from episodes alone. No task regeneration."""
    arms = aggregate_all(
        episodes, n_boot=cfg.n_bootstrap, alpha=cfg.alpha, seed=cfg.seed
    )
    arm_rows: list[dict[str, Any]] = []
    for arm in arms:
        row = arm.to_dict()
        row["theory"] = theory_expectation(arm.policy, arm.n_bits or 0)
        arm_rows.append(row)

    gate = evaluate_gate(cfg, arms)
    return {
        "experiment": cfg.name,
        "phase": "phase0",
        "task": cfg.task_kind,
        "task_version": INDEX_TASK_VERSION,
        "residual_deficit_bits": RESIDUAL_DEFICIT_BITS,
        "n_episodes": len(episodes),
        "arms": arm_rows,
        "compression": compression_summary(arms),
        "gate": gate,
    }


def compression_summary(arms: Sequence[ArmMetrics]) -> list[dict[str, Any]]:
    """Targeted vs generic ask at equal decision quality -- the headline.

    This is the Phase 0 shadow of gate G2: same final distortion, how many
    fewer bits did the source have to send? Reported only where both arms
    actually closed the deficit, because a cost saving bought by answering
    worse is not a saving.
    """
    by_n: dict[int | None, dict[str, ArmMetrics]] = {}
    for arm in arms:
        by_n.setdefault(arm.n_bits, {})[arm.policy] = arm

    out: list[dict[str, Any]] = []
    for n, arms_at_n in sorted(by_n.items(), key=lambda kv: kv[0] or -1):
        generic = arms_at_n.get("ask_generic")
        targeted = arms_at_n.get("ask_targeted")
        if generic is None or targeted is None:
            continue
        equal_quality = (
            abs(generic.mean_loss_after - targeted.mean_loss_after) < 1e-12
            and generic.sufficient
            and targeted.sufficient
        )
        row: dict[str, Any] = {
            "n_bits": n,
            "equal_decision_quality": equal_quality,
            "generic_bits": generic.mean_external_bits,
            "targeted_bits": targeted.mean_external_bits,
            "generic_omega": generic.coordination_overhead,
            "targeted_omega": targeted.coordination_overhead,
        }
        if equal_quality and targeted.mean_external_bits > 0:
            row["bits_saved"] = generic.mean_external_bits - targeted.mean_external_bits
            row["compression_ratio"] = (
                generic.mean_external_bits / targeted.mean_external_bits
            )
        else:
            row["bits_saved"] = None
            row["compression_ratio"] = None
        out.append(row)
    return out


def evaluate_gate(cfg: GroundingConfig, arms: Sequence[ArmMetrics]) -> dict[str, Any]:
    """Check the declared acceptance criteria (plan section 8).

    Returns failures rather than raising, so a run always produces a readable
    report; the runner turns a non-empty failure list into a non-zero exit.
    """
    g = cfg.gate
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            failures.append(f"{name}: {detail}")

    for arm in arms:
        label = f"{arm.policy}@n={arm.n_bits}"
        theory = theory_expectation(arm.policy, arm.n_bits or 0)

        if arm.policy == "act":
            # Criterion 1: ACT approaches chance.
            delta = abs(arm.accuracy - 0.5)
            record(
                f"act_accuracy_at_chance[{label}]",
                delta <= g.act_accuracy_tol,
                f"accuracy {arm.accuracy:.4f}, |diff from 0.5| = {delta:.4f} "
                f"(tolerance {g.act_accuracy_tol})",
            )
            record(
                f"act_costs_nothing[{label}]",
                arm.mean_external_bits == 0.0,
                f"external bits {arm.mean_external_bits}",
            )

        if arm.policy in ("ask_generic", "ask_targeted"):
            # Criteria 2 and 3: grounded decisions are exactly right.
            record(
                f"ask_accuracy[{label}]",
                arm.accuracy >= g.min_ask_accuracy,
                f"accuracy {arm.accuracy:.6f} (required >= {g.min_ask_accuracy})",
            )

        if arm.policy == "ask_generic" and g.expect_generic_bits_equal_n:
            # Criterion 4: generic response cost scales linearly with n.
            record(
                f"generic_bits_equal_n[{label}]",
                arm.mean_external_bits == float(arm.n_bits),
                f"external bits {arm.mean_external_bits} vs n = {arm.n_bits}",
            )
            record(
                f"generic_omega_is_n_minus_1[{label}]",
                arm.coordination_overhead == theory["omega"],
                f"Omega {arm.coordination_overhead} vs expected {theory['omega']}",
            )

        if arm.policy == "ask_targeted":
            # Criterion 5: the targeted response stays one bit at every n.
            record(
                f"targeted_bits_constant[{label}]",
                arm.mean_external_bits == g.expect_targeted_bits,
                f"external bits {arm.mean_external_bits} "
                f"(required {g.expect_targeted_bits})",
            )
            record(
                f"targeted_omega_is_zero[{label}]",
                arm.coordination_overhead == 0.0
                and arm.overhead_status == OVERHEAD_DEFINED,
                f"Omega {arm.coordination_overhead} "
                f"(status {arm.overhead_status})",
            )

        if arm.policy == "ask_targeted_ablated":
            # Joint sufficiency (plan section 24): removing the queried
            # variable from M_H must make perfect recovery impossible, while
            # the bits are still paid.
            record(
                f"ablation_breaks_recovery[{label}]",
                arm.accuracy <= g.max_ablated_accuracy,
                f"accuracy {arm.accuracy:.4f} "
                f"(required <= {g.max_ablated_accuracy})",
            )
            record(
                f"ablation_still_pays[{label}]",
                arm.mean_external_bits == g.expect_targeted_bits,
                f"external bits {arm.mean_external_bits}; the ablated arm must "
                "pay the same cost for a useless answer",
            )
            record(
                f"ablation_omega_undefined[{label}]",
                arm.coordination_overhead is None,
                f"Omega {arm.coordination_overhead} with status "
                f"{arm.overhead_status}; Omega must not be reported for "
                "communication that did not close the deficit",
            )

    passed = not failures
    return {
        "gate": "G0",
        "title": "Mathematical sanity: the simulator reproduces known theory",
        "passed": passed,
        "n_checks": len(checks),
        "failures": failures,
        "checks": checks,
        "verdict": (
            "Phase 0 reproduces the index theorem exactly. The abstractions "
            "(AI private state, external private state, decision variable, "
            "query, response cost, final decision, episode, analysis pipeline) "
            "carry the result, so the same infrastructure can carry the LM "
            "experiments. Nothing here is evidence about any model."
            if passed
            else "Phase 0 FAILED. Per plan section 27, gate G0 is a stop: do "
            "not proceed to Phase 1 until the simulator reproduces the theory."
        ),
    }


# ------------------------------------------------------------------ figures ---
def write_figures(run: GroundingRunDir, analysis: Mapping[str, Any]) -> dict[str, bool]:
    """Accuracy, response cost, and coordination overhead against n."""
    arms = analysis["arms"]
    policies = sorted({a["policy"] for a in arms})

    def series(field: str, *, only_defined: bool = False) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for policy in policies:
            points = [
                (float(a["n_bits"]), float(a[field]))
                for a in arms
                if a["policy"] == policy
                and a["n_bits"] is not None
                and a[field] is not None
                and (not only_defined or a["overhead_status"] == OVERHEAD_DEFINED)
            ]
            if points:
                out[policy] = sorted(points)
        return out

    wrote = {}
    wrote["accuracy"] = plot_grounding_sweep(
        run.figure("phase0_accuracy.png"),
        series("accuracy"),
        xlabel="n = number of bits the source holds",
        ylabel="decision accuracy",
        title="Phase 0: decision quality is not what separates the ask policies",
        # Accuracy is bounded in [0, 1], so a log2 x-axis cannot be misread as
        # a growth rate here -- it just spreads the geometric sweep evenly.
        logx=True,
        ylim_bottom=0.0,
        hline=0.5,
    )
    wrote["response_bits"] = plot_grounding_sweep(
        run.figure("phase0_response_bits.png"),
        series("mean_external_bits"),
        xlabel="n = number of bits the source holds",
        ylabel="mean external response cost (bits)",
        title="Phase 0: generic escalation costs n bits, a targeted query costs 1",
        # Linear on both axes: the generic cost is linear in n, and a log x
        # against a linear y would draw that straight line as a curve.
        logx=False,
        logy=False,
    )
    wrote["overhead"] = plot_grounding_sweep(
        run.figure("phase0_coordination_overhead.png"),
        series("coordination_overhead", only_defined=True),
        xlabel="n = number of bits the source holds",
        ylabel="coordination overhead Omega (bits)",
        title="Phase 0: Omega = n - 1 for generic escalation, 0 for a targeted query",
        logx=False,
        hline=0.0,
    )
    return wrote


# ------------------------------------------------------------------- report ---
def render_report(cfg: GroundingConfig, analysis: Mapping[str, Any], run: GroundingRunDir) -> str:
    """The Phase 0 table and the gate verdict, as markdown."""
    gate = analysis["gate"]
    lines: list[str] = [
        f"# {analysis['experiment']} — Phase 0 (gate G0)",
        "",
        f"Task `{analysis['task']}` version `{analysis['task_version']}`, "
        f"{analysis['n_episodes']} episodes, seed {cfg.seed}.",
        "",
        "**No model was involved.** This run checks the HG-v2 abstractions "
        "against the index theorem, where "
        f"H(J | M_A) = {analysis['residual_deficit_bits']:.0f} bit exactly. "
        "None of the numbers below are evidence about an LLM.",
        "",
        "## Policies",
        "",
        "| policy | what it does | predicted accuracy | predicted response bits |",
        "|---|---|---:|---:|",
        "| `act` | guess from M_A alone | 0.5 | 0 |",
        "| `ask_generic` | source transmits all n bits | 1.0 | n |",
        "| `ask_targeted` | AI asks for X_K | 1.0 | 1 |",
        "| `ask_targeted_ablated` | source does not hold X_K (control) | 0.5 | 1 |",
        "",
        "## Results",
        "",
        "| n | policy | accuracy | 95% CI | bits | Ω | ΔD | ΔD/bit | informative |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for a in analysis["arms"]:
        ci = a.get("accuracy_ci")
        ci_s = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "—"
        omega = a["coordination_overhead"]
        omega_s = f"{omega:.2f}" if omega is not None else f"— ({a['overhead_status']})"
        eff = a["information_efficiency"]
        eff_s = f"{eff:.4f}" if eff is not None else "—"
        inf = a["informative_response_rate"]
        inf_s = "—" if inf is None else f"{inf:.2f}"
        lines.append(
            f"| {a['n_bits']} | `{a['policy']}` | {a['accuracy']:.4f} | {ci_s} | "
            f"{a['mean_external_bits']:.2f} | {omega_s} | "
            f"{a['delta_distortion']:+.4f} | {eff_s} | {inf_s} |"
        )

    lines += [
        "",
        "## Targeted vs generic at equal decision quality",
        "",
        "The comparison Phase 0 exists to make. Both ask policies reach "
        "identical final distortion, so the only difference is what the source "
        "had to transmit.",
        "",
        "| n | equal quality | generic bits | targeted bits | bits saved | compression |",
        "|---:|:---:|---:|---:|---:|---:|",
    ]
    for c in analysis["compression"]:
        saved = c["bits_saved"]
        ratio = c["compression_ratio"]
        lines.append(
            f"| {c['n_bits']} | {'yes' if c['equal_decision_quality'] else 'no'} | "
            f"{c['generic_bits']:.2f} | {c['targeted_bits']:.2f} | "
            f"{'—' if saved is None else f'{saved:.2f}'} | "
            f"{'—' if ratio is None else f'{ratio:.1f}x'} |"
        )

    failures = gate["failures"]
    lines += [
        "",
        "## Gate G0",
        "",
        f"**{'PASSED' if gate['passed'] else 'FAILED'}** — {gate['title']} "
        f"({gate['n_checks']} checks).",
        "",
        gate["verdict"],
    ]
    if failures:
        lines += ["", "Failures:", ""]
        lines += [f"- {f}" for f in failures]

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python experiments/hg_v2/run_phase0.py --config "
        + (str(cfg.source_path) if cfg.source_path else "<config>"),
        f"uv run python experiments/hg_v2/analyze.py --run-dir {run.root}",
        "```",
        "",
        "The second command recomputes every number above from "
        "`episodes/*.jsonl` with no task regeneration and no model calls.",
        "",
    ]
    return "\n".join(lines)


# -------------------------------------------------------------- entry points ---
def analyze_run_dir(
    run_root: str | Path, config_path: str | Path | None = None
) -> dict[str, Any]:
    """Re-run the whole analysis over a saved run directory.

    The config comes from the run directory itself, so a re-analysis uses the
    configuration the run actually used rather than whatever `configs/` says
    today -- the same rule as `hg.pipeline.reanalyze`.
    """
    from .config import load_grounding_config

    run = GroundingRunDir(run_root)
    cfg = load_grounding_config(config_path or run.config_path)
    episodes = read_run_episodes(run)
    analysis = build_analysis(cfg, episodes)
    write_json(run.analysis_path, analysis)
    write_figures(run, analysis)
    run.report_path.write_text(render_report(cfg, analysis, run), encoding="utf-8")
    return analysis


def run_phase0(
    cfg: GroundingConfig,
    *,
    output_root: Path | None = None,
    tag: str | None = None,
) -> tuple[GroundingRunDir, dict[str, Any], list[str]]:
    """Run the sweep, save it, and analyse it *from disk*.

    Returns the run directory, the analysis, and any reproducibility failures.
    The analysis is deliberately computed from the written episodes rather than
    from the in-memory ones, and then compared against the in-memory result:
    that comparison is acceptance criterion 6, and it can only fail loudly.
    """
    episodes = run_sweep(cfg)
    run = write_run(cfg, episodes, output_root=output_root, tag=tag)

    live = build_analysis(cfg, episodes)
    from_disk = analyze_run_dir(run.root)

    problems = compare_analyses(live, from_disk)
    return run, from_disk, problems


def _same(a: Any, b: Any) -> bool:
    """Equality that treats two NaNs as equal.

    A metric may legitimately be NaN, and `nan != nan` would report every such
    field as a round-trip corruption -- turning this check into noise exactly
    when it needs to be trusted.
    """
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


def compare_analyses(live: Mapping[str, Any], from_disk: Mapping[str, Any]) -> list[str]:
    """Acceptance criterion 6: saved artifacts must reproduce the live metrics.

    Compares the metric fields arm by arm. Bootstrap CIs are included: they are
    seeded, so a difference there means the episode ordering or content changed
    across the round-trip, which is exactly the kind of silent corruption this
    check exists to catch.
    """
    problems: list[str] = []
    live_arms = {(a["policy"], a["n_bits"]): a for a in live["arms"]}
    disk_arms = {(a["policy"], a["n_bits"]): a for a in from_disk["arms"]}

    if set(live_arms) != set(disk_arms):
        problems.append(
            f"arms differ across the round-trip: live {sorted(map(str, live_arms))} "
            f"vs saved {sorted(map(str, disk_arms))}"
        )
        return problems

    for key, la in live_arms.items():
        da = disk_arms[key]
        for field in (
            "accuracy", "mean_loss_before", "mean_loss_after", "delta_distortion",
            "mean_external_bits", "mean_external_cost", "coordination_overhead",
            "overhead_status", "external_query_rate", "information_efficiency",
            "n_items", "accuracy_ci", "external_bits_ci",
            "informative_response_rate", "sufficient",
        ):
            if not _same(la[field], da[field]):
                problems.append(
                    f"{key[0]}@n={key[1]}: {field} = {la[field]!r} live but "
                    f"{da[field]!r} recomputed from saved episodes"
                )
    return problems
