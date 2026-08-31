"""End-to-end orchestration: config in, run directory out.

One module so the experiment scripts, the CLI and the tests all drive the same
code path. The stages are kept separate (`prepare_dataset`, `prepare_members`,
`run_inference`, `analyze`) because protocol section 14 asks for inference,
fine-tuning, dataset construction and analysis to be decoupled -- in practice
that means being able to re-run the analysis over saved prediction matrices
without loading a model, which is the difference between iterating on a
dependence measure in seconds and in GPU-hours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .anchors.agreement import clean_subset_warnings, panel_report
from .anchors.panel import (
    AnchoredItem,
    build_anchored_items,
    load_panel,
    save_panel,
    validate_labels,
)
from .config import ExperimentConfig, load_experiment, resolved_dict, validate_panel
from .diversity.overlap import overlap_matrix, overlap_report
from .diversity.splits import build_splits, split_summary
from .diversity.survival import (
    contested_transition,
    excess_survival,
    residual_survival,
    survival_shape,
)
from .finetune.lora import LoRASettings, train_lora
from .inference.runner import MemberSpec, run_condition
from .io import RunDir, read_predictions, run_id, write_json, write_manifest, write_predictions, write_yaml
from .metrics.bootstrap import align_matrices, bootstrap_metric, odds_ratio, paired_bootstrap
from .metrics.core import (
    clean_consensus_wrong,
    condition_metrics,
    consensus_wrong,
    contested_share_of_residual,
    disagreement_rate,
    diversity_gap,
    error_conversion,
)
from .metrics.dependence import dependence_report
from .metrics.matrix import ErrorMatrix, build_matrix, save_matrix
from .report.decision import decision_gate, evaluate_predictions, render_report
from .report.figures import plot_race, plot_survival
from .routing.strategies import run_race
from .schema import ItemResult

log = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResult:
    run_dir: RunDir
    config: ExperimentConfig
    items: list[AnchoredItem]
    rows: dict[str, list[ItemResult]] = field(default_factory=dict)
    matrices: dict[str, ErrorMatrix] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- dataset ---
def prepare_dataset(cfg: ExperimentConfig, data_root: Path = Path("data")) -> list[AnchoredItem]:
    """Build or load the human panel, then derive (y_ref, A_i) for every item."""
    spec = dict(cfg.dataset.anchors or {})
    source = spec.get("source", "file")

    if source == "generate":
        from .tasks.synthetic import SyntheticSpec, generate_synthetic_panel

        gen = dict(spec.get("spec") or {})
        gen.setdefault("seed", cfg.seed)
        panel = generate_synthetic_panel(
            SyntheticSpec(labels=tuple(cfg.dataset.labels), **gen)
        )
        # Written out so the generated panel is as inspectable as a real one.
        path = data_root / "anchors" / f"{cfg.dataset.name}_panel.jsonl"
        save_panel(path, panel)
        log.info("generated %d synthetic panel items -> %s", len(panel), path)
    elif source == "file":
        path = Path(spec.get("path") or (data_root / "anchors" / f"{cfg.dataset.name}_panel.jsonl"))
        if not path.exists():
            raise FileNotFoundError(
                f"anchor panel not found: {path}. The panel must carry raw "
                "annotator judgments, not a majority label -- the clean/contested "
                "split every headline metric conditions on cannot be recovered "
                "from a collapsed label (protocol section 6)."
            )
        panel = load_panel(path)
    else:
        raise ValueError(f"unknown anchor source {source!r}; expected 'generate' or 'file'")

    anchor_opts = dict(cfg.dataset.options.get("anchor", {}) or {})
    items = build_anchored_items(
        panel,
        majority_threshold=float(anchor_opts.get("majority_threshold", 0.5)),
        contested_margin=float(anchor_opts.get("contested_margin", 0.0)),
        tie_break=str(anchor_opts.get("tie_break", "lexicographic")),
    )
    validate_labels(items, cfg.dataset.labels)
    return items


# ---------------------------------------------------------------- members ---
def prepare_members(
    cfg: ExperimentConfig,
    condition: str,
    items: Sequence[AnchoredItem],
    *,
    dry_run_finetune: bool = True,
    adapter_root: str | Path = "adapters",
) -> tuple[list[MemberSpec], list[AnchoredItem], dict[str, Any]]:
    """Cut splits, adapt each member, and return specs plus the eval set."""
    cond = cfg.conditions[condition]
    n_members = len(cond.models)

    # Member ids must be unique even when a condition repeats one model config
    # five times (which is exactly what HS does).
    member_ids = [
        f"{condition}/{model}#s{seed}"
        for model, seed in zip(cond.models, cond.adapter_seeds)
    ]
    if len(set(member_ids)) != n_members:
        raise ValueError(
            f"condition {condition}: member ids collide {member_ids}; "
            "repeated (model, adapter_seed) pairs would be identical members"
        )

    n_eval = cfg.options.get("n_eval", cfg.dataset.n_eval)
    n_train = cfg.options.get("n_train_per_member", cfg.dataset.n_train_per_member)
    plan = build_splits(
        [it.item_id for it in items],
        member_ids,
        policy=cond.train_data,
        n_eval=n_eval,
        n_train_per_member=n_train,
        seed=cfg.seed,
    )

    by_id = {it.item_id: it for it in items}
    eval_items = [by_id[i] for i in plan.eval_ids]

    lora_opts = dict(cfg.options.get("lora", {}) or {})
    specs: list[MemberSpec] = []
    for member_id, model_name, seed in zip(member_ids, cond.models, cond.adapter_seeds):
        model = cfg.models[model_name]
        train_ids = plan.train_by_member[member_id]
        train_items = [by_id[i] for i in train_ids]
        settings = LoRASettings(seed=int(seed), **lora_opts)

        # Per-member adapter directory: the protocol forbids sharing adapters
        # across families, and a shared output path is how that happens by
        # accident.
        safe_id = member_id.replace("/", "_").replace("#", "_")
        out_dir = Path(adapter_root) / cfg.name / condition / safe_id

        result = train_lora(
            checkpoint=model.checkpoint,
            output_dir=out_dir,
            train_items=train_items,
            labels=cfg.dataset.labels,
            prompt_version=cfg.dataset.prompt_version,
            settings=settings,
            member_id=member_id,
            family=model.family,
            lineage=model.lineage or model.family,
            adapter_seed=int(seed),
            dtype=model.dtype,
            prompt_extra={
                k: str(v) for k, v in (cfg.dataset.options.get("prompt_extra") or {}).items()
            },
            dry_run=dry_run_finetune,
        )
        specs.append(
            MemberSpec(
                member_id=member_id,
                model=model,
                condition=condition,
                adapter_seed=int(seed),
                train_split=(
                    # Shared slices must share a split *name* too: the simulator
                    # keys its data-shared error component on this string.
                    f"{condition}:shared"
                    if cond.train_data == "shared"
                    else f"{condition}:{safe_id}"
                ),
                adapter_path=result["adapter_path"],
                adapter_hash=result["adapter_hash"],
                train_item_ids=result["train_item_ids"],
            )
        )

    matrix = overlap_matrix({s.member_id: s.train_item_ids for s in specs})
    diag = {
        "splits": split_summary(plan),
        "overlap": overlap_report(matrix, cond.train_data),
        "adapter_hashes": {s.member_id: s.adapter_hash for s in specs},
    }
    # Distinct adapter hashes are the check that "five members" are five runs.
    hashes = [s.adapter_hash for s in specs]
    if len(set(hashes)) != len(hashes):
        diag["overlap"]["warnings"].append(  # type: ignore[index]
            "two members share an adapter hash: they were trained identically and "
            "are not independent ensemble members"
        )
    return specs, eval_items, diag


# -------------------------------------------------------------- inference ---
def run_inference(
    cfg: ExperimentConfig,
    condition: str,
    items: Sequence[AnchoredItem],
    specs: Sequence[MemberSpec],
    run: RunDir,
    *,
    unparsed_policy: str = "wrong",
) -> list[ItemResult]:
    rows, manifest = run_condition(
        cfg, condition, items, specs, unparsed_policy=unparsed_policy
    )
    write_predictions(run.predictions(condition), rows)
    write_manifest(run.manifest(condition), manifest)
    return rows


# --------------------------------------------------------------- analysis ---
def analyze(
    cfg: ExperimentConfig,
    matrices: Mapping[str, ErrorMatrix],
    items: Sequence[AnchoredItem],
    run: RunDir,
    *,
    homogeneous: str = "HS",
    diverse: str = "DD",
    run_localization_race: bool = True,
    extra_warnings: Sequence[str] = (),
) -> dict[str, Any]:
    """Compute every section 8-11 quantity and write the report.

    Reads only matrices, so it re-runs over saved predictions with no model
    calls -- which is the point of keeping the item-level rows.
    """
    warnings: list[str] = list(extra_warnings)
    out: dict[str, Any] = {}

    panel = panel_report(list(items))
    warnings.extend(clean_subset_warnings(panel))
    out["panel"] = panel

    out["conditions"] = {
        name: condition_metrics(m).to_dict() for name, m in matrices.items()
    }
    for name, m in matrices.items():
        spread = out["conditions"][name]["capability_spread"]["range"]
        if isinstance(spread, float) and spread > 0.10:
            warnings.append(
                f"{name}: standalone accuracy spans {spread:.1%} across members. "
                "Diversity can look useful merely because a weak member "
                "manufactures disagreement -- match capability or adjust "
                "statistically before reading Delta_div (protocol section 18)."
            )

    out["dependence"] = {name: dependence_report(m) for name, m in matrices.items()}

    # -- paired endpoint comparison -------------------------------------
    comparisons = []
    if homogeneous in matrices and diverse in matrices:
        a, b = align_matrices(matrices[homogeneous], matrices[diverse])
        n_boot = cfg.n_bootstrap

        diff = paired_bootstrap(
            a, b, clean_consensus_wrong,
            name=f"C_{homogeneous}^clean - C_{diverse}^clean",
            n_boot=n_boot, seed=cfg.seed,
        )
        clean_dd = bootstrap_metric(
            b, clean_consensus_wrong,
            name=f"C_{diverse}^clean", n_boot=n_boot, seed=cfg.seed, null_value=0.0,
        )
        clean_hs = bootstrap_metric(
            a, clean_consensus_wrong,
            name=f"C_{homogeneous}^clean", n_boot=n_boot, seed=cfg.seed,
        )
        ratio = paired_bootstrap(
            a, b, clean_consensus_wrong,
            name="Delta_div = C_H^clean / C_D^clean",
            n_boot=n_boot, seed=cfg.seed, null_value=1.0,
            combine=lambda x, y: diversity_gap(x, y),
        )
        or_res = odds_ratio(a, b, clean_consensus_wrong, n_boot=n_boot, seed=cfg.seed)
        d_diff = paired_bootstrap(
            a, b, disagreement_rate,
            name=f"D_{homogeneous} - D_{diverse}", n_boot=n_boot, seed=cfg.seed,
        )
        comparisons = [clean_hs, clean_dd, diff, ratio, or_res, d_diff]

        out["comparisons"] = [c.to_dict() for c in comparisons]
        out["diversity_gap"] = {
            "ratio": ratio.point,
            "ratio_ci": [ratio.ci_low, ratio.ci_high],
            "difference": diff.point,
            "difference_ci": [diff.ci_low, diff.ci_high],
            "odds_ratio": or_res.point,
            "odds_ratio_ci": [or_res.ci_low, or_res.ci_high],
        }
        out["error_conversion"] = error_conversion(a, b)
        write_json(
            run.bootstrap("endpoint_comparisons.json"),
            [c.to_dict(include_draws=True) for c in comparisons],
        )

    # -- survival curve on the diverse condition ------------------------
    survival: dict[str, Any] = {}
    if diverse in matrices:
        m = matrices[diverse]
        curve = residual_survival(m, clean_consensus_wrong, seed=cfg.seed)
        # The observed curve alone cannot separate "heading for zero" from
        # "heading for a floor" at k = 5, so it is always read against the
        # shared-difficulty model's own curve.
        excess = excess_survival(m, seed=cfg.seed)
        survival = survival_shape(curve, excess=excess, seed=cfg.seed)
        survival["contested"] = contested_transition(m, seed=cfg.seed)
        out["survival"] = survival
        plot_survival(
            run.figure("residual_survival.png"),
            {"C_k^clean": [p.to_dict() for p in curve]},
            title=f"Residual survival curve ({diverse})",
        )

    # -- localization race ----------------------------------------------
    race_out: dict[str, Any] = {}
    if run_localization_race and diverse in matrices:
        m = matrices[diverse]
        embeddings = _embeddings_for(run, diverse, m)
        race = run_race(m, embeddings=embeddings, seed=cfg.seed)
        race_out = race.to_dict()
        race_out["interpretation"] = race.interpret()
        out["race"] = race_out
        plot_race(run.figure("localization_race.png"), race_out)
        if embeddings is None:
            warnings.append(
                "no representation embeddings in the prediction rows, so the "
                "failure-memory strategy had no way to generalize from anchored "
                "history and scored uniformly. Set embedding_dim / store "
                "embeddings before reading P6."
            )

    # -- decision gate and preregistered predictions --------------------
    gate = None
    if diverse in matrices:
        m = matrices[diverse]
        clean_res = clean_consensus_wrong(m)
        ci = None
        for c in comparisons:
            if c.name == f"C_{diverse}^clean":
                ci = (c.ci_low, c.ci_high)
        gate = decision_gate(
            clean_residual=clean_res,
            survival_shape=survival,
            residual_ci=ci,
            n_clean=int(m.clean_mask.sum()),
            contested_share=contested_share_of_residual(m),
        )
        out["gate"] = gate.to_dict()

    clean_dd_res = next(
        (c for c in comparisons if c.name == f"C_{diverse}^clean"), None
    )
    diff_res = next(
        (c for c in comparisons if c.name.startswith(f"C_{homogeneous}^clean -")), None
    )
    predictions = evaluate_predictions(
        diff_hs_dd=diff_res,
        clean_dd=clean_dd_res,
        q_contested_hs=(
            contested_share_of_residual(matrices[homogeneous])
            if homogeneous in matrices else float("nan")
        ),
        q_contested_dd=(
            contested_share_of_residual(matrices[diverse])
            if diverse in matrices else float("nan")
        ),
        conversion=out.get("error_conversion"),
        race=race_out.get("interpretation"),
        gate=gate,
    )
    out["predictions"] = [p.to_dict() for p in predictions]
    out["warnings"] = warnings

    report = render_report(
        experiment=cfg.name,
        conditions=out["conditions"],
        panel=panel,
        comparisons=comparisons,
        survival=survival,
        dependence=out["dependence"],
        race=race_out,
        gate=gate,
        predictions=predictions,
        warnings=warnings,
    )
    run.report_path.write_text(report, encoding="utf-8")
    write_json(run.root / "analysis.json", out)
    return out


def _embeddings_for(run: RunDir, condition: str, matrix: ErrorMatrix) -> np.ndarray | None:
    """Mean per-item embedding across members, if any backend stored one."""
    path = run.predictions(condition)
    if not path.exists():
        return None
    sums: dict[str, list[np.ndarray]] = {}
    for row in read_predictions(path, strict=False):
        if row.embedding:
            sums.setdefault(row.item_id, []).append(np.asarray(row.embedding, dtype=float))
    if not sums:
        return None
    dim = len(next(iter(sums.values()))[0])
    out = np.zeros((matrix.n_items, dim))
    for i, item_id in enumerate(matrix.item_ids):
        vecs = sums.get(item_id)
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


# ------------------------------------------------------------------- main ---
def run_experiment(
    config_path: str | Path,
    *,
    output_root: Path | None = None,
    conditions: Sequence[str] | None = None,
    data_root: Path = Path("data"),
    tag: str | None = None,
    config_root: str | Path | None = None,
) -> PipelineResult:
    """Run every stage for one experiment config.

    `config_root` overrides where model and dataset references are resolved
    from. It defaults to the config file's grandparent, which is right for
    `configs/experiments/*.yaml`; a config written to a scratch directory needs
    it set explicitly.
    """
    cfg = load_experiment(config_path, config_root=config_root)
    root = Path(output_root or cfg.output_root) / run_id(tag or cfg.name)
    run = RunDir(root).create()
    write_yaml(run.config_path, resolved_dict(cfg))

    items = prepare_dataset(cfg, data_root=data_root)
    log.info("dataset %s: %d anchored items", cfg.dataset.name, len(items))

    wanted = list(conditions or cfg.conditions.keys())
    result = PipelineResult(run_dir=run, config=cfg, items=items)
    diagnostics: dict[str, Any] = {}

    for condition in wanted:
        result.warnings.extend(validate_panel(cfg, condition))
        specs, eval_items, diag = prepare_members(
            cfg,
            condition,
            items,
            dry_run_finetune=bool(cfg.options.get("dry_run_finetune", True)),
            adapter_root=cfg.options.get("adapter_root", "adapters"),
        )
        diagnostics[condition] = diag
        result.warnings.extend(diag["overlap"]["warnings"])

        rows = run_inference(
            cfg, condition, eval_items, specs, run,
            unparsed_policy=str(cfg.options.get("unparsed_policy", "wrong")),
        )
        result.rows[condition] = rows
        matrix = build_matrix(rows, condition)
        save_matrix(run.matrix(condition), matrix)
        result.matrices[condition] = matrix

    write_json(run.root / "diagnostics.json", diagnostics)

    eval_ids = {i for m in result.matrices.values() for i in m.item_ids}
    eval_items_all = [it for it in items if it.item_id in eval_ids]
    result.analysis = analyze(
        cfg,
        result.matrices,
        eval_items_all,
        run,
        run_localization_race=bool(cfg.options.get("race", True)),
        extra_warnings=result.warnings,
    )
    return result


def reanalyze(run_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    """Re-run the analysis over an existing run directory. No model calls.

    The config is read from the run directory itself, so a re-analysis uses the
    configuration the run actually used rather than whatever `configs/` says
    today.
    """
    run = RunDir(run_root)
    cfg = load_experiment(config_path or run.config_path, config_root="configs")
    conditions = run.available_conditions()
    if not conditions:
        raise FileNotFoundError(f"no prediction files under {run.root}/predictions")

    matrices: dict[str, ErrorMatrix] = {}
    items_by_id: dict[str, AnchoredItem] = {}
    for condition in conditions:
        rows = read_predictions(run.predictions(condition))
        matrices[condition] = build_matrix(rows, condition)
        for r in rows:
            if r.item_id not in items_by_id and r.anchor_label and r.anchor_status:
                items_by_id[r.item_id] = AnchoredItem(
                    item_id=r.item_id, x="", judgments=[],
                    reference=r.anchor_label, status=r.anchor_status,
                    agreement=float("nan"), n_annotators=0,
                )
    return analyze(cfg, matrices, list(items_by_id.values()), run)
