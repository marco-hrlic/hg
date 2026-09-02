#!/usr/bin/env python3
"""M0-R1 vanilla-control comparison (`plans/HG_Native_M0_R1_Experiment_Kickoff.md`).

Evaluates one matched baseline/HG checkpoint pair (same seed, same frozen
pilot config) and produces every kickoff-doc comparison that needs *both*
checkpoints at once: N4 raw/PCA-64/HG quotient probes, N1/N3 vanilla
confidence-proxy controls, and the N2 vanilla-router price-grid comparison.
`run_eval.py` remains the per-checkpoint gate-metric evaluator (N0/N1/N2/N3/
N4-HG-only/N5) and should still be run once per checkpoint alongside this.

Two disjoint held-out episode pools are drawn by seed alone (no shared
state): `--seed` is the paired evaluation pool both systems are scored on
(same episode IDs for both, kickoff section 19); `--fit-seed` (default
`--seed + 500`, so it never collides with a training seed 0-2 or the
default eval seed 12345) is a separate pool used *only* to fit PCA-64 and
the g_info/g_access logistic proxies -- never scored on, per section 4.2's
"do not fit PCA ... on the evaluation split."

Usage:
    python experiments/hg_native/run_eval_r1.py \
        --baseline-config experiments/hg_native/configs/pilot_baseline.yaml \
        --baseline-checkpoint results/.../baseline/checkpoints/final.pt \
        --hg-config experiments/hg_native/configs/pilot_hg.yaml \
        --hg-checkpoint results/.../hg/checkpoints/final.pt \
        --seed 1 --output-root results/m0_r1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg_native.data import accessibility, capability_refinement, sequential_quotient, static_hidden  # noqa: E402
from hg_native.data.tokenizer import BPETokenizer  # noqa: E402
from hg_native.eval.binary_metrics import auroc, summary  # noqa: E402
from hg_native.eval.bootstrap import paired_bootstrap_diff  # noqa: E402
from hg_native.eval.confidence import confidence_features, fit_logistic_proxy  # noqa: E402
from hg_native.eval.grounding import grounding_auroc_improvement, needs_grounding_labels  # noqa: E402
from hg_native.eval.latents import extract_hidden_states, extract_latents  # noqa: E402
from hg_native.eval.predict import predict  # noqa: E402
from hg_native.eval.projections import fit_pca  # noqa: E402
from hg_native.eval.quotient_probes import quotient_recovery  # noqa: E402
from hg_native.eval.reuse import repeated_ask_rate  # noqa: E402
from hg_native.eval.routing import route_value_report  # noqa: E402
from hg_native.eval.scaffolding import accessibility_labels_and_mask  # noqa: E402
from hg_native.eval.vanilla_router import vanilla_route_predictions  # noqa: E402
from hg_native.model.config import HGNativeConfig  # noqa: E402
from hg_native.model.hg_model import build_model  # noqa: E402
from hg_native.training.batch import ByteTokenizer  # noqa: E402
from hg_native.training.checkpoint import load_checkpoint  # noqa: E402

# Kickoff section 10's price grid.
_LAMBDAS = (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10)
PRICE_GRID = [(b, k) for b in _LAMBDAS for k in _LAMBDAS]


def held_out_episode_set(seed: int, n_per_family: int) -> list:
    """Same construction as `run_eval.py`'s -- kept as a separate copy since
    `run_eval.py` is a script, not an importable module, matching this repo's
    existing convention of each `experiments/hg_native/*.py` being
    self-contained."""
    episodes = []
    for subclass in static_hidden.SUBCLASSES:
        episodes += static_hidden.generate_batch(seed, subclass, n_per_family)
    for subclass in accessibility.SUBCLASSES:
        episodes += accessibility.generate_batch(seed, subclass, n_per_family)
    for traj in sequential_quotient.generate_batch(seed, n_per_family, n_steps=5):
        episodes += traj
    for pair in capability_refinement.generate_batch(seed, n_per_family):
        episodes += list(pair.values())
    return episodes


def _family(episodes: list, name: str) -> list:
    return [ep for ep in episodes if ep.family == name]


def _load(config_path: str, checkpoint_path: str, device: str):
    config = HGNativeConfig.from_yaml(config_path)
    model = build_model(config)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.to(device)
    return config, model


def _q_sel_metric(latents, episodes) -> float:
    return quotient_recovery(latents, list(episodes), seed=0)["q_sel"]


def n4_comparison(hg_model, baseline_model, tokenizer, eval_eps, fit_eps, max_seq_len, device, seed, n_boot) -> dict:
    sequential_eval = _family(eval_eps, "sequential_quotient")
    sequential_fit = _family(fit_eps, "sequential_quotient")
    if len(sequential_eval) < 10 or len(sequential_fit) < 10:
        return {"n": len(sequential_eval), "note": "too few sequential_quotient episodes for probes"}

    hg_latents = extract_latents(hg_model, tokenizer, sequential_eval, max_seq_len, device)
    baseline_raw_eval = extract_hidden_states(baseline_model, tokenizer, sequential_eval, max_seq_len, device)
    baseline_raw_fit = extract_hidden_states(baseline_model, tokenizer, sequential_fit, max_seq_len, device)

    pca = fit_pca(baseline_raw_fit, n_components=hg_latents.shape[1], seed=seed)
    baseline_pca_eval = pca.transform(baseline_raw_eval)

    q_hg = quotient_recovery(hg_latents, sequential_eval, seed=seed)
    q_raw = quotient_recovery(baseline_raw_eval, sequential_eval, seed=seed)
    q_pca = quotient_recovery(baseline_pca_eval, sequential_eval, seed=seed)

    delta_64 = paired_bootstrap_diff(
        (hg_latents, sequential_eval), (baseline_pca_eval, sequential_eval),
        metric=_q_sel_metric, n_boot=n_boot, seed=seed, name="delta_q_sel_64",
    )
    delta_raw = paired_bootstrap_diff(
        (hg_latents, sequential_eval), (baseline_raw_eval, sequential_eval),
        metric=_q_sel_metric, n_boot=n_boot, seed=seed, name="delta_q_sel_raw",
    )
    return {
        "n": len(sequential_eval),
        "q_sel_hg": q_hg,
        "q_sel_vanilla_raw": q_raw,
        "q_sel_vanilla_pca64": q_pca,
        "delta_q_sel_64": delta_64.to_dict(),
        "delta_q_sel_raw": delta_raw.to_dict(),
    }


def n1_comparison(hg_model, baseline_model, tokenizer, eval_eps, fit_eps, max_seq_len, device, seed, n_boot) -> dict:
    static_eval = _family(eval_eps, "static_hidden")
    static_fit = _family(fit_eps, "static_hidden")

    hg_preds = predict(hg_model, tokenizer, static_eval, max_seq_len, device)
    fit_features = confidence_features(baseline_model, tokenizer, static_fit, max_seq_len, device)
    eval_features = confidence_features(baseline_model, tokenizer, static_eval, max_seq_len, device)
    g_info = fit_logistic_proxy(fit_features, needs_grounding_labels(static_fit), seed=seed)
    vanilla_info_prob = g_info.predict_proba(eval_features)

    delta = grounding_auroc_improvement(
        static_eval, hg_preds.info_deficit_prob, vanilla_info_prob, n_boot=n_boot, seed=seed
    )
    return {
        "n": len(static_eval),
        "hg_auroc": summary(hg_preds.info_deficit_prob, needs_grounding_labels(static_eval)),
        "vanilla_confidence_auroc": summary(vanilla_info_prob, needs_grounding_labels(static_eval)),
        "delta_auroc_info": delta.to_dict(),
        "g_info": g_info,  # consumed by n2_comparison, stripped before writing to disk
    }


def n3_comparison(hg_model, baseline_model, tokenizer, eval_eps, fit_eps, max_seq_len, device, seed, n_boot) -> dict:
    access_eval = _family(eval_eps, "accessibility")
    access_fit = _family(fit_eps, "accessibility")

    hg_preds = predict(hg_model, tokenizer, access_eval, max_seq_len, device)
    fit_features = confidence_features(baseline_model, tokenizer, access_fit, max_seq_len, device)
    eval_features = confidence_features(baseline_model, tokenizer, access_eval, max_seq_len, device)
    fit_labels, fit_mask = accessibility_labels_and_mask(access_fit)
    g_access = fit_logistic_proxy(fit_features[fit_mask], fit_labels[fit_mask], seed=seed)
    vanilla_access_prob = g_access.predict_proba(eval_features)

    eval_labels, eval_mask = accessibility_labels_and_mask(access_eval)
    delta = paired_bootstrap_diff(
        (hg_preds.accessibility_prob[eval_mask], eval_labels[eval_mask]),
        (vanilla_access_prob[eval_mask], eval_labels[eval_mask]),
        metric=auroc, n_boot=n_boot, seed=seed, name="delta_auroc_access",
    )
    return {
        "n": int(eval_mask.sum()),
        "hg_auroc": summary(hg_preds.accessibility_prob[eval_mask], eval_labels[eval_mask]),
        "vanilla_confidence_auroc": summary(vanilla_access_prob[eval_mask], eval_labels[eval_mask]),
        "delta_auroc_access": delta.to_dict(),
        "g_access": g_access,  # consumed by n2_comparison, stripped before writing to disk
    }


def n2_comparison(hg_model, baseline_model, tokenizer, eval_eps, g_info, g_access, max_seq_len, device) -> dict:
    hg_preds = predict(hg_model, tokenizer, eval_eps, max_seq_len, device)
    features = confidence_features(baseline_model, tokenizer, eval_eps, max_seq_len, device)
    vanilla_info_prob = g_info.predict_proba(features)
    vanilla_access_prob = g_access.predict_proba(features)
    # One fixed lambda=1 routing decision, evaluated across the whole price
    # grid -- matches how the HG route is actually produced (`predict()`
    # calls the model with `c_t` defaulted to zero, so the HG router's
    # decision is likewise not re-made per grid cell; see module docstring).
    vanilla_route_pred = vanilla_route_predictions(eval_eps, vanilla_info_prob, vanilla_access_prob, 1.0, 1.0)

    hg_report = route_value_report(eval_eps, hg_preds.route_pred, PRICE_GRID)
    vanilla_report = route_value_report(eval_eps, vanilla_route_pred, PRICE_GRID)
    cells = []
    hg_wins = 0
    for hg_cell, vanilla_cell in zip(hg_report["grid"], vanilla_report["grid"]):
        delta = hg_cell["model"] - vanilla_cell["model"]
        hg_wins += delta < 0
        cells.append({"lambda_b": hg_cell["lambda_b"], "lambda_k": hg_cell["lambda_k"], "hg_cost": hg_cell["model"], "vanilla_cost": vanilla_cell["model"], "delta": delta})

    sequential_eval = _family(eval_eps, "sequential_quotient")
    sequential_idx = [i for i, ep in enumerate(eval_eps) if ep.family == "sequential_quotient"]
    return {
        "n_cells": len(cells),
        "hg_wins_fraction": hg_wins / len(cells) if cells else float("nan"),
        "grid": cells,
        "reuse_vanilla_repeated_ask_rate": repeated_ask_rate(sequential_eval, vanilla_route_pred[sequential_idx]) if sequential_eval else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--hg-config", required=True)
    parser.add_argument("--hg-checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=12345, help="Paired held-out eval seed (both systems scored on this pool)")
    parser.add_argument("--fit-seed", type=int, default=None, help="Default: --seed + 500 (must not collide with training/eval seeds)")
    parser.add_argument("--n-per-family", type=int, default=100)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-boot-qsel", type=int, default=200, help="Fewer draws for N4: each draw refits 5 probes from scratch")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None, help="Write comparison.json here; default results/m0_r1/seed<seed>/comparison.json")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    fit_seed = args.fit_seed if args.fit_seed is not None else args.seed + 500
    if fit_seed == args.seed:
        raise ValueError("--fit-seed must differ from --seed (probe-fit and eval pools must be disjoint)")

    tokenizer = BPETokenizer(args.tokenizer) if args.tokenizer else ByteTokenizer()
    _, hg_model = _load(args.hg_config, args.hg_checkpoint, device)
    _, baseline_model = _load(args.baseline_config, args.baseline_checkpoint, device)

    eval_eps = held_out_episode_set(args.seed, args.n_per_family)
    fit_eps = held_out_episode_set(fit_seed, args.n_per_family)

    n4 = n4_comparison(hg_model, baseline_model, tokenizer, eval_eps, fit_eps, args.max_seq_len, device, args.seed, args.n_boot_qsel)
    n1 = n1_comparison(hg_model, baseline_model, tokenizer, eval_eps, fit_eps, args.max_seq_len, device, args.seed, args.n_boot)
    n3 = n3_comparison(hg_model, baseline_model, tokenizer, eval_eps, fit_eps, args.max_seq_len, device, args.seed, args.n_boot)
    g_info = n1.pop("g_info")
    g_access = n3.pop("g_access")
    n2 = n2_comparison(hg_model, baseline_model, tokenizer, eval_eps, g_info, g_access, args.max_seq_len, device)

    comparison = {"seed": args.seed, "fit_seed": fit_seed, "n4_quotient_recovery": n4, "n1_grounding": n1, "n3_accessibility": n3, "n2_routing": n2}
    comparison = _json_safe(comparison)

    output_path = Path(args.output) if args.output else Path("results") / "m0_r1" / f"seed{args.seed}" / "comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    print(f"wrote {output_path}", file=sys.stderr)


def _json_safe(obj):
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
