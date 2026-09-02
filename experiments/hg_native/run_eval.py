#!/usr/bin/env python3
"""Evaluate a trained HG-variant checkpoint against spec section 10's metrics:
gates N0-N5 plus query efficiency (section 10.4, Family A only -- see
`eval/query_efficiency.py`'s docstring for why M0's structured action space
narrows soundness/minimality from the spec's original free-form-query
framing).

Usage:
    python experiments/hg_native/run_eval.py \
        --config experiments/hg_native/configs/debug_hg.yaml \
        --checkpoint results/.../checkpoints/final.pt \
        --n-per-family 200 --seed 12345
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
from hg_native.eval.capability import capability_floor  # noqa: E402
from hg_native.eval.grounding import grounding_discrimination  # noqa: E402
from hg_native.eval.latents import extract_latents  # noqa: E402
from hg_native.eval.predict import predict  # noqa: E402
from hg_native.eval.query_efficiency import query_efficiency  # noqa: E402
from hg_native.eval.quotient_probes import quotient_recovery  # noqa: E402
from hg_native.eval.refinement import refinement_report  # noqa: E402
from hg_native.eval.reuse import future_external_bits, repeated_ask_rate  # noqa: E402
from hg_native.eval.routing import route_accuracy, route_value_report  # noqa: E402
from hg_native.eval.scaffolding import accessibility_discrimination, route_rates_on_zero_deficit  # noqa: E402
from hg_native.model.config import HGNativeConfig  # noqa: E402
from hg_native.model.hg_model import build_model  # noqa: E402
from hg_native.training.batch import ByteTokenizer  # noqa: E402
from hg_native.training.checkpoint import load_checkpoint  # noqa: E402


def held_out_episode_set(seed: int, n_per_family: int) -> list:
    """A fixed, seed-isolated evaluation pool: a distinct seed from training
    (spec section 17's split-isolation requirement -- training in
    `run_train.py` draws from `sample_episodes`' own seed/index stream, so a
    disjoint `seed` here is enough given every generator's own
    seed-disjointness tests)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--n-per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345, help="Held-out eval seed, disjoint from any training seed")
    parser.add_argument("--output", default=None, help="Write metrics.json here; defaults to alongside --checkpoint")
    parser.add_argument(
        "--tokenizer", default=None,
        help="Path to the BPE tokenizer.json the checkpoint was trained with. Must match --config's vocab_size. Default: ByteTokenizer.",
    )
    parser.add_argument("--device", default=None, help="torch device. Default: cuda if available, else cpu.")
    args = parser.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = HGNativeConfig.from_yaml(args.config)
    model = build_model(config)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.to(device)
    tokenizer = BPETokenizer(args.tokenizer) if args.tokenizer else ByteTokenizer()
    if tokenizer.vocab_size != config.transformer.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size ({tokenizer.vocab_size}) does not match "
            f"transformer.vocab_size in {args.config} ({config.transformer.vocab_size})"
        )

    episodes = held_out_episode_set(args.seed, args.n_per_family)
    preds = predict(model, tokenizer, episodes, max_seq_len=args.max_seq_len, device=device)

    static_hidden_eps = [ep for ep in episodes if ep.family == "static_hidden"]
    static_hidden_idx = [i for i, ep in enumerate(episodes) if ep.family == "static_hidden"]
    accessibility_eps = [ep for ep in episodes if ep.family == "accessibility"]
    accessibility_idx = [i for i, ep in enumerate(episodes) if ep.family == "accessibility"]
    sequential_eps = [ep for ep in episodes if ep.family == "sequential_quotient"]
    sequential_idx = [i for i, ep in enumerate(episodes) if ep.family == "sequential_quotient"]

    metrics = {
        "n_episodes": len(episodes),
        "N0_capability_floor": capability_floor(episodes, preds.decision_pred),
        "N1_grounding_discrimination": grounding_discrimination(static_hidden_eps, preds.info_deficit_prob[static_hidden_idx]),
        "N2_routing": {
            "route_accuracy_lambda_1_1": route_accuracy(episodes, preds.route_pred),
            "route_value_report": route_value_report(episodes, preds.route_pred, lambda_grid=[(0.1, 1.0), (1.0, 1.0), (10.0, 1.0)]),
        },
        "query_efficiency_static_hidden": query_efficiency(static_hidden_eps, preds.distortion_hat[static_hidden_idx]),
        "N3_accessibility_discrimination": accessibility_discrimination(accessibility_eps, preds.accessibility_prob[accessibility_idx]),
        "N3_route_rates_on_zero_deficit": route_rates_on_zero_deficit(accessibility_eps, preds.route_pred[accessibility_idx]),
        "N5_grounding_reuse": {
            "repeated_ask_rate": repeated_ask_rate(sequential_eps, preds.route_pred[sequential_idx]),
            "future_external_bits": future_external_bits(sequential_eps, preds.route_pred[sequential_idx]),
        },
    }

    capability_refinement_eps = [ep for ep in episodes if ep.family == "capability_refinement"]

    if len(sequential_eps) >= 10 and config.variant == "hg":
        latents = extract_latents(model, tokenizer, sequential_eps, max_seq_len=args.max_seq_len, device=device)
        metrics["N4_quotient_recovery"] = quotient_recovery(latents, sequential_eps, seed=args.seed)

    if len(capability_refinement_eps) >= 10 and config.variant == "hg":
        # M1 kickoff section 8.3/16 Stage M1-0: verify Delta_refine's
        # plumbing on a real trained checkpoint (still a smoke check at
        # debug scale, not a scientific claim -- see the kickoff's own
        # "no scientific claims" instruction for this stage).
        cap_latents = extract_latents(model, tokenizer, capability_refinement_eps, max_seq_len=args.max_seq_len, device=device)
        metrics["M1_capability_refinement"] = refinement_report(
            cap_latents, capability_refinement_eps, refining_var="v", control_var="u", seed=args.seed
        )

    metrics = _json_safe(metrics)
    output_path = Path(args.output) if args.output else Path(args.checkpoint).parent.parent / "eval" / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metrics, indent=2, default=str))
    print(f"wrote {output_path}", file=sys.stderr)


def _json_safe(obj):
    """Recursively replace NaN/Infinity with None: `json.dumps`'s default
    `allow_nan=True` emits bare `NaN`/`Infinity` tokens, which are not valid
    JSON and break strict parsers reading these files back -- and several
    of these metrics are legitimately, meaningfully NaN (e.g. AUROC on a
    single-class subclass), not a bug to hide."""
    if isinstance(obj, float):
        return None if (obj != obj or obj in (float("inf"), float("-inf"))) else obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


if __name__ == "__main__":
    main()
