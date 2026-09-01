#!/usr/bin/env python3
"""Run an HG-Native training stage (spec sections 8, 19): smoke or pilot.

Usage:
    python experiments/hg_native/run_train.py --config configs/experiments/hg_native/smoke_hg.yaml

Writes a spec-section-16 run directory under `--output-root` (default
`results/`): `manifest.json`, `config.yaml`, `train_metrics.jsonl`,
`checkpoints/final.pt`, `report.md`. Deliberately does not write
`eval/*` (checklist items 13-18, the evaluator suite, are a later slice) or
`git.json`/`tokenizer_hash.json` as separate files -- both live inside
`manifest.json` here instead of as siblings, since nothing else in this repo
reads them from a split-out file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from hg.io import ensure_dir, run_id  # noqa: E402
from hg_native.model.config import HGNativeConfig  # noqa: E402
from hg_native.model.hg_model import build_model  # noqa: E402
from hg_native.training.batch import ByteTokenizer, tokenizer_hash  # noqa: E402
from hg_native.training.checkpoint import save_checkpoint  # noqa: E402
from hg_native.training.dataset import SyntheticMix  # noqa: E402
from hg_native.training.manifest import build_manifest, write_manifest  # noqa: E402
from hg_native.training.trainer import OptimConfig, TrainConfig, build_optimizer, train  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="HGNativeConfig YAML (model architecture)")
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--experiment-name", default="hg_native_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = HGNativeConfig.from_yaml(args.config)
    model = build_model(config)
    tokenizer = ByteTokenizer()

    train_cfg = TrainConfig(
        total_steps=args.total_steps, batch_size=args.batch_size, max_seq_len=args.max_seq_len,
        seed=args.seed, mix=SyntheticMix(),
    )
    optim_cfg = OptimConfig(lr=args.lr)

    print(f"variant={config.variant} params={model.num_parameters():,} steps={args.total_steps}", file=sys.stderr)
    history = train(model, config, train_cfg, optim_cfg=optim_cfg, tokenizer=tokenizer)

    run_dir = ensure_dir(Path(args.output_root) / run_id(args.experiment_name, config.variant))
    (run_dir / "config.yaml").write_text(yaml.safe_dump(_config_to_dict(config)), encoding="utf-8")
    with (run_dir / "train_metrics.jsonl").open("w", encoding="utf-8") as fh:
        for m in history:
            fh.write(json.dumps(asdict(m)) + "\n")

    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    optimizer = build_optimizer(model, optim_cfg)  # fresh handle just for the state-dict shape; weights are model's
    save_checkpoint(checkpoints_dir / "final.pt", model, optimizer, step=args.total_steps)

    manifest = build_manifest(
        run_id=run_dir.name,
        seed=args.seed,
        config_dict=_config_to_dict(config),
        tokenizer_hash=tokenizer_hash(tokenizer),
        model_parameter_count=model.num_parameters(),
        extra={"total_steps": args.total_steps, "batch_size": args.batch_size},
    )
    write_manifest(run_dir / "manifest.json", manifest)

    first, last = history[0], history[-1]
    report = (
        f"# HG-Native training run: {run_dir.name}\n\n"
        f"- variant: {config.variant}\n"
        f"- parameters: {model.num_parameters():,}\n"
        f"- steps: {args.total_steps}, batch_size: {args.batch_size}\n"
        f"- first-step total loss: {first.total:.4f}\n"
        f"- last-step total loss: {last.total:.4f}\n"
        f"- final lm loss: {last.losses['lm']:.4f}\n"
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    print(report, file=sys.stderr)
    print(f"wrote {run_dir}", file=sys.stderr)


def _config_to_dict(config: HGNativeConfig) -> dict:
    from dataclasses import asdict as _asdict

    return _asdict(config)


if __name__ == "__main__":
    main()
