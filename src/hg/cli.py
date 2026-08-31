"""Command-line entry point.

One CLI so a containerized run is `hg <stage> --config ...` and the stages stay
decoupled (protocol section 14). Every stage takes a config and writes into a
run directory; none of them depend on another stage's in-memory state.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="experiment config path")
    parser.add_argument("--config-root", default=None, help="where to resolve model/dataset refs")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("-v", "--verbose", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hg",
        description=(
            "M2-LLM: does clean shared error survive strong internal model "
            "diversity? Every stage is driven by a config file."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run every stage for an experiment config")
    _add_common(p_run)
    p_run.add_argument("--conditions", nargs="*", default=None)

    p_infer = sub.add_parser(
        "infer", help="run inference only (alias for `run`, kept for container stages)"
    )
    _add_common(p_infer)
    p_infer.add_argument("--conditions", nargs="*", default=None)

    p_an = sub.add_parser("analyze", help="re-analyze a saved run directory, no model calls")
    p_an.add_argument("--run-dir", required=True)
    p_an.add_argument("--config", default=None)
    p_an.add_argument("--print-json", action="store_true")
    p_an.add_argument("-v", "--verbose", action="store_true")

    p_ds = sub.add_parser("dataset", help="build the anchor panel and report its structure")
    _add_common(p_ds)

    p_val = sub.add_parser("validate", help="check a config without running anything")
    p_val.add_argument("--config", required=True)
    p_val.add_argument("--config-root", default=None)
    p_val.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.command in ("run", "infer"):
        return _cmd_run(args)
    if args.command == "analyze":
        return _cmd_analyze(args)
    if args.command == "dataset":
        return _cmd_dataset(args)
    if args.command == "validate":
        return _cmd_validate(args)
    parser.error(f"unknown command {args.command}")
    return 2


def _cmd_run(args) -> int:
    from .pipeline import run_experiment

    result = run_experiment(
        args.config,
        output_root=Path(args.output_root) if args.output_root else None,
        conditions=args.conditions,
        data_root=Path(args.data_root),
        config_root=args.config_root,
    )
    print(f"run directory: {result.run_dir.root}")
    print(f"report:        {result.run_dir.report_path}")
    gate = result.analysis.get("gate") or {}
    if gate:
        print(f"decision gate: {gate.get('gate')} — {gate.get('title')}")
    for w in result.analysis.get("warnings") or []:
        print(f"warning: {w}")
    return 0


def _cmd_analyze(args) -> int:
    from .pipeline import reanalyze

    analysis = reanalyze(args.run_dir, args.config)
    if args.print_json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(Path(args.run_dir) / "report.md")
        gate = analysis.get("gate") or {}
        if gate:
            print(f"decision gate: {gate.get('gate')} — {gate.get('title')}")
    return 0


def _cmd_dataset(args) -> int:
    from .anchors.agreement import clean_subset_warnings, panel_report
    from .config import load_experiment
    from .pipeline import prepare_dataset

    cfg = load_experiment(args.config, config_root=args.config_root)
    items = prepare_dataset(cfg, data_root=Path(args.data_root))
    report = panel_report(items)
    print(json.dumps(report, indent=2, default=str))
    for w in clean_subset_warnings(report):
        print(f"warning: {w}", file=sys.stderr)
    return 0


def _cmd_validate(args) -> int:
    from .config import load_experiment, validate_panel
    from .inference.prompts import get_template

    cfg = load_experiment(args.config, config_root=args.config_root)
    print(f"experiment: {cfg.name}")
    print(f"dataset:    {cfg.dataset.name} (task_family {cfg.dataset.task_family})")
    get_template(cfg.dataset.prompt_version)  # raises if unknown
    print(f"prompt:     {cfg.dataset.prompt_version}")

    problems = 0
    for name in cfg.conditions:
        models = cfg.member_models(name)
        lineages = sorted({m.lineage or m.family for m in models})
        print(f"  {name}: E={len(models)}, lineages={lineages}, "
              f"train_data={cfg.conditions[name].train_data}")
        for w in validate_panel(cfg, name):
            print(f"    warning: {w}")
            problems += 1

    if len(cfg.dataset.labels) < 3:
        print(
            "  note: binary label space -- predictions P5 and P6 are not testable "
            "here (a consensus failure has zero disagreement by construction)"
        )
    return 0 if problems == 0 else 0  # warnings do not fail validation


if __name__ == "__main__":
    raise SystemExit(main())
