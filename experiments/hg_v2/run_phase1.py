#!/usr/bin/env python3
"""HG-v2 Phase 1 — learned routing on a controlled task (plan sections 9-11).

Phase 0's task always needed the source, so routing was trivial. Here the AI
sometimes already holds enough to decide, the optimal route varies per item and
is known from the counterfactuals, and the question is whether a small router
trained on an internal signal can find it.

    C_r = L_r + lambda*B_r,   r* = argmin_r C_r

Gate G1 is specific: learned routing must beat **both** always-ACT and
always-ASK over a nontrivial range of external-information prices (plan section
11). This script exits non-zero if it does not.

No RL (plan section 9): a logistic regression on internal-agreement features,
fitted on a training split and scored out of sample.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.grounding.config import load_grounding_config  # noqa: E402
from hg.grounding.phase1 import run_phase1  # noqa: E402
from hg.routing.grounding_router import FIXED_POLICIES  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/hg_v2/phase1_router.yaml"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="experiment config")
    parser.add_argument("--output-root", default=None, help="override results root")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_grounding_config(args.config)
    if cfg.task_kind != "noisy_index":
        print(
            f"config {args.config} has task kind {cfg.task_kind!r}; "
            "run_phase1.py needs 'noisy_index'",
            file=sys.stderr,
        )
        return 2

    run, analysis, repro_problems = run_phase1(
        cfg, output_root=Path(args.output_root) if args.output_root else None
    )

    print(f"run directory: {run.root}")
    print(f"report:        {run.report_path}")
    print()

    header = (
        f"{'lambda':>7}  {'policy':<12} {'C=L+lB':>8} {'route_acc':>10} "
        f"{'loss':>7} {'ask_rate':>9} {'regret':>8}"
    )
    print(header)
    print("-" * len(header))
    last_lambda = None
    for a in analysis["arms"]:
        if last_lambda is not None and a["lambda_bit"] != last_lambda:
            print()
        last_lambda = a["lambda_bit"]
        print(
            f"{a['lambda_bit']:>7.2f}  {a['policy']:<12} "
            f"{a['mean_total_cost']:>8.4f} {a['route_accuracy']:>10.4f} "
            f"{a['mean_final_loss']:>7.4f} {a['external_query_rate']:>9.4f} "
            f"{a['routing_regret']:>8.4f}"
        )

    print()
    print("Learned routing vs both fixed policies (C_baseline - C_learned, item-paired):")
    for c in analysis["comparisons"]:
        parts = []
        for baseline in FIXED_POLICIES:
            b = c["beats"].get(baseline)
            if b is None:
                continue
            mark = "win " if b["significant"] else "lose"
            parts.append(
                f"{baseline}: {mark} {b['advantage']:+.4f} "
                f"[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]"
            )
        verdict = "BEATS BOTH" if c["beats_both"] else ""
        print(f"  lambda={c['lambda_bit']:.2f}  " + "   ".join(parts) + f"   {verdict}")

    gate = analysis["gate"]
    failures = list(gate["failures"]) + [
        f"reproducibility: {p}" for p in repro_problems
    ]

    print()
    if failures:
        print("PHASE 1 FAILED:")
        for f in failures:
            print(f"  - {f}")
        print()
        print(gate["verdict"])
        return 1

    print(
        f"PHASE 1 PASSED — gate G1 met at {len(gate['winning_lambdas'])} of "
        f"{gate['n_lambdas']} prices (lambda = {gate['winning_lambdas']}), "
        "plus artifact reproducibility."
    )
    print()
    print(gate["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
