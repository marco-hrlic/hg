#!/usr/bin/env python3
"""HG-v2 E1 — the full bounded-query frontier (improved plan sections 4 and 5).

Phase 0 measured two points of the index problem: r = 0 (generic escalation,
n bits) and r = log2 n (a targeted query, 1 bit). The v0.8 theorem describes
the whole curve,

    b*(n, r) = ceil(n / 2**r)

for deterministic fixed-length single-shot zero-error protocols, and this
script runs every cell of it. Each (n, r) is run twice: once at b* bits, which
must reach accuracy 1, and once at b* - 1, which must not. An identity that no
code path can violate is not a regression test, so the protocol is simulated
rather than asserted.

E1b adds the lossy relaxation at a set of tolerances, and reports the finite
converse, the asymptotic rate and what a constructed code actually achieved as
three separate columns -- per plan section 5, they are three different claims.

No model, no GPU. Gate H0 is a stop: if the infrastructure disagrees with the
theorem, nothing measured on top of it can be read.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.grounding.bounded_query import run_e1  # noqa: E402
from hg.grounding.config import load_grounding_config  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/hg_v2/e1_frontier.yaml"


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
    if cfg.task_kind != "bounded_query":
        print(
            f"config {args.config} has task kind {cfg.task_kind!r}; "
            "run_e1.py needs 'bounded_query'",
            file=sys.stderr,
        )
        return 2

    run, analysis, repro_problems = run_e1(
        cfg, output_root=Path(args.output_root) if args.output_root else None
    )
    gate = analysis["gate"]

    print(f"run directory: {run.root}")
    print(f"report:        {run.report_path}")
    print()

    header = f"{'n':>4} {'r':>3} {'fiber':>6} {'b*':>4} {'observed':>9} {'acc':>7} {'b*-1 acc':>9}"
    print(header)
    print("-" * len(header))
    for row in analysis["frontier"]:
        trunc = row["truncated_accuracy"]
        print(
            f"{row['n_bits']:>4} {row['query_bits']:>3} {row['fiber_size']:>6} "
            f"{row['predicted_response_bits']:>4} "
            f"{row['observed_response_bits']:>9.2f} {row['accuracy']:>7.4f} "
            + ("        —" if trunc is None else f"{trunc:>9.4f}")
        )

    lossy = analysis.get("lossy") or []
    if lossy:
        print()
        header = (
            f"{'n':>4} {'r':>3} {'eps':>5} {'achiev':>7} {'converse':>9} "
            f"{'asympt':>7} {'err':>7}  code"
        )
        print(header)
        print("-" * len(header))
        for row in lossy:
            print(
                f"{row['n_bits']:>4} {row['query_bits']:>3} {row['epsilon']:>5.2f} "
                f"{row['achievable_response_bits']:>7.2f} "
                f"{row['converse_bits']:>9.2f} {row['asymptotic_bits']:>7.2f} "
                f"{row['realized_error']:>7.4f}  {row['code_method']}"
            )

    failures = [f"reproducibility: {p}" for p in repro_problems]
    failures += [f"gate H0: {f}" for f in gate["failures"]]
    print()
    if failures:
        print("E1 FAILED:")
        for f in failures[:20]:
            print(f"  - {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        print()
        print(gate["verdict"])
        return 1

    print(f"E1 PASSED ({gate['n_checks']} checks).")
    print()
    print(gate["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
