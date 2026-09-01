#!/usr/bin/env python3
"""HG-v2 Phase 0 — the exact synthetic complementarity experiment.

    "Before using an LLM, demonstrate that the experiment infrastructure
     reproduces exact known theoretical results."   (plan section 2.4)

The index problem: the source privately holds n i.i.d. fair bits, the AI
privately holds an index K, and the decision is J = X_K. So H(J | M_A) = 1 bit
exactly, and the whole question is what the source has to transmit:

    act                     0 bits, accuracy 1/2
    ask_generic             n bits, accuracy 1   -> Omega = n - 1
    ask_targeted            1 bit,  accuracy 1   -> Omega = 0
    ask_targeted_ablated    1 bit,  accuracy 1/2 -> cost paid, nothing resolved

This is trivial mathematically. That is the point: it validates that the new
abstractions correctly represent AI private state, external private state, the
decision variable, the query, the response cost, the final decision, the
episode artifact and the analysis pipeline -- before any of it is pointed at a
model (plan section 29).

Gate G0 is a stop, not a suggestion: this script exits non-zero if any declared
acceptance criterion fails.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.grounding.config import load_grounding_config  # noqa: E402
from hg.grounding.phase0 import run_phase0  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/hg_v2/phase0_index.yaml"


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
    run, analysis, repro_problems = run_phase0(
        cfg, output_root=Path(args.output_root) if args.output_root else None
    )

    print(f"run directory: {run.root}")
    print(f"report:        {run.report_path}")
    print()

    header = (
        f"{'n':>3}  {'policy':<22} {'accuracy':>8} {'bits':>7} {'Omega':>8} "
        f"{'dD':>8} {'dD/bit':>8}"
    )
    print(header)
    print("-" * len(header))
    for a in analysis["arms"]:
        omega = a["coordination_overhead"]
        eff = a["information_efficiency"]
        print(
            f"{a['n_bits']:>3}  {a['policy']:<22} {a['accuracy']:>8.4f} "
            f"{a['mean_external_bits']:>7.2f} "
            f"{(f'{omega:.2f}' if omega is not None else '—'):>8} "
            f"{a['delta_distortion']:>+8.4f} "
            f"{(f'{eff:.4f}' if eff is not None else '—'):>8}"
        )

    print()
    print("Targeted vs generic ask, at identical final distortion:")
    for c in analysis["compression"]:
        ratio = c["compression_ratio"]
        print(
            f"  n={c['n_bits']:>3}: {c['generic_bits']:.0f} bits -> "
            f"{c['targeted_bits']:.0f} bit"
            + (f"  ({ratio:.0f}x less external information)" if ratio else "  (—)")
        )

    gate = analysis["gate"]
    failures = list(gate["failures"])
    # Acceptance criterion 6 is about the artifacts, not the theory, so it is
    # reported separately from the gate's own checks.
    failures += [f"reproducibility: {p}" for p in repro_problems]

    print()
    if failures:
        print(f"PHASE 0 FAILED — {len(failures)} of {gate['n_checks']} checks:")
        for f in failures:
            print(f"  - {f}")
        print()
        print(
            "Gate G0 is a stop (plan section 27): do not proceed to Phase 1 "
            "until the simulator reproduces the theory."
        )
        return 1

    print(f"PHASE 0 PASSED — {gate['n_checks']} checks, plus artifact reproducibility.")
    print(
        "Reminder: no model was involved, so none of the numbers above are "
        "evidence about an LLM. They establish that the episode artifact and "
        "the analysis pipeline represent the grounding problem correctly."
    )
    print()
    print("Next step is a review of this result before Phase 1 (plan section 28).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
