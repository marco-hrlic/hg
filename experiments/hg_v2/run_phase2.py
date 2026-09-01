#!/usr/bin/env python3
"""HG-v2 Phase 2 — a frozen primary LM and the query-compression claim.

Plan sections 12 to 17. The route-choice task withholds one decision-relevant
variable and hands it to a simulated source. Five baselines, of which two
matter against each other:

    B2  generic ask-human escalation   -- the source dumps every preference
    B4  a synthesized targeted query   -- the source answers what was asked

    "At equal final distortion, does HG-v2 reduce external response cost?"

Gate G2 is that conditional comparison, and it is only meaningful once
`backend` names a real model. With the simulated backend this script validates
that the task, the query representation, the cost accounting and the
query-quality metrics can detect the effect when it is present -- the
precondition for the test, not the test.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.grounding.config import load_grounding_config  # noqa: E402
from hg.grounding.phase2 import G2_BASELINE, G2_CHALLENGER, run_phase2  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/hg_v2/phase2_frozen_lm.yaml"


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
    if cfg.task_kind != "route_choice":
        print(
            f"config {args.config} has task kind {cfg.task_kind!r}; "
            "run_phase2.py needs 'route_choice'",
            file=sys.stderr,
        )
        return 2

    run, analysis, repro_problems = run_phase2(
        cfg, output_root=Path(args.output_root) if args.output_root else None
    )
    gate = analysis["gate"]

    print(f"run directory: {run.root}")
    print(f"report:        {run.report_path}")
    if gate.get("backend_is_simulated"):
        print()
        print(
            "NOTE: backend is the simulator, so this run does NOT evaluate gate "
            "G2. It checks that the metrics can detect query compression."
        )
    print()

    header = (
        f"{'baseline':<22} {'accuracy':>8} {'loss':>7} {'bits':>6} "
        f"{'resolve':>8} {'excess':>8} {'gold':>6}"
    )
    print(header)
    print("-" * len(header))
    for a in analysis["arms"]:
        gold = a["gold_match_rate"]
        print(
            f"{a['baseline']:<22} {a['accuracy']:>8.4f} "
            f"{a['mean_final_loss']:>7.4f} {a['mean_external_bits']:>6.2f} "
            f"{a['resolution_accuracy']:>8.4f} "
            f"{a['mean_excess_response_cost']:>+8.4f} "
            f"{(f'{gold:.2f}' if gold is not None else '—'):>6}"
        )

    check = analysis.get("internal_compute_check") or {}
    if check.get("available"):
        print()
        print(check["note"])

    mech = analysis.get("mechanism_check") or {}
    if mech.get("available"):
        print()
        print(
            f"mechanism ({mech['challenger']} vs {mech['baseline']}): "
            f"{mech['bits_saved']:+.4f} bits "
            f"[{mech['bits_saved_ci'][0]:+.4f}, {mech['bits_saved_ci'][1]:+.4f}]"
            + (
                f", {mech['compression_ratio']:.2f}x"
                if mech.get("compression_ratio")
                else ""
            )
            + f", equal quality: {mech['equal_decision_quality']}"
        )

    print(
        f"gate G2 ({G2_CHALLENGER} vs {G2_BASELINE}): "
        f"{gate['bits_saved']:+.4f} bits "
        f"[{gate['bits_saved_ci'][0]:+.4f}, {gate['bits_saved_ci'][1]:+.4f}], "
        f"equal quality: {gate['equal_decision_quality']} "
        f"(loss gap {gate['distortion_gap']:+.4f})"
    )

    # What makes the *run* fail is separated from what makes the *gate* fail.
    # On a simulated backend G2 is explicitly not evaluable, so gating the exit
    # code on it would report a statement about a stand-in synthesizer as if it
    # were a result about the architecture.
    pipeline_failures = [f"reproducibility: {p}" for p in repro_problems]
    if mech.get("available") and not mech["passed"]:
        pipeline_failures.append(
            "mechanism check: the metrics did not detect compression even for "
            "the oracle query -- " + " ".join(mech["failures"])
        )
    if gate.get("evaluated") and not gate["passed"]:
        pipeline_failures.extend(f"gate G2: {f}" for f in gate["failures"])

    print()
    if pipeline_failures:
        print("PHASE 2 FAILED:")
        for f in pipeline_failures:
            print(f"  - {f}")
        print()
        print(gate["verdict"])
        return 1

    print(
        "PHASE 2 CHECKS PASSED"
        + ("" if gate.get("evaluated") else " (gate G2 not evaluated: simulated backend)")
        + "."
    )
    print()
    print(gate["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
