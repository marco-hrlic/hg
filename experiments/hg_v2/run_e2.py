#!/usr/bin/env python3
"""HG-v2 E2/E3 — routing and query synthesis in one system on a frozen LM.

The improved plan's central experiment (sections 6 to 10). Eight arms crossing
a routing policy with a query source, so that the two contributions can be read
apart:

    B4  always ask + synthesized query      (what Phase 2 measured)
    B5  learned route + synthesized query   (the integrated HG-v2 arm)
    B6  learned route + oracle query        (isolates routing error)
    B7  oracle route + synthesized query    (isolates synthesis error)

Gate H1 is B5 against B4 at matched final distortion. Gate H2 is E3: at least
one minimality intervention must reduce response cost without losing soundness
or decision quality.

Every model behaviour is measured once per item and the arms are assembled from
that cache, so the arms are exactly item-paired and the lambda sweep is free.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.grounding.config import load_grounding_config  # noqa: E402
from hg.grounding.integrated import H1_BASELINE, H1_CHALLENGER, run_e2  # noqa: E402

DEFAULT_CONFIG = "configs/experiments/hg_v2/e2_integrated_sim.yaml"


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
    if cfg.task_kind != "route_choice_integrated":
        print(
            f"config {args.config} has task kind {cfg.task_kind!r}; "
            "run_e2.py needs 'route_choice_integrated'",
            file=sys.stderr,
        )
        return 2

    run, analysis, repro_problems = run_e2(
        cfg, output_root=Path(args.output_root) if args.output_root else None
    )
    gates = analysis["gates"]
    h1, h2 = gates["H1"], gates["H2"]

    print(f"run directory: {run.root}")
    print(f"report:        {run.report_path}")
    if analysis["backend_is_simulated"]:
        print()
        print(
            "NOTE: backend is the simulator, so this run does NOT evaluate "
            "gates H1 or H2. It checks that the metrics detect what they must."
        )
    print()

    # One representative price for the console; the report has the full sweep.
    lambdas = analysis["lambda_grid"]
    focus = lambdas[len(lambdas) // 2] if lambdas else None
    print(f"arms at lambda = {focus}:")
    header = (
        f"{'arm':<32} {'acc':>7} {'bits':>7} {'ask':>6} {'sound':>7} "
        f"{'waste':>7} {'cost':>8}"
    )
    print(header)
    print("-" * len(header))
    for a in analysis["arms"]:
        if a["lambda_bit"] != focus:
            continue
        q = a["query"]
        print(
            f"{a['arm']:<32} {a['accuracy']:>7.4f} {a['mean_external_bits']:>7.3f} "
            f"{q['ask_rate']:>6.3f} {q['soundness']:>7.3f} "
            f"{q['waste_bits_per_item']:>7.3f} {a['mean_total_cost']:>8.4f}"
        )

    print()
    print(f"gate H1 ({H1_CHALLENGER} vs {H1_BASELINE}):")
    for row in analysis["h1_routing_benefit"]:
        print(
            f"  lambda={row['lambda_bit']:<5.2f} bits saved "
            f"{row['bits_saved']:+.3f} [{row['bits_saved_ci'][0]:+.3f}, "
            f"{row['bits_saved_ci'][1]:+.3f}]  matched quality: "
            f"{row['equal_decision_quality']}  -> "
            f"{'WIN' if row['passed'] else 'no'}"
        )

    calibration = analysis.get("router_calibration") or {}
    if calibration.get("available"):
        p = calibration["p_wrong"]
        print()
        print(
            f"router: Brier {p['brier']:.4f}, ECE "
            f"{p['expected_calibration_error']:.4f}, mean p̂ "
            f"{p['mean_predicted']:.4f} vs observed {p['mean_observed']:.4f}"
        )

    e3 = analysis.get("minimality") or {}
    if e3.get("available") and e3.get("interventions"):
        print()
        print(f"E3 minimality (comparator {e3['baseline_arm']}, "
              f"{e3['baseline']['mean_external_bits']:.3f} bits, "
              f"soundness {e3['baseline']['soundness']:.3f}):")
        for row in e3["interventions"]:
            print(
                f"  {row['variant']:<16} {row['mean_external_bits']:>6.3f} bits "
                f"({row['bits_saved_vs_q0']:+.3f}), soundness "
                f"{row['soundness']:.3f} ({row['soundness_loss_vs_q0']:+.3f}) "
                f"-> {'IMPROVES' if row['improves'] else 'no'}"
            )

    failures = [f"reproducibility: {p}" for p in repro_problems]
    for gate in (h1, h2):
        if gate.get("evaluated") and not gate["passed"]:
            failures.extend(f"gate {gate['gate']}: {f}" for f in gate["failures"])

    print()
    if failures:
        print("E2 FAILED:")
        for f in failures[:15]:
            print(f"  - {f}")
        print()
        print(h1["verdict"])
        return 1

    print(
        "E2 CHECKS PASSED"
        + ("" if h1.get("evaluated") else " (gates not evaluated: simulated backend)")
        + "."
    )
    print()
    print(h1["verdict"])
    print()
    print(h2["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
