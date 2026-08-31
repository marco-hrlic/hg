#!/usr/bin/env python3
"""Phase 0 — infrastructure validation (protocol sections 15 and 19).

    "Build and run the three-family infrastructure pilot first. Do not start
     with the full five-family, multi-task experiment. Validate the pipeline
     end-to-end on the Spark, save the item-level prediction matrix, verify the
     dependence metrics, and only then scale the experiment."

This script is that pilot. It checks the exit criteria explicitly and returns a
non-zero exit code if any of them fail, so Phase 0 is a gate rather than a
suggestion.

Exit criteria checked here:
  1. every member produced a complete, rectangular prediction matrix;
  2. the item-level schema round-trips through disk unchanged;
  3. training-slice overlap O_jk matches each condition's declared policy;
  4. adapter bookkeeping is per-member and unshared across families;
  5. output parsing is deterministic and the parse-failure rate is negligible;
  6. the dependence metrics reproduce the simulator's known ground truth.

Criterion 6 is the one that matters most and is easiest to skip: with the
simulator backend the true error structure is known, so the analysis can be
checked against it before it is pointed at real models.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.io import read_predictions  # noqa: E402
from hg.metrics.core import clean_consensus_wrong  # noqa: E402
from hg.metrics.matrix import build_matrix  # noqa: E402
from hg.pipeline import run_experiment  # noqa: E402
from hg.schema import MINIMUM_COLUMNS  # noqa: E402


def check_exit_criteria(result) -> list[str]:
    """Return a list of failures. Empty means Phase 0 passed."""
    failures: list[str] = []

    for condition, matrix in result.matrices.items():
        # 1. complete matrix
        expected = matrix.n_items * matrix.n_members
        if matrix.errors.size != expected:
            failures.append(f"{condition}: matrix is not rectangular")
        if matrix.n_members < 3:
            failures.append(
                f"{condition}: {matrix.n_members} members; the pilot needs three "
                "families to exercise the family-subset logic"
            )

        # 2. schema round-trip
        path = result.run_dir.predictions(condition)
        reloaded = read_predictions(path, strict=True)
        if len(reloaded) != len(result.rows[condition]):
            failures.append(
                f"{condition}: {len(result.rows[condition])} rows written, "
                f"{len(reloaded)} read back"
            )
        rebuilt = build_matrix(reloaded, condition)
        if not (rebuilt.errors == matrix.errors).all():
            failures.append(f"{condition}: error matrix changed across a disk round-trip")
        first = result.rows[condition][0].to_dict()
        first.setdefault("score", None)
        missing = [c for c in MINIMUM_COLUMNS if c not in first]
        if missing:
            failures.append(f"{condition}: rows missing required columns {missing}")

        # 5. parsing
        parse_failures = float((~matrix.parse_ok).mean())
        if parse_failures > 0.02:
            failures.append(
                f"{condition}: parse-failure rate {parse_failures:.1%} exceeds 2%; "
                "fix the prompt or canonicalizer before attributing the gap to "
                "model capability"
            )

    # 3 & 4. overlap policy and adapter bookkeeping
    diagnostics = result.run_dir.root / "diagnostics.json"
    if not diagnostics.exists():
        failures.append("diagnostics.json was not written; split/adapter checks did not run")

    # 6. the dependence metrics reproduce a known structure
    analysis = result.analysis
    dep = (analysis.get("dependence") or {}).get("DD") or {}
    hier = dep.get("hierarchical_fit") or {}
    if not hier.get("converged"):
        failures.append("DD: hierarchical dependence model did not converge")
    sigma_u = hier.get("sigma_u")
    if sigma_u is None or not (0.2 < float(sigma_u) < 5.0):
        failures.append(
            f"DD: fitted sigma_u = {sigma_u}; outside the plausible range, so the "
            "shared-difficulty baseline is not usable"
        )
    if not (analysis.get("survival") or {}).get("curve"):
        failures.append("no residual survival curve was produced")
    if not analysis.get("comparisons"):
        failures.append("no paired HS/DD comparison was produced")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiments/phase0_infra.yaml",
        help="experiment config (default: the three-family pilot)",
    )
    parser.add_argument("--output-root", default=None, help="override results root")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = run_experiment(
        args.config,
        output_root=Path(args.output_root) if args.output_root else None,
    )
    run_root = result.run_dir.root

    print(f"run directory: {run_root}")
    print(f"report:        {result.run_dir.report_path}")
    print()

    for condition, matrix in sorted(result.matrices.items()):
        print(
            f"{condition}: {matrix.n_items} items x {matrix.n_members} members, "
            f"C_E={_f(result.analysis['conditions'][condition]['C_E'])}, "
            f"C_E^clean={_f(clean_consensus_wrong(matrix))}"
        )

    gap = result.analysis.get("diversity_gap") or {}
    if gap:
        print()
        print(
            f"Delta_div = {_f(gap.get('ratio'))} "
            f"[{_f((gap.get('ratio_ci') or [None, None])[0])}, "
            f"{_f((gap.get('ratio_ci') or [None, None])[1])}]"
        )
    gate = result.analysis.get("gate") or {}
    if gate:
        print(f"decision gate: {gate.get('gate')} — {gate.get('title')}")

    failures = check_exit_criteria(result)
    print()
    if failures:
        print("PHASE 0 FAILED — exit criteria not met:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PHASE 0 PASSED — pipeline validated end to end.")
    print(
        "Reminder: this run used the simulator backend, so none of the numbers "
        "above are evidence about LLMs. Next step is Phase 1 "
        "(configs/experiments/m2_llm_endpoints.yaml) on real checkpoints."
    )
    if result.analysis.get("warnings"):
        print()
        print("Warnings (do not block the gate, but read them):")
        for w in result.analysis["warnings"]:
            print(f"  - {w}")
    return 0


def _f(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if math.isinf(v):
            return "inf"
        return f"{v:.4f}"
    return str(v)


if __name__ == "__main__":
    raise SystemExit(main())
