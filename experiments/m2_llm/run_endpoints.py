#!/usr/bin/env python3
"""Phase 1 — endpoint M2-LLM: HS then DD (protocol sections 4 and 15).

    "Run the two endpoints first. If the effect is informative, fill in HD and
     DS to decompose the source of the change."

So this script runs exactly the two endpoints and then says whether expanding
is warranted, rather than quietly proceeding to Phase 2. The recommendation is
based on the diversity gap and on whether a clean residual survives DD -- the
two things Phase 2 and Phase 4 respectively depend on.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.pipeline import run_experiment  # noqa: E402


def recommend_next(analysis: dict) -> list[str]:
    """Translate the endpoint result into the protocol's next step."""
    lines: list[str] = []
    gap = analysis.get("diversity_gap") or {}
    gate = analysis.get("gate") or {}
    diff_ci = gap.get("difference_ci") or [float("nan"), float("nan")]

    informative = bool(diff_ci[0] > 0)
    if informative:
        lines.append(
            "The HS -> DD gap excludes zero: the effect is informative. "
            "Proceed to Phase 2 (configs/experiments/m2_llm_decomposition.yaml) "
            "to split it into data-overlap and model-family components."
        )
    else:
        lines.append(
            "The HS -> DD gap does not exclude zero. Decomposing a gap that is "
            "not there would spend GPU time on noise -- check panel size, "
            "capability matching and parse rates before expanding."
        )

    g = gate.get("gate")
    if g == "C":
        lines.append(
            "Gate C: a clean shared residual survives DD. Phase 4 (localization "
            "race) is now the informative next experiment; it is not informative "
            "at gates A or B."
        )
    elif g == "B":
        lines.append(
            "Gate B: the residual is nonzero but still falling. Extend the "
            "diversity axis before treating the endpoint as a floor."
        )
    elif g == "A":
        lines.append(
            "Gate A: no clean residual to target. Do not build an anchored "
            "failure-memory system around a proxy that disappeared under better "
            "internals."
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/m2_llm_endpoints.yaml")
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--conditions", nargs="*", default=["HS", "DD"],
        help="conditions to run, in order (default: the two endpoints)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    result = run_experiment(
        args.config,
        output_root=Path(args.output_root) if args.output_root else None,
        conditions=args.conditions,
    )

    print(f"run directory: {result.run_dir.root}")
    print(f"report:        {result.run_dir.report_path}")
    print()
    for name, m in sorted(result.analysis["conditions"].items()):
        print(
            f"{name}: C_E={_f(m['C_E'])}  C_E^clean={_f(m['C_E_clean'])}  "
            f"D_E={_f(m['D_E'])}  Q_contested={_f(m['Q_contested'])}"
        )
    print()
    for c in result.analysis.get("comparisons") or []:
        print(f"{c['name']}: {_f(c['point'])} [{_f(c['ci_low'])}, {_f(c['ci_high'])}]")
    print()
    for line in recommend_next(result.analysis):
        print(f"* {line}")
    return 0


def _f(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    raise SystemExit(main())
