#!/usr/bin/env python3
"""Run the analysis-validation control pair and check it discriminates.

Runs `configs/experiments/sim_discrimination.yaml` twice -- once with a real
information-limited core, once without -- and asserts that the analysis reaches
different verdicts. Simulator only: no GPU, and nothing it prints is evidence
about LLMs.

This is the check that makes a later gate C claim credible. If the analysis
cannot separate a simulated core from simulated shared difficulty, it cannot be
trusted to separate them in real data either.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from hg.pipeline import run_experiment  # noqa: E402

BASE_CONFIG = "configs/experiments/sim_discrimination.yaml"


def run_arm(base: dict, core_rate: float, work_dir: Path) -> dict:
    cfg = copy.deepcopy(base)
    cfg["name"] = f"sim_discrimination_core{core_rate}"
    cfg["dataset"]["name"] = f"sim_discrimination_core{core_rate}"
    cfg["dataset"]["anchors"]["spec"]["core_rate"] = core_rate
    path = work_dir / f"arm_core{core_rate}.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = run_experiment(
        path,
        config_root="configs",
        data_root=work_dir / "data",
        output_root=work_dir / "results",
    )
    return result.analysis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=BASE_CONFIG)
    parser.add_argument(
        "--work-dir", default="results/sim_discrimination",
        help="where the two arms' runs are written",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    arms = {}
    for core_rate in (0.02, 0.0):
        analysis = run_arm(base, core_rate, work_dir)
        survival = analysis.get("survival") or {}
        ex = survival.get("excess_survival") or {}
        arms[core_rate] = {
            "shape": survival.get("shape"),
            "gate": (analysis.get("gate") or {}).get("gate"),
            "endpoint": survival.get("endpoint"),
            "ratio": ex.get("ratio"),
            "ratio_growth": ex.get("ratio_growth"),
            "excess_verdict": survival.get("excess_verdict"),
        }

    print("Analysis-validation control (simulator only; not evidence about LLMs)")
    print()
    for core_rate, a in arms.items():
        truth = "a real core exists" if core_rate else "no core exists"
        print(f"core_rate = {core_rate}  ({truth})")
        print(f"  survival shape     : {a['shape']}  -> gate {a['gate']}")
        print(f"  C_5^clean          : {_f(a['endpoint'])}")
        print(f"  observed/predicted : {[round(r, 3) for r in (a['ratio'] or [])]}")
        print(f"  ratio growth       : {_f(a['ratio_growth'])}")
        print(f"  excess verdict     : {a['excess_verdict']}")
        print()

    with_core = arms[0.02]
    without = arms[0.0]
    failures: list[str] = []
    if with_core["excess_verdict"] != "residual_exceeds_shared_difficulty":
        failures.append(
            "with a real core, the analysis did not detect a residual exceeding "
            f"shared difficulty (got {with_core['excess_verdict']!r})"
        )
    if without["excess_verdict"] != "consistent_with_shared_difficulty":
        failures.append(
            "with no core, the analysis reported a residual beyond shared "
            f"difficulty (got {without['excess_verdict']!r}) -- a false positive "
            "in the direction that would wrongly justify the grounding architecture"
        )
    if with_core["shape"] == "apparent_plateau" and without["shape"] == "apparent_plateau":
        failures.append(
            "the plateau test alone reported a plateau in both arms; it cannot be "
            "the sole basis for gate C"
        )

    if failures:
        print("CONTROL FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("CONTROL PASSED — the analysis separates a simulated core from")
    print("simulated shared difficulty. Note that the survival *shape* test does")
    print("not: both arms are 'still_falling' at E = 5. The discriminating signal")
    print("is the observed-vs-model ratio growth, so read `excess_verdict`")
    print("alongside the curve in any real run.")
    return 0


def _f(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    raise SystemExit(main())
