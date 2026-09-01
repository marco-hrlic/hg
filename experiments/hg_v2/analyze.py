#!/usr/bin/env python3
"""Re-run the HG-v2 analysis over a saved run directory. No model calls.

The payoff of storing episodes rather than summaries (plan section 2.2): a new
cost metric or a corrected calculation applies to an existing run in seconds,
and the task is never regenerated -- everything is recomputed from
`episodes/*.jsonl`.

Works for any phase: the run directory's own config names the task, and the
task selects the analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.grounding.config import load_grounding_config  # noqa: E402
from hg.grounding.episodes import GroundingRunDir  # noqa: E402
from hg.grounding import bounded_query, integrated, phase0, phase1, phase2  # noqa: E402

# Which analysis a run directory gets is decided by the task its config names,
# not by a flag the caller has to remember. A run directory carries its own
# resolved config, so this cannot drift from what the run actually did.
ANALYZERS = {
    "index": phase0.analyze_run_dir,
    "noisy_index": phase1.analyze_run_dir,
    "route_choice": phase2.analyze_run_dir,
    "bounded_query": bounded_query.analyze_run_dir,
    "route_choice_integrated": integrated.analyze_run_dir,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="results/<run_id>")
    parser.add_argument("--config", default=None, help="override the run's stored config")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    run = GroundingRunDir(args.run_dir)
    cfg = load_grounding_config(args.config or run.config_path)
    analyzer = ANALYZERS.get(cfg.task_kind)
    if analyzer is None:
        print(
            f"no analysis registered for task kind {cfg.task_kind!r}; "
            f"known: {sorted(ANALYZERS)}",
            file=sys.stderr,
        )
        return 2
    analysis = analyzer(args.run_dir, args.config)

    gates = analysis.get("gates")
    if gates:
        # E2 reports H1 and H2 together; a run fails if any *evaluated* gate
        # fails, and a gate that was not evaluated cannot fail the run.
        if args.print_json:
            print(json.dumps(analysis, indent=2, default=str))
        else:
            print(Path(args.run_dir) / "report.md")
            for gate in gates.values():
                if not gate.get("evaluated", True):
                    status = "not evaluated (simulated backend)"
                else:
                    status = "PASSED" if gate.get("passed") else "FAILED"
                print(f"gate {gate['gate']}: {status} — {gate['title']}")
                for f in gate.get("failures") or []:
                    print(f"  - {f}")
        failed = [
            g for g in gates.values() if g.get("evaluated", True) and not g.get("passed")
        ]
        return 1 if failed else 0

    gate = analysis.get("gate") or {}
    # A gate can be reported but not *evaluated*: phase 2's G2 is a claim about
    # a frozen off-the-shelf model, so a simulated run computes the comparison
    # without it opening or closing the gate. Printing "FAILED" and exiting
    # non-zero there would report a statement about a stand-in as a result.
    evaluated = bool(gate.get("evaluated", True))

    if args.print_json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(Path(args.run_dir) / "report.md")
        if evaluated:
            status = "PASSED" if gate.get("passed") else "FAILED"
        else:
            status = "not evaluated (simulated backend)"
        print(f"gate {gate.get('gate')}: {status} — {gate.get('title')}")
        for f in gate.get("failures") or []:
            print(f"  - {f}")

        mech = analysis.get("mechanism_check") or {}
        if mech.get("available"):
            print(
                f"mechanism check ({mech['challenger']} vs {mech['baseline']}): "
                f"{'detected' if mech['passed'] else 'NOT detected'}, "
                f"{mech['bits_saved']:+.4f} bits saved"
            )

    if not evaluated:
        # Fall back to the check a simulated run *can* fail.
        mech = analysis.get("mechanism_check") or {}
        return 0 if mech.get("passed", True) else 1
    return 0 if gate.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
