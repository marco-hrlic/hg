#!/usr/bin/env python3
"""Re-run the analysis over a saved run directory. No model calls.

This is the payoff of storing item-level rows (protocol section 7): a new
dependence measure or a corrected metric can be applied to an existing
prediction matrix without touching a GPU.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg.pipeline import reanalyze  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="results/<run_id>")
    parser.add_argument("--config", default=None, help="override the run's stored config")
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    analysis = reanalyze(args.run_dir, args.config)

    if args.print_json:
        print(json.dumps(analysis, indent=2, default=str))
    else:
        print(Path(args.run_dir) / "report.md")
        gate = analysis.get("gate") or {}
        if gate:
            print(f"decision gate: {gate.get('gate')} — {gate.get('title')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
