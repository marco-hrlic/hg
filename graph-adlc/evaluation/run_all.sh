#!/usr/bin/env bash
# One command to reproduce every v0.2 evaluation output (run from anywhere).
set -euo pipefail
cd "$(dirname "$0")/.."            # graph-adlc/
export PYTHONPATH="evaluation${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests            # v0.1 regression suite (44 tests)
python -m graph_adlc_eval.run    --dataset evaluation/scenarios --out evaluation/out
python -m graph_adlc_eval.report --predictions evaluation/out/predictions.jsonl \
                                 --oracle evaluation/oracle --dataset evaluation/scenarios --out evaluation/out
python -m pytest -q evaluation/tests # evaluation regression gate
