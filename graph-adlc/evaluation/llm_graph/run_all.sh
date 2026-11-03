#!/usr/bin/env bash
# Deterministic v0.3 replay + scoring + report + tests (no network, no model). Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../.."            # graph-adlc/
export GADLC_FIXED_TIME="2026-11-03T00:00:00Z"
python evaluation/llm_graph/run_replay.py --mode scripted --out evaluation/llm_graph/out/scripted
python evaluation/llm_graph/score.py      --run evaluation/llm_graph/out/scripted
python evaluation/llm_graph/report.py
python -m pytest -q -p no:cacheprovider evaluation/llm_graph/tests   # includes L14: v0.1 (44) + v0.2 (18) suites
