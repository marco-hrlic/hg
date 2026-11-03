import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # evaluation/llm_graph
ROOT = HERE.parents[1]  # graph-adlc
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("GADLC_FIXED_TIME", "2026-11-03T00:00:00Z")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def scripted_run(tmp_path_factory):
    import run_replay
    out = tmp_path_factory.mktemp("scripted")
    assert run_replay.main(["--mode", "scripted", "--out", str(out)]) == 0
    return out
