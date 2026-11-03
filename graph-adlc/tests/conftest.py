import copy
import json
from pathlib import Path

import pytest

from graph_adlc import GraphADLC

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def load(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    monkeypatch.setenv("GADLC_FIXED_TIME", "2026-01-01T00:00:00Z")


@pytest.fixture
def seed() -> dict:
    return load("payment_seed.json")


@pytest.fixture
def app(tmp_path, seed) -> GraphADLC:
    a = GraphADLC(str(tmp_path / "graph.sqlite"))
    a.seed_graph(seed)
    yield a
    a.close()


@pytest.fixture
def refined(app) -> GraphADLC:
    """Graph after committing the retry refinement (graph rev 2)."""
    app.commit(load("retry_refinement.json"))
    return app


@pytest.fixture
def verified_retry(refined) -> tuple[GraphADLC, str]:
    """REQ-RETRY implemented and verified against graph rev 2 / CON-IDEMPOTENCY r1."""
    w = refined.create_work_unit("REQ-RETRY")
    refined.start_work_unit(w["id"])
    out = refined.submit_work_unit(w["id"], "git:abc123", [{"node_id": "TEST-RETRY", "result": "passed"}])
    assert out["status"] == "verified"
    return refined, w["id"]


def delta(base: int, *ops: dict, actor: str = "test", rationale: str = "test", **extra) -> dict:
    return {"base_graph_revision": base, "actor": actor, "rationale": rationale,
            "operations": [copy.deepcopy(o) for o in ops], **extra}


def by_target(records) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for r in records:
        tid = r["target_id"] if isinstance(r, dict) else r.target_id
        ob = r["obligation"] if isinstance(r, dict) else r.obligation
        out.setdefault(tid, set()).add(ob)
    return out
