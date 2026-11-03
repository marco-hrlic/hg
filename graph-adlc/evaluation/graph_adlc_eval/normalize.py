"""Pre-registered normalization (config.json:normalization). Uses scenario inputs only."""

from __future__ import annotations

from typing import Any

from graph_adlc.models import Graph

OBLIGATIONS = {"review_spec", "review_implementation", "rerun_test", "check_contract", "investigate_unknown"}


def artifact_owner(g: Graph) -> dict[str, str]:
    """artifact id -> realizing component, from the pre-change graph."""
    return {e.dst: e.src for e in g.live_edges() if e.relation == "realized_by"}


def changed_targets(delta: dict[str, Any], owner: dict[str, str]) -> set[str]:
    ids = set()
    for op in delta["operations"]:
        if op["op"] in ("update_node", "remove_node"):
            ids.add(op["id"])
        elif op["op"] == "add_node":
            ids.add(op["node"]["id"])
    return {owner.get(i, i) for i in ids}


def normalize(scenario_id: str, preds: list[dict[str, Any]], g: Graph, delta: dict[str, Any],
              equivalence: dict[str, str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Returns (scored records, excluded/erroneous records). Deterministic order."""
    owner = artifact_owner(g)
    changed = changed_targets(delta, owner)
    kept: dict[tuple[str, str, str], dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for p in preds:
        ob = equivalence.get(p["obligation"], p["obligation"])
        rec = {"scenario_id": scenario_id, **p, "obligation": ob, "raw_target_id": p["target_id"],
               "target_id": owner.get(p["target_id"], p["target_id"])}
        if ob not in OBLIGATIONS:
            excluded.append({**rec, "excluded_reason": "unmappable obligation"})
            continue
        if rec["target_id"] in changed:
            excluded.append({**rec, "excluded_reason": "target is a directly changed entity"})
            continue
        key = (scenario_id, rec["method"], rec["target_id"], ob)
        if key in kept:
            kept[key].setdefault("merged_from", []).append(rec["raw_target_id"])
            continue
        kept[key] = rec
    return sorted(kept.values(), key=lambda r: (r["target_id"], r["obligation"])), excluded
