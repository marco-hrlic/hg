"""Transparent, deterministic next-action policy (SPEC §4, "Minimal deterministic scheduler").

Returns a *ranked set* of suggestions with the score components spelled out. It
is not a claim of a globally optimal plan.
"""

from __future__ import annotations

from typing import Any

from .lifecycle import evaluate
from .models import Graph, Readiness, Relation, SuggestedAction, WorkStatus
from .readiness import frontier

DEFAULT_WEIGHTS: dict[str, float] = {
    "tier_reconcile": 4000.0,
    "tier_unblock": 3000.0,
    "tier_implement": 2000.0,
    "tier_refine": 1000.0,
    "per_stale_item": 10.0,
    "per_unlocked_leaf": 10.0,
    "conditional_penalty": 50.0,
    "per_criterion_penalty": 1.0,
    "per_uncertain_dependency_penalty": 5.0,
}


def _leaf_descendants(g: Graph, node_id: str) -> set[str]:
    out, stack, seen = set(), [node_id], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        if g.is_leaf_requirement(cur):
            out.add(cur)
        stack.extend(g.children_of(cur))
    return out


def suggest_actions(g: Graph, work_units: list[dict[str, Any]], evidence: list[dict[str, Any]],
                    weights: dict[str, float] | None = None, limit: int | None = None) -> list[SuggestedAction]:
    wt = {**DEFAULT_WEIGHTS, **(weights or {})}
    raw: list[tuple[int, float, str, list[str], str]] = []
    leaves = {r.requirement_id: r for r in frontier(g)}

    # Tier 1: reconcile invalidated work / evidence.
    for w in work_units:
        if w["status"] != WorkStatus.NEEDS_REVALIDATION:
            continue
        ev = evaluate(w, g, evidence)
        risk = len(ev["baseline_mismatches"]) + len(ev["stale_evidence"])
        raw.append((1, wt["tier_reconcile"] + wt["per_stale_item"] * risk, "reconcile",
                    [w["id"], *w["target_ids"]],
                    f"{w['id']} was produced against superseded revisions "
                    f"({len(ev['baseline_mismatches'])} baseline mismatches, {len(ev['stale_evidence'])} stale evidence)"))

    # Tier 2: resolve blocking questions, by count of leaf requirements they block.
    for q in g.live_nodes():
        if q.kind != "question" or q.data.get("state") != "open" or not q.data.get("blocking"):
            continue
        unlocked: set[str] = set()
        for e in g.out_edges(q.id, Relation.BLOCKS):
            unlocked |= {lid for lid in _leaf_descendants(g, e.dst)
                         if lid in leaves and leaves[lid].readiness == Readiness.BLOCKED}
        raw.append((2, wt["tier_unblock"] + wt["per_unlocked_leaf"] * len(unlocked), "resolve_question",
                    [q.id, *sorted(unlocked)],
                    f"open blocking question; currently blocks {len(unlocked)} leaf requirement(s) {sorted(unlocked)}"))

    # Tier 3: implement ready/conditional requirements without open or verified work.
    taken = {t for w in work_units if w["status"] != WorkStatus.ABANDONED for t in w["target_ids"]}
    for rid, r in sorted(leaves.items()):
        if not r.implementable or rid in taken:
            continue
        crit = len(g.nodes[rid].data.get("acceptance_criteria", []))
        uncertain = len([c for c in r.conditions if c.code != "open_non_blocking_question"])
        score = (wt["tier_implement"] - wt["per_criterion_penalty"] * crit
                 - wt["per_uncertain_dependency_penalty"] * uncertain
                 - (wt["conditional_penalty"] if r.readiness == Readiness.CONDITIONAL else 0))
        raw.append((3, score, "implement", [rid],
                    f"{r.readiness}; {crit} acceptance criteria; {uncertain} uncertain dependencies/assumptions"))

    # Tier 4: refine under-specified requirements (missing criteria, no open blocking question).
    for rid, r in sorted(leaves.items()):
        codes = {b.code for b in r.blockers}
        if "missing_acceptance_criteria" in codes and "open_blocking_question" not in codes:
            raw.append((4, wt["tier_refine"], "refine", [rid], "acceptance criteria missing"))
    for n in g.live_nodes():
        if n.kind in ("capability", "goal") and not g.children_of(n.id):
            raw.append((4, wt["tier_refine"] - 1, "refine", [n.id], f"{n.kind} has no refinements yet"))

    raw.sort(key=lambda t: (-t[1], t[0], t[2], t[3]))
    out = [SuggestedAction(rank=i + 1, tier=t, action=a, target_ids=ids, score=round(s, 3), rationale=why)
           for i, (t, s, a, ids, why) in enumerate(raw)]
    return out[:limit] if limit else out
