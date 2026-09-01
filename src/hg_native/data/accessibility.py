"""Family B — Accessibility / Scaffolding generator (spec section 5.2).

Purpose: separate new information from easier representation of existing
information. Every subclass here is information-closed by construction
(`H(J|M) = 0`; there is no hidden variable at all, unlike Family A) — what
varies across the 8 required variants is only how hard the visible
information is to *use*, not what information is present. Per spec section
5.2: "All variants must preserve the underlying information set", so
`visible_state`/`full_state` are identical across every subclass for a given
`(seed, index)`; only `render_text`'s presentation differs.

M0 has no separate external `SCAFFOLD` action (spec section 5.2): `THINK`
stands in for it, exposing a deterministic canonical/normalized rendering of
the same information. So the action space here is `{ACT_A, ACT_B, THINK}`,
never `ASK_*` — this family tests accessibility, not information deficit.

Three variants are already in canonical/scaffolded form (`ACCESSIBLE`,
correct action is the direct `ACT_A`/`ACT_B` decision); the other five
require normalization first (correct action is `THINK`, which -- like an
`ASK` that resolves the one relevant fact in Family A -- always succeeds:
`distortion_profile["THINK"]` is always `0.0`, the same "an information
action that helps costs nothing in decision-loss terms" convention
`static_hidden.py` uses for a resolving `ASK`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import Episode, rng_for

SUBCLASSES = (
    "canonical_compact",
    "explicit_derived_comparison",
    "minimal_explanatory_scaffold",
    "reordered_form",
    "irrelevant_constraints_block",
    "redundant_restatement",
    "chain_decomposed_form",
    "misleading_distractor_scaffold",
)

# Already in a normalized/scaffolded form: no THINK needed, spec section
# 10.5's "information present and accessible -> ACT".
ACCESSIBLE = frozenset({"canonical_compact", "explicit_derived_comparison", "minimal_explanatory_scaffold"})

BASE_LOW, BASE_HIGH = 10, 50
ADJ_LOW, ADJ_HIGH = 1, 20


@dataclass(frozen=True)
class _Components:
    a_base: int
    a_adj: int
    a_op: str  # "+" or "-"
    b_base: int
    b_adj: int
    b_op: str


def _apply(op: str, base: int, adj: int) -> int:
    return base + adj if op == "+" else base - adj


def _draw_components(rng) -> _Components:
    while True:
        a_base = int(rng.integers(BASE_LOW, BASE_HIGH + 1))
        a_adj = int(rng.integers(ADJ_LOW, ADJ_HIGH + 1))
        a_op = "+" if int(rng.integers(0, 2)) == 1 else "-"
        b_base = int(rng.integers(BASE_LOW, BASE_HIGH + 1))
        b_adj = int(rng.integers(ADJ_LOW, ADJ_HIGH + 1))
        b_op = "+" if int(rng.integers(0, 2)) == 1 else "-"
        a_cost = _apply(a_op, a_base, a_adj)
        b_cost = _apply(b_op, b_base, b_adj)
        if a_cost != b_cost:  # avoid an ambiguous tie
            return _Components(a_base, a_adj, a_op, b_base, b_adj, b_op)


def generate_episode(seed: int, subclass: str, index: int) -> Episode:
    """Generate one deterministic episode of `subclass` (same seed+index+subclass -> identical episode)."""
    if subclass not in SUBCLASSES:
        raise ValueError(f"unknown accessibility subclass {subclass!r}; expected one of {SUBCLASSES}")
    # Keyed by (seed, index) only, *not* subclass: spec section 5.2 requires
    # every variant to preserve the same underlying information set, so the
    # same (seed, index) must draw identical components across all 8
    # subclasses -- only `render_text`'s presentation may differ.
    rng = rng_for(seed, "accessibility", index)
    c = _draw_components(rng)
    a_cost = _apply(c.a_op, c.a_base, c.a_adj)
    b_cost = _apply(c.b_op, c.b_base, c.b_adj)
    ground_truth_decision = "ACT_B" if b_cost < a_cost else "ACT_A"

    accessible = subclass in ACCESSIBLE
    correct_action = ground_truth_decision if accessible else "THINK"

    # Identical for every subclass -- only `render_text` varies (spec section
    # 5.2's "all variants must preserve the underlying information set").
    state: dict[str, Any] = {
        "A.base": c.a_base,
        "A.adj": c.a_adj,
        "A.op": c.a_op,
        "A.cost": a_cost,
        "B.base": c.b_base,
        "B.adj": c.b_adj,
        "B.op": c.b_op,
        "B.cost": b_cost,
    }

    available_actions = ["ACT_A", "ACT_B", "THINK"]
    distortion_profile = {
        "ACT_A": 0.0 if ground_truth_decision == "ACT_A" else 1.0,
        "ACT_B": 0.0 if ground_truth_decision == "ACT_B" else 1.0,
        "THINK": 0.0,  # THINK always exposes the canonical form; never wrong
    }

    canonical_text = _render(subclass="canonical_compact", components=c)
    scaffold_variants = [] if accessible else [{"kind": "canonical", "text": canonical_text}]

    return Episode(
        episode_id=f"accessibility-{subclass}-{seed}-{index}",
        family="accessibility",
        visible_state=state,
        hidden_state={},
        full_state=state,
        available_actions=available_actions,
        correct_action=correct_action,
        distortion_profile=distortion_profile,
        needs_grounding=False,  # H(J|M) = 0 by construction for every subclass
        min_grounding_bits=0.0,
        minimal_query_set=[],
        scaffold_variants=scaffold_variants,
        metadata={
            "subclass": subclass,
            "accessibility_deficit": not accessible,
            "ground_truth_decision": ground_truth_decision,
        },
    )


def generate_batch(seed: int, subclass: str, n_items: int) -> list[Episode]:
    return [generate_episode(seed, subclass, i) for i in range(n_items)]


def _render(subclass: str, components: _Components) -> str:
    c = components
    a_cost = _apply(c.a_op, c.a_base, c.a_adj)
    b_cost = _apply(c.b_op, c.b_base, c.b_adj)

    if subclass == "canonical_compact":
        lines = [f"A.cost = {a_cost}", f"B.cost = {b_cost}"]
    elif subclass == "explicit_derived_comparison":
        lines = [
            f"A.cost = {c.a_base} {c.a_op} {c.a_adj} = {a_cost}",
            f"B.cost = {c.b_base} {c.b_op} {c.b_adj} = {b_cost}",
        ]
    elif subclass == "minimal_explanatory_scaffold":
        cheaper = "A" if a_cost < b_cost else "B"
        lines = [
            f"A.cost = {c.a_base} {c.a_op} {c.a_adj} = {a_cost}",
            f"B.cost = {c.b_base} {c.b_op} {c.b_adj} = {b_cost}",
            f"Note: {cheaper} is cheaper.",
        ]
    elif subclass == "reordered_form":
        # Raw components only, B before A, no totals -- must be derived.
        lines = [f"B.base = {c.b_base}", f"B.adj = {c.b_adj} ({c.b_op})", f"A.adj = {c.a_adj} ({c.a_op})", f"A.base = {c.a_base}"]
    elif subclass == "irrelevant_constraints_block":
        lines = [
            "CONSTRAINT: all costs are quoted in USD.",
            "CONSTRAINT: costs were last revised 2026-01-01.",
            f"A.base = {c.a_base}",
            f"A.adj = {c.a_adj} ({c.a_op})",
            f"B.base = {c.b_base}",
            f"B.adj = {c.b_adj} ({c.b_op})",
        ]
    elif subclass == "redundant_restatement":
        lines = [
            f"A.base = {c.a_base}",
            f"A.adj = {c.a_adj} ({c.a_op})",
            f"B.base = {c.b_base}",
            f"B.adj = {c.b_adj} ({c.b_op})",
            f"In other words, A combines {c.a_base} and {c.a_adj} by {c.a_op}.",
            f"In other words, B combines {c.b_base} and {c.b_adj} by {c.b_op}.",
        ]
    elif subclass == "chain_decomposed_form":
        lines = [
            f"STEP 1: A.partial = {c.a_base}",
            f"STEP 2: A.cost = A.partial {c.a_op} {c.a_adj}",
            f"STEP 1: B.partial = {c.b_base}",
            f"STEP 2: B.cost = B.partial {c.b_op} {c.b_adj}",
        ]
    elif subclass == "misleading_distractor_scaffold":
        adj_note = "A.adj is larger than B.adj" if c.a_adj > c.b_adj else "A.adj is not larger than B.adj"
        lines = [
            f"A.base = {c.a_base}",
            f"A.adj = {c.a_adj} ({c.a_op})",
            f"B.base = {c.b_base}",
            f"B.adj = {c.b_adj} ({c.b_op})",
            f"Distractor: {adj_note}.",  # true, but not decision-relevant
        ]
    else:
        raise ValueError(f"unknown accessibility subclass {subclass!r}")

    return "\n".join(lines)


def render_text(episode: Episode) -> str:
    """Render `episode`'s stored subclass presentation of TASK ACCESSIBILITY-CHOICE."""
    c = _Components(
        a_base=episode.full_state["A.base"],
        a_adj=episode.full_state["A.adj"],
        a_op=episode.full_state["A.op"],
        b_base=episode.full_state["B.base"],
        b_adj=episode.full_state["B.adj"],
        b_op=episode.full_state["B.op"],
    )
    body = _render(episode.metadata["subclass"], c)
    return "\n".join(["TASK ACCESSIBILITY-CHOICE", body, "DECIDE OR THINK TO NORMALIZE."])
