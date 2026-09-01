"""Family D — Capability Refinement generator (spec section 5.4).

Exploratory, not required for M0 success (spec section 5.4's own words) --
included anyway since the checklist (spec section 26, item 9) asks for it
and it's cheap once Family A's cost-comparison pattern exists.

Purpose: does the learned decision representation change when the action
frontier changes? Each call generates one **pair** of underlying states
`z`, `z'` and renders each under *both* capability tokens, i.e. 4 episodes:

    (z,  C1)   (z,  C2)
    (z', C1)   (z', C2)

constructed so `z ~_C1 z'` (both correct_action = ACT_A under the 2-action
frontier `{ACT_A, ACT_B}`) but `z !~_C2 z'` once a third action `ACT_C`
is introduced: `z` is built so `ACT_C` becomes strictly the cheapest
option under C2 (its correct_action flips), while `z'` is built so
`ACT_C` is never competitive (its correct_action stays `ACT_A`). A
representation that is sensitive to capability should therefore separate
`z` from `z'` more under C2 than under C1 -- the spec's own prediction
(section 5.4: "lower separability under C1, higher under C2").
"""

from __future__ import annotations

from typing import Any

from .schema import Episode, rng_for

MEMBERS = ("z", "z_prime")
CAPABILITIES = ("C1", "C2")

A_LOW, A_HIGH = 20, 50
B_MARGIN_LOW, B_MARGIN_HIGH = 1, 30  # B.cost = A.cost + margin, so A is always cheaper under C1


def _build_episode(pair_id: str, member: str, capability: str, a_cost: int, b_cost: int, c_cost: int) -> Episode:
    visible_state: dict[str, Any] = {"A.cost": a_cost, "B.cost": b_cost, "C.cost": c_cost}
    if capability == "C1":
        available_actions = ["ACT_A", "ACT_B"]
        costs = {"ACT_A": a_cost, "ACT_B": b_cost}
    else:
        available_actions = ["ACT_A", "ACT_B", "ACT_C"]
        costs = {"ACT_A": a_cost, "ACT_B": b_cost, "ACT_C": c_cost}
    correct_action = min(costs, key=costs.get)
    distortion_profile = {a: (0.0 if a == correct_action else 1.0) for a in available_actions}

    return Episode(
        episode_id=f"capability_refinement-{pair_id}-{member}-{capability}",
        family="capability_refinement",
        visible_state=visible_state,
        hidden_state={},
        full_state=visible_state,
        available_actions=available_actions,
        correct_action=correct_action,
        distortion_profile=distortion_profile,
        needs_grounding=False,
        min_grounding_bits=0.0,
        minimal_query_set=[],
        metadata={
            "pair_id": pair_id,
            "member": member,
            "capability": capability,
        },
    )


def generate_pair(seed: int, index: int) -> dict[str, Episode]:
    """Generate the 4 episodes for one (z, z') pair under both capabilities.

    Keys: `"z_C1"`, `"z_C2"`, `"z_prime_C1"`, `"z_prime_C2"`.
    """
    rng = rng_for(seed, "capability_refinement", index)
    a_cost = int(rng.integers(A_LOW, A_HIGH + 1))
    b_cost = a_cost + int(rng.integers(B_MARGIN_LOW, B_MARGIN_HIGH + 1))  # always > a_cost

    z_c_cost = int(rng.integers(1, a_cost))  # strictly cheaper than A: ACT_C wins under C2
    zprime_c_cost = a_cost + int(rng.integers(1, 50))  # strictly costlier than A: ACT_C never wins

    pair_id = f"pair-{seed}-{index}"
    return {
        "z_C1": _build_episode(pair_id, "z", "C1", a_cost, b_cost, z_c_cost),
        "z_C2": _build_episode(pair_id, "z", "C2", a_cost, b_cost, z_c_cost),
        "z_prime_C1": _build_episode(pair_id, "z_prime", "C1", a_cost, b_cost, zprime_c_cost),
        "z_prime_C2": _build_episode(pair_id, "z_prime", "C2", a_cost, b_cost, zprime_c_cost),
    }


def generate_batch(seed: int, n_pairs: int) -> list[dict[str, Episode]]:
    return [generate_pair(seed, i) for i in range(n_pairs)]
