"""Per-episode head targets, computed from generator ground truth alone.

Pure Python (lists/floats/bools, no torch) so every target function is
testable everywhere and the batch serializer (`batch.py`) can stay
torch-free too; the trainer converts to tensors at the point it actually
needs them.

Every function returns `(value, defined)` except `distortion_target`
(which returns a fixed-size `(vector, mask)` pair) and
`info_deficit_target` (always defined, since spec section 3.3's
`G_t^info` is defined for every episode). `defined=False` means "this
episode's family/state doesn't pin an answer for this head" -- the loss
registry must mask these out rather than train against a fabricated 0,
per spec section 2.3's instruction not to collapse or fake the two
deficit heads into a single always-defined score.
"""

from __future__ import annotations

from hg_native.data.schema import Episode
from hg_native.model.config import DEFAULT_DECISION_LABELS, DEFAULT_ROUTES

from .actions import ACTIONS, action_id

# Fixed unit costs for the router's B(r)/K(r) (spec section 3.3's route cost
# C(r) = L(r) + lambda_B*B(r) + lambda_K*K(r)): one query costs 0.1 bit-units,
# one THINK costs 0.1 compute-units, scaled by the caller's lambda_b/lambda_k.
# Not pinned by the spec -- an implementation placeholder, flagged per
# [[hg-theory-implementation-split]].
COST_PER_ASK = 0.1
COST_PER_THINK = 0.1

# L(ACT) when a genuine deficit (informational or accessibility) is present.
#
# `distortion_profile["ACT_*"]` is *realized*, generator-known ground truth
# (spec section 4: it is computed from `full_state`, which includes the true
# hidden value) -- so whichever action matches `ground_truth_decision`
# always scores 0 there, deficit or not. Using that realized value here
# would make acting blind look free even when the router has no way to know
# which action is correct, silently defeating the entire point of routing.
# `BLIND_ACT_RISK` stands in for "expected distortion of acting without
# resolving a real deficit" -- 0.5 for a binary decision with no
# discriminating side information, i.e. a fair coin flip. Like the costs
# above, this is a placeholder base rate the generator declares, not a
# quantity measured from any real model's actual failure rate.
BLIND_ACT_RISK = 0.5


def decision_target(episode: Episode) -> tuple[int | None, bool]:
    """p_theta(J_t | S_t) target: index into DEFAULT_DECISION_LABELS, or (None, False).

    Undefined for Family D's `ACT_C` outcomes: spec section 3.3 defines the
    decision head as binary A/B "for M0 binary tasks", and Family D's whole
    point (a third action appearing under a richer capability) does not fit
    that space -- it is probed by representation separability, not this head.
    """
    business = episode.metadata.get("ground_truth_decision")
    if business is None and episode.correct_action in ("ACT_A", "ACT_B"):
        business = episode.correct_action
    if business not in ("ACT_A", "ACT_B"):
        return None, False
    label = DEFAULT_DECISION_LABELS.index(business[-1])  # "ACT_A" -> "A" -> index 0
    return label, True


def distortion_target(episode: Episode) -> tuple[list[float], list[bool]]:
    """g_d(S_t, a) targets: fixed-size (over ACTIONS) vector + mask.

    Only the actions actually present in `episode.distortion_profile` are
    marked defined; a family that never offers `ASK_h2`, say, must not train
    that slot toward a fabricated value.
    """
    vec = [0.0] * len(ACTIONS)
    mask = [False] * len(ACTIONS)
    for name, value in episode.distortion_profile.items():
        idx = action_id(name)
        vec[idx] = value
        mask[idx] = True
    return vec, mask


def info_deficit_target(episode: Episode) -> float:
    """p(G_t^info=1 | S_t) target. Always defined: `needs_grounding` is
    generator-known for every M0 episode (spec section 3.3)."""
    return float(episode.needs_grounding)


def accessibility_target(episode: Episode) -> tuple[float, bool]:
    """p(G_t^access=1 | S_t) target.

    Defined only when `needs_grounding` is False: spec section 3.3's own
    definition (`G_t^access = 1[H(J_t|M_t)=0 AND base model fails]`) is
    conditioned on zero information deficit, so a deficit episode has no
    accessibility label to give, not a label of 0.
    """
    if episode.needs_grounding:
        return 0.0, False
    return float(episode.metadata.get("accessibility_deficit", False)), True


def route_target(episode: Episode, lambda_b: float = 1.0, lambda_k: float = 1.0) -> tuple[int, bool]:
    """r* = argmin_r C(r) = L(r) + lambda_B*B(r) + lambda_K*K(r) (spec section 3.3).

    `L(r)` is the best available action's distortion within that route
    (e.g. `L(ACT) = min` over `ACT_*` distortions); `B`/`K` are the fixed
    unit costs above, zero for routes that don't incur them. A route absent
    from this episode's `available_actions` (e.g. `THINK` outside Family B)
    is simply not a candidate, rather than being scored as free.
    """
    candidates: dict[str, tuple[float, float, float]] = {}  # route -> (L, B, K)
    act_costs = [v for a, v in episode.distortion_profile.items() if a.startswith("ACT_")]
    if act_costs:
        has_deficit = episode.needs_grounding or bool(episode.metadata.get("accessibility_deficit", False))
        act_l = BLIND_ACT_RISK if has_deficit else min(act_costs)
        candidates["ACT"] = (act_l, 0.0, 0.0)
    think_costs = [v for a, v in episode.distortion_profile.items() if a == "THINK"]
    if think_costs:
        candidates["THINK"] = (min(think_costs), 0.0, COST_PER_THINK)
    ask_costs = [v for a, v in episode.distortion_profile.items() if a.startswith("ASK")]
    if ask_costs:
        candidates["ASK"] = (min(ask_costs), COST_PER_ASK, 0.0)

    if not candidates:
        return 0, False

    route_cost = {r: (l + lambda_b * b + lambda_k * k) for r, (l, b, k) in candidates.items()}
    best_route = min(route_cost, key=route_cost.get)
    return DEFAULT_ROUTES.index(best_route), True
