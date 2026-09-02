"""A confidence-based ACT/THINK/ASK router built from vanilla-model/proxy
quantities only -- M0-R1 kickoff sections 9-11 / spec section 13's
"vanilla + confidence router" control.

HG's own `RouterHead` decides at test time from `[S_t, r_info_hat,
r_access_hat, c_t]` -- its own (detached) info-deficit/accessibility
*estimates*, not generator ground truth (`model/hg_model.py`). This module
is the same idea built by hand for a vanilla model: it takes the
`LogisticProxy` estimates from `eval.confidence` (`g_info(x)`, `g_access(x)`)
plus the same fixed unit costs `training.targets` uses (`COST_PER_ASK`,
`COST_PER_THINK`, `BLIND_ACT_RISK`), and picks `argmin_r C_hat(r)` under
`C(r) = L(r) + lambda_B*B(r) + lambda_K*K(r)`.

The one thing this router is allowed to know about an episode beyond its
own estimates is which routes are *structurally* on offer
(`episode.available_actions`, e.g. Family A never offers `THINK`) -- that is
visible action-space structure, the same information the rendered prompt
itself carries, not a hidden-state leak. It is never given
`needs_grounding`, `distortion_profile`, or any other hidden-state-derived
ground truth; `test_hg_native_eval_vanilla_router.py` asserts this directly
by construction (the function signature has no way to accept them).
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode
from hg_native.model.config import DEFAULT_ROUTES
from hg_native.training.actions import route_of
from hg_native.training.targets import BLIND_ACT_RISK, COST_PER_ASK, COST_PER_THINK


def _available_routes(episode: Episode) -> set[str]:
    return {route_of(action) for action in episode.available_actions}


def vanilla_route_predictions(
    episodes: list[Episode],
    info_deficit_prob: np.ndarray,
    accessibility_prob: np.ndarray,
    lambda_b: float = 1.0,
    lambda_k: float = 1.0,
) -> np.ndarray:
    """`route_pred` (index into `DEFAULT_ROUTES`) for every episode, from
    `g_info(x)`/`g_access(x)` proxy probabilities alone -- no generator
    ground truth.

    `L_hat(ACT)` uses a soft-OR of the two deficit probabilities against the
    same `BLIND_ACT_RISK` placeholder the true generator cost uses for a
    genuine deficit (`training.targets.route_costs`); `ASK`/`THINK` are
    assumed to fully resolve whatever they target (`L_hat=0`), so their cost
    is just the fixed unit price. This mirrors `route_costs`'s structure
    with estimates standing in for ground truth, exactly as HG's own router
    does with its own heads' outputs.
    """
    info = np.asarray(info_deficit_prob, dtype=float)
    access = np.asarray(accessibility_prob, dtype=float)
    if len(episodes) != len(info) or len(episodes) != len(access):
        raise ValueError(f"{len(episodes)} episodes but {len(info)} info / {len(access)} access probabilities")

    p_deficit = info + access - info * access  # soft OR, assuming independence
    act_cost = p_deficit * BLIND_ACT_RISK
    ask_cost = np.full(len(episodes), lambda_b * COST_PER_ASK)
    think_cost = np.full(len(episodes), lambda_k * COST_PER_THINK)

    preds = np.zeros(len(episodes), dtype=int)
    for i, episode in enumerate(episodes):
        available = _available_routes(episode)
        candidates = {}
        if "ACT" in available:
            candidates["ACT"] = act_cost[i]
        if "ASK" in available:
            candidates["ASK"] = ask_cost[i]
        if "THINK" in available:
            candidates["THINK"] = think_cost[i]
        if not candidates:
            raise ValueError(f"episode {episode.episode_id!r} offers no routable actions")
        best = min(candidates, key=candidates.get)
        preds[i] = DEFAULT_ROUTES.index(best)
    return preds
