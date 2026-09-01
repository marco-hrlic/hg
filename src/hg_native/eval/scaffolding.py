"""Accessibility / scaffolding evaluation -- spec section 10.5, gate N3.

Torch-free: takes `episodes` plus an `accessibility_prob`/`route_pred`
array, not a model.
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode
from hg_native.model.config import DEFAULT_ROUTES

from .binary_metrics import summary


def accessibility_labels_and_mask(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    """`(labels, mask)`: `mask` selects only zero-information-deficit episodes
    -- spec section 3.3 defines `G_t^access` conditioned on `H(J_t|M_t)=0`,
    so a genuine-deficit episode has no accessibility label to evaluate
    against, not a label of 0 (mirrors `training.targets.accessibility_target`)."""
    mask = np.array([not ep.needs_grounding for ep in episodes])
    labels = np.array([float(ep.metadata.get("accessibility_deficit", False)) for ep in episodes])
    return labels, mask


def accessibility_discrimination(episodes: list[Episode], accessibility_prob) -> dict:
    labels, mask = accessibility_labels_and_mask(episodes)
    scores = np.asarray(accessibility_prob, dtype=float)
    if len(scores) != len(episodes):
        raise ValueError(f"{len(episodes)} episodes but {len(scores)} scores")
    if not mask.any():
        return {"n": 0, "n_zero_deficit": 0}
    result = summary(scores[mask], labels[mask])
    result["n_zero_deficit"] = int(mask.sum())
    return result


def route_rates_on_zero_deficit(episodes: list[Episode], route_pred) -> dict:
    """ACT/THINK/ASK rates on zero-information-deficit items (spec section 10.5).

    Desired qualitative behavior per that section: THINK-worthy computation
    failures should route to THINK, not ASK -- there is no missing
    information for ASK to resolve, so any ASK here is definitionally
    unnecessary.
    """
    labels, mask = accessibility_labels_and_mask(episodes)
    route_pred = np.asarray(route_pred)
    if len(route_pred) != len(episodes):
        raise ValueError(f"{len(episodes)} episodes but {len(route_pred)} route predictions")
    if not mask.any():
        return {"n": 0}

    routes = route_pred[mask]
    sub_labels = labels[mask]
    rates = {route: float((routes == i).mean()) for i, route in enumerate(DEFAULT_ROUTES)}
    deficit_mask = sub_labels == 1

    result = {
        "n": int(mask.sum()),
        "route_rates": rates,
        "unnecessary_ask_rate": rates["ASK"],  # every item here has zero info deficit
    }
    if deficit_mask.any():
        think_idx = DEFAULT_ROUTES.index("THINK")
        result["think_rate_on_accessibility_deficit"] = float((routes[deficit_mask] == think_idx).mean())
    return result
