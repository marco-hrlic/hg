"""Grounding reuse -- spec section 10.7, gate N5.

Torch-free: takes Family C episodes (from `sequential_quotient.py`, run
through `eval.predict.predict`) plus `route_pred`.
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode
from hg_native.model.config import DEFAULT_ROUTES
from hg_native.training.targets import COST_PER_ASK


def _grounded_mask(episodes: list[Episode]) -> np.ndarray:
    return np.array([bool(ep.metadata.get("grounded", False)) for ep in episodes])


def repeated_ask_rate(episodes: list[Episode], route_pred) -> dict:
    """R_repeat = P(ASK again | h already grounded and still valid) (spec section 10.7).

    Restricted to episodes with `metadata["grounded"] is True` (t>0 in
    `sequential_quotient.py`, where the idealized policy should never need
    to ask again) -- a t=0 episode is *supposed* to ask, so it must not
    count as a repeat.
    """
    route_pred = np.asarray(route_pred)
    if len(route_pred) != len(episodes):
        raise ValueError(f"{len(episodes)} episodes but {len(route_pred)} route predictions")
    mask = _grounded_mask(episodes)
    if not mask.any():
        return {"n": 0}
    ask_idx = DEFAULT_ROUTES.index("ASK")
    return {"n": int(mask.sum()), "r_repeat": float((route_pred[mask] == ask_idx).mean())}


def future_external_bits(episodes: list[Episode], route_pred) -> float:
    """B_future = sum_{t>t0} B_t (spec section 10.7): total external-info cost
    paid on already-grounded steps, at the fixed per-ask cost `route_costs`/
    the router's own training target use (`training.targets.COST_PER_ASK`)."""
    route_pred = np.asarray(route_pred)
    mask = _grounded_mask(episodes)
    if not mask.any():
        return 0.0
    ask_idx = DEFAULT_ROUTES.index("ASK")
    return float(((route_pred[mask] == ask_idx).astype(float) * COST_PER_ASK).sum())
