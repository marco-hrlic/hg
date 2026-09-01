"""Router value under a cost grid -- spec section 10.2/11, gate N2.

Torch-free: takes `episodes` plus `route_pred` (e.g. from
`eval.predict.Predictions.route_pred`), not a model.
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode
from hg_native.model.config import DEFAULT_ROUTES
from hg_native.training.targets import route_costs, route_target


def _episode_route_cost(episode: Episode, route: str, lambda_b: float, lambda_k: float) -> float | None:
    return route_costs(episode, lambda_b, lambda_k).get(route)


def expected_cost(episodes: list[Episode], chosen_routes, lambda_b: float, lambda_k: float) -> float:
    """Mean `C(chosen_route)` over episodes.

    Skips an episode if its chosen route isn't actually one it offers --
    harmless for a model's own prediction (sized to `DEFAULT_ROUTES`, but a
    real family may not offer all three), essential for a fixed "always X"
    baseline, which routinely hits a route a given family never offers
    (e.g. "always THINK" on Family A, which has no THINK action at all).
    """
    total, n = 0.0, 0
    for ep, route in zip(episodes, chosen_routes):
        cost = _episode_route_cost(ep, route, lambda_b, lambda_k)
        if cost is not None:
            total += cost
            n += 1
    return total / n if n > 0 else float("nan")


def route_value_report(episodes: list[Episode], route_pred, lambda_grid: list[tuple[float, float]]) -> dict:
    """Gate N2 (spec section 11): "HG routing must improve expected C ... over
    always ACT, always ASK, always THINK, and the matched baseline router
    ... at more than one isolated price." This reports the model's and every
    "always X"/oracle baseline's expected cost at each `(lambda_b, lambda_k)`
    in `lambda_grid`; the caller supplies the matched-baseline-router
    comparison separately (this module only knows "always X" and the
    oracle, not a second model's predictions) and checks the improvement
    holds at more than one grid point before declaring N2 passed.
    """
    model_routes = [DEFAULT_ROUTES[i] for i in np.asarray(route_pred)]
    grid = []
    for lambda_b, lambda_k in lambda_grid:
        oracle_routes = [DEFAULT_ROUTES[route_target(ep, lambda_b, lambda_k)[0]] for ep in episodes]
        point = {
            "lambda_b": lambda_b,
            "lambda_k": lambda_k,
            "model": expected_cost(episodes, model_routes, lambda_b, lambda_k),
            "oracle": expected_cost(episodes, oracle_routes, lambda_b, lambda_k),
        }
        for route in DEFAULT_ROUTES:
            point[f"always_{route.lower()}"] = expected_cost(episodes, [route] * len(episodes), lambda_b, lambda_k)
        grid.append(point)
    return {"grid": grid}


def route_accuracy(episodes: list[Episode], route_pred, lambda_b: float = 1.0, lambda_k: float = 1.0) -> float:
    """Fraction of episodes where the model's chosen route matches r* (spec
    section 3.3) at one `(lambda_b, lambda_k)` price point."""
    route_pred = np.asarray(route_pred)
    oracle = np.array([route_target(ep, lambda_b, lambda_k)[0] for ep in episodes])
    return float((route_pred == oracle).mean())
