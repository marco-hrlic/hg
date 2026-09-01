"""Routing: which items to audit (M2-LLM), and act-vs-ask (HG-v2).

Two related but distinct decisions live here. `strategies` ranks *items* to
spend a human-label budget on; `grounding_router` chooses a *route* for one
item. They are kept separate because the decision variable differs -- see
experiments/hg_v2/REUSE_AUDIT.md.
"""

from .decision_state import (
    FEATURE_NAMES,
    DecisionState,
    agreement_features,
    feature_matrix,
)
from .grounding_router import (
    FIXED_POLICIES,
    LogisticRouter,
    oracle_route,
    route_from_expected_cost,
)
from .strategies import (
    STRATEGIES,
    LocalizationRace,
    RaceResult,
    disagreement_score,
    failure_memory_score,
    run_race,
    uncertainty_score,
)

__all__ = [
    "DecisionState",
    "FEATURE_NAMES",
    "agreement_features",
    "feature_matrix",
    "FIXED_POLICIES",
    "LogisticRouter",
    "oracle_route",
    "route_from_expected_cost",
    "STRATEGIES",
    "LocalizationRace",
    "RaceResult",
    "disagreement_score",
    "uncertainty_score",
    "failure_memory_score",
    "run_race",
]
