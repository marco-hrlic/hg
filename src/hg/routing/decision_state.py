"""The decision state a router sees, and the features it is built from.

Plan section 4.1. `DecisionState` is the practical object: a bundle of
estimator outputs describing how the decision stands *right now*, before a
route is chosen.

The honesty rule is enforced by the `exact` flag rather than by convention.
Phase 0's synthetic task can fill these fields analytically, because its joint
distribution is constructed; anything a learned predictor produces is a proxy.
Nothing here should be read as an estimate of a Shannon quantity unless that
correspondence has been derived and validated (plan sections 4.1 and 18).

The features are deliberately computed from AI-private state alone. That is not
a stylistic choice: a router that can see the reference decision, the source's
bits, or the generative error rate would report its own leak as routing skill.
`agreement_features` therefore takes the observations and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

# Feature order is a contract with a fitted router's weight vector, so it is an
# explicit tuple rather than whatever order a dict happens to iterate in.
FEATURE_NAMES = ("majority_margin", "vote_entropy_bits", "unanimous")


@dataclass(slots=True)
class DecisionState:
    """How the decision stands before a route is chosen (plan section 4.1)."""

    item_id: str
    candidate_actions: list[str]
    predicted_action: str | None = None

    # Empirical / proxy quantities. None means "not estimated here", which is
    # different from zero and must stay distinguishable in the artifact.
    expected_loss_now: float | None = None
    internal_reducibility: float | None = None
    external_value: float | None = None
    human_value: float | None = None
    contestedness: float | None = None
    consequence: float | None = None

    # False for anything a predictor produced. Only a constructed synthetic
    # task may set this True.
    exact: bool = False
    features: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "candidate_actions": list(self.candidate_actions),
            "predicted_action": self.predicted_action,
            "expected_loss_now": self.expected_loss_now,
            "internal_reducibility": self.internal_reducibility,
            "external_value": self.external_value,
            "human_value": self.human_value,
            "contestedness": self.contestedness,
            "consequence": self.consequence,
            "exact": self.exact,
            "features": dict(self.features),
            "metadata": dict(self.metadata),
        }


def agreement_features(observations: Sequence[int]) -> dict[str, float]:
    """Internal-agreement features from the AI's own observations.

    Takes the observations and nothing else, so a router built on this cannot
    accidentally see the reference decision or the generative noise level.

    * `majority_margin`  how lopsided the internal vote is, in [0, 1].
    * `vote_entropy_bits` the same information as entropy, which is the shape
      the existing M2-LLM disagreement metric uses.
    * `unanimous`        1.0 when every observation agrees. Called out
      separately because the mapping from agreement to error probability is
      steepest at the unanimous end, and a linear model in the margin alone
      fits that corner badly.
    """
    m = len(observations)
    if m == 0:
        raise ValueError("need at least one observation to build features")
    ones = int(sum(observations))
    zeros = m - ones

    # Both features are derived from the integer counts rather than from a
    # probability and its complement. `max(p, 1 - p)` is not exactly
    # label-symmetric in floating point -- for m = 3, a 2-1 split for one and a
    # 2-1 split for zero differ by one ULP -- and these vectors are used as
    # dictionary keys when enumerating the router's feature space, so two
    # identical evidence states must produce byte-identical features or they
    # split into spurious levels.
    majority, minority = max(ones, zeros), min(ones, zeros)
    majority_share = majority / m

    if minority == 0:
        entropy = 0.0
    else:
        p_hi, p_lo = majority / m, minority / m
        entropy = float(-(p_hi * math.log2(p_hi) + p_lo * math.log2(p_lo)))
    return {
        "majority_margin": float((majority - minority) / m),
        "vote_entropy_bits": entropy,
        "unanimous": 1.0 if minority == 0 else 0.0,
    }


def feature_matrix(rows: Sequence[dict[str, float]]) -> np.ndarray:
    """Stack feature dicts into an (n, len(FEATURE_NAMES)) array in fixed order."""
    return np.asarray(
        [[float(r[name]) for name in FEATURE_NAMES] for r in rows], dtype=float
    )
