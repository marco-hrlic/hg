"""Choosing between acting and asking (plan sections 9 to 11).

The route objective is

    C_r = L_r + lambda * B_r

and the optimal route is r* = argmin_r C_r. With ACT costing no external bits
and a targeted ask costing B_ask bits that resolve the decision, this reduces to

    route ACT   iff   E[L_act] <= lambda * B_ask

so a router only ever needs one estimate: the probability that acting now is
wrong. Everything else in the comparison is known cost.

That is also why the router here is a probability model thresholded by lambda
rather than a classifier trained per lambda. For 0 < lambda*B_ask < 1 the
supervised target r* *is* the indicator that acting now would be wrong, so
fitting P(wrong | features) is fitting r* -- with the practical advantage that
one fitted model can be repriced across the whole lambda sweep without
refitting, which is the same repricing property the episode artifact has.

No RL, per plan section 9. A logistic regression on internal-agreement features,
fitted by full-batch gradient descent in numpy: it has a closed decision rule,
deterministic training, and no new dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..grounding.channels import Route
from .decision_state import FEATURE_NAMES, feature_matrix

# The fixed policies every learned router has to beat (plan section 11).
FIXED_POLICIES = ("always_act", "always_ask")
# References that bound what any router could achieve.
REFERENCE_POLICIES = ("oracle", "bayes")
POLICIES = FIXED_POLICIES + ("learned",) + REFERENCE_POLICIES


def route_from_expected_cost(
    p_wrong: float,
    *,
    lambda_bit: float,
    ask_bits: float = 1.0,
    loss_if_wrong: float = 1.0,
) -> Route:
    """argmin over {ACT, ASK} of L_r + lambda * B_r.

    Ties go to ACT: at exact indifference the two routes cost the same, and
    spending someone else's attention for nothing is the worse tie-break.
    """
    cost_act = float(p_wrong) * float(loss_if_wrong)
    cost_ask = float(lambda_bit) * float(ask_bits)
    return Route.ACT if cost_act <= cost_ask else Route.ASK_TARGETED


def oracle_route(
    loss_act: float,
    *,
    lambda_bit: float,
    ask_bits: float = 1.0,
    loss_ask: float = 0.0,
) -> Route:
    """The clairvoyant route, using the realized counterfactual losses.

    Unattainable by construction -- it knows whether acting *did* go wrong --
    which is exactly what makes it the right denominator for routing regret.
    """
    cost_act = float(loss_act)
    cost_ask = float(loss_ask) + float(lambda_bit) * float(ask_bits)
    return Route.ACT if cost_act <= cost_ask else Route.ASK_TARGETED


@dataclass(slots=True)
class LogisticRouter:
    """P(acting now is wrong | internal-agreement features).

    Standardizes features on the training set, then fits by gradient descent
    with L2 regularization. Deterministic: no shuffling, no random restarts,
    weights initialized at zero.
    """

    learning_rate: float = 0.5
    n_steps: int = 2000
    l2: float = 1e-4
    weights: list[float] = field(default_factory=list)
    intercept: float = 0.0
    feature_names: tuple[str, ...] = FEATURE_NAMES
    # Training-set standardization, stored so evaluation applies the same
    # transform. Refitting these on the evaluation set would leak.
    mean: list[float] = field(default_factory=list)
    scale: list[float] = field(default_factory=list)
    n_train: int = 0
    final_loss: float = float("nan")

    def fit(
        self,
        features: Sequence[dict[str, float]],
        labels: Sequence[float],
    ) -> "LogisticRouter":
        x = feature_matrix(features)
        y = np.asarray(list(labels), dtype=float)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{x.shape[0]} feature rows but {y.shape[0]} labels")
        if x.shape[0] == 0:
            raise ValueError("cannot fit a router on an empty training set")

        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        # A constant feature carries no information; scaling it by zero would
        # produce NaNs and silently poison every downstream weight.
        scale = np.where(scale < 1e-12, 1.0, scale)
        z = (x - mean) / scale

        w = np.zeros(z.shape[1])
        b = 0.0
        n = z.shape[0]
        for _ in range(self.n_steps):
            p = _sigmoid(z @ w + b)
            residual = p - y
            grad_w = z.T @ residual / n + self.l2 * w
            grad_b = float(residual.mean())
            w -= self.learning_rate * grad_w
            b -= self.learning_rate * grad_b

        p = _sigmoid(z @ w + b)
        eps = 1e-12
        self.weights = [float(v) for v in w]
        self.intercept = float(b)
        self.mean = [float(v) for v in mean]
        self.scale = [float(v) for v in scale]
        self.n_train = int(n)
        self.final_loss = float(
            -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        )
        return self

    def predict_proba(self, features: Sequence[dict[str, float]]) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("router has not been fitted")
        x = feature_matrix(features)
        z = (x - np.asarray(self.mean)) / np.asarray(self.scale)
        return _sigmoid(z @ np.asarray(self.weights) + self.intercept)

    def route(
        self,
        features: dict[str, float],
        *,
        lambda_bit: float,
        ask_bits: float = 1.0,
    ) -> Route:
        p = float(self.predict_proba([features])[0])
        return route_from_expected_cost(p, lambda_bit=lambda_bit, ask_bits=ask_bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "logistic",
            "feature_names": list(self.feature_names),
            "weights": list(self.weights),
            "intercept": self.intercept,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "n_train": self.n_train,
            "final_train_loss": self.final_loss,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "l2": self.l2,
        }


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Split on sign so neither exp() branch can overflow.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out
