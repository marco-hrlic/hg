"""Vanilla decision-probe control -- M1 kickoff section 12.

A stronger vanilla baseline than `eval.confidence`'s generic next-token
proxy (M0-R1's own control, still kept as the secondary comparison per
kickoff section 12's own instruction: "report both... the stronger
decision-probe baseline should be the primary control"). Where
`confidence.confidence_features` reads properties of the model's own
next-token *language-modeling* distribution, this fits a genuinely
supervised linear probe on the **vanilla trunk's raw hidden state**
(`eval.latents.extract_hidden_states`, works for either variant) to
predict the actual business decision, then derives
`decision_probability`/`decision_entropy`/`top1_top2_margin` from that
probe's own output -- the kickoff's own three named features, and the
same shape/contract as `confidence.py`'s `FEATURE_NAMES` so
`vanilla_router.py` can consume either interchangeably.

Torch-free: takes a saved hidden-state array, not a model, matching every
other probe in this eval layer. Reuses `quotient_probes.fit_logistic_regression`
(the same hand-rolled fitter `confidence.fit_logistic_proxy` uses) and
`training.targets.decision_target` for the ground-truth label -- only
episodes with a *defined* decision are used to fit (kickoff section 3.3's
binary A/B decision head has no label for Family D's `ACT_C` outcomes,
`decision_target`'s own documented exclusion).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hg_native.data.schema import Episode
from hg_native.training.targets import decision_target

DECISION_PROBE_FEATURE_NAMES = ("decision_probability", "decision_entropy", "top1_top2_margin")


@dataclass(slots=True)
class DecisionProbe:
    """`P(label=1 | x)` from a linear readout of raw hidden state, fit on a
    train-only split (never the eval split -- caller's responsibility, same
    convention as `eval.projections`/`eval.confidence`)."""

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = ((X - self.mean) / self.std) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_decision_probe(
    hidden_states,
    episodes: list[Episode],
    n_iters: int = 500,
    lr: float = 0.5,
    l2: float = 1e-3,
    seed: int = 0,
) -> DecisionProbe:
    """Fit only on episodes with a defined `decision_target` -- excluding
    e.g. Family D's `ACT_C` outcomes rather than fabricating a label for
    them."""
    from .quotient_probes import fit_logistic_regression

    X = np.asarray(hidden_states, dtype=float)
    if len(X) != len(episodes):
        raise ValueError(f"{len(X)} hidden states but {len(episodes)} episodes")

    rows, labels = [], []
    for x, ep in zip(X, episodes):
        label, defined = decision_target(ep)
        if defined:
            rows.append(x)
            labels.append(float(label))
    if len(rows) < 10:
        raise ValueError(f"need at least 10 defined-decision episodes to fit a decision probe, got {len(rows)}")

    X_defined = np.asarray(rows)
    y_defined = np.asarray(labels)
    mean, std = X_defined.mean(axis=0), X_defined.std(axis=0) + 1e-8
    X_norm = (X_defined - mean) / std
    w, b = fit_logistic_regression(X_norm, y_defined, n_iters=n_iters, lr=lr, l2=l2, seed=seed)
    return DecisionProbe(mean=mean, std=std, weights=w, bias=b)


def decision_probe_features(probe: DecisionProbe, hidden_states) -> np.ndarray:
    """`(N, 3)` array of `DECISION_PROBE_FEATURE_NAMES` for arbitrary
    (e.g. held-out) hidden states -- no episode labels needed here, only
    the probe's own output."""
    p1 = probe.predict_proba(hidden_states)
    p0 = 1.0 - p1
    eps = 1e-12
    decision_probability = np.maximum(p0, p1)
    decision_entropy = -(p0 * np.log(np.clip(p0, eps, 1.0)) + p1 * np.log(np.clip(p1, eps, 1.0)))
    top1_top2_margin = np.abs(p1 - p0)
    return np.stack([decision_probability, decision_entropy, top1_top2_margin], axis=-1)
