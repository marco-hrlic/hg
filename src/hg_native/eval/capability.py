"""Capability floor -- spec section 10.2, gate N0.

Torch-free: takes `episodes` plus `decision_pred`, not a model.
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode
from hg_native.model.config import DEFAULT_DECISION_LABELS


def business_decision_labels_and_mask(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
    """(labels, mask): mask selects episodes with a defined binary A/B
    ground-truth decision -- mirrors `training.targets.decision_target`
    exactly (Family D's `ACT_C` outcomes are excluded the same way)."""
    labels = np.full(len(episodes), -1, dtype=int)
    mask = np.zeros(len(episodes), dtype=bool)
    for i, ep in enumerate(episodes):
        business = ep.metadata.get("ground_truth_decision")
        if business is None and ep.correct_action in ("ACT_A", "ACT_B"):
            business = ep.correct_action
        if business in ("ACT_A", "ACT_B"):
            labels[i] = DEFAULT_DECISION_LABELS.index(business[-1])
            mask[i] = True
    return labels, mask


def capability_floor(episodes: list[Episode], decision_pred, collapse_threshold: float = 0.99) -> dict:
    """Gate N0: "the fully grounded model must materially exceed chance and
    avoid constant-label collapse" (spec section 11). `collapse_threshold`
    flags a prediction distribution that has collapsed onto one label.
    """
    labels, mask = business_decision_labels_and_mask(episodes)
    decision_pred = np.asarray(decision_pred)
    if len(decision_pred) != len(episodes):
        raise ValueError(f"{len(episodes)} episodes but {len(decision_pred)} decision predictions")
    if not mask.any():
        return {"n": 0}

    y, yhat = labels[mask], decision_pred[mask]
    per_class = {}
    for i, label in enumerate(DEFAULT_DECISION_LABELS):
        class_mask = y == i
        per_class[label] = float((yhat[class_mask] == i).mean()) if class_mask.any() else float("nan")
    prediction_rate = {label: float((yhat == i).mean()) for i, label in enumerate(DEFAULT_DECISION_LABELS)}

    return {
        "n": int(mask.sum()),
        "accuracy": float((y == yhat).mean()),
        "balanced_accuracy": float(np.nanmean(list(per_class.values()))),
        "per_class_accuracy": per_class,
        "prediction_rate": prediction_rate,
        "constant_label_collapse": max(prediction_rate.values()) > collapse_threshold,
    }
