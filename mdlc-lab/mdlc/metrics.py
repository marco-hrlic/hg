"""Evaluation metrics written out in plain NumPy so they can be read, not just called."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from . import LABELS


def confusion_matrix(y_true: Sequence, y_pred: Sequence, labels: Sequence = LABELS) -> np.ndarray:
    """cm[i, j] = number of items whose true label is labels[i] and predicted label is labels[j]."""
    index = {lab: k for k, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        cm[index[t], index[p]] += 1
    return cm


def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def per_class_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Precision, recall and F1 for every class, from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0)  # column sums: how often we said "class j"
    actual = cm.sum(axis=1)  # row sums: how often class i really occurred
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual}


def macro_f1(y_true: Sequence, y_pred: Sequence, labels: Sequence = LABELS) -> float:
    """Unweighted mean of per-class F1: every class counts the same, however rare."""
    return float(per_class_scores(confusion_matrix(y_true, y_pred, labels))["f1"].mean())


def classification_report(y_true: Sequence, y_pred: Sequence, labels: Sequence = LABELS) -> pd.DataFrame:
    s = per_class_scores(confusion_matrix(y_true, y_pred, labels))
    df = pd.DataFrame(s, index=list(labels))
    df.loc["macro avg"] = [s["precision"].mean(), s["recall"].mean(), s["f1"].mean(), s["support"].sum()]
    df["support"] = df["support"].astype(int)
    return df.round(3)


def cohen_kappa(a: Sequence, b: Sequence, labels: Sequence = LABELS) -> float:
    """Agreement between two raters, corrected for the agreement expected by chance.

    kappa = (p_observed - p_chance) / (1 - p_chance)
    1 = perfect agreement, 0 = no better than chance, < 0 = worse than chance.
    """
    cm = confusion_matrix(a, b, labels).astype(float)
    n = cm.sum()
    p_obs = np.trace(cm) / n
    p_chance = (cm.sum(axis=1) / n) @ (cm.sum(axis=0) / n)
    if p_chance == 1.0:
        return 1.0
    return float((p_obs - p_chance) / (1 - p_chance))


def fleiss_kappa(ratings: pd.DataFrame, item_col: str = "item_id", label_col: str = "label",
                 labels: Sequence = LABELS) -> float:
    """Agreement among many raters. `ratings` has one row per (item, rater) judgement.

    Items may have different numbers of raters; items with fewer than 2 are ignored.
    """
    counts = (
        ratings.groupby([item_col, label_col]).size().unstack(fill_value=0)
        .reindex(columns=list(labels), fill_value=0)
    )
    counts = counts[counts.sum(axis=1) >= 2]
    n_i = counts.sum(axis=1).to_numpy(dtype=float)
    c = counts.to_numpy(dtype=float)
    p_i = ((c * (c - 1)).sum(axis=1)) / (n_i * (n_i - 1))  # agreement within each item
    p_bar = p_i.mean()
    p_j = c.sum(axis=0) / c.sum()
    p_e = (p_j**2).sum()
    if p_e == 1.0:
        return 1.0
    return float((p_bar - p_e) / (1 - p_e))


def bootstrap_ci(y_true: Sequence, y_pred: Sequence, metric=None, n: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval for a metric by resampling the evaluation items with replacement.

    It answers "how much would this score move with a different sample of evaluation items of the
    same size?" A difference between two models that's smaller than this is not evidence of anything.
    """
    metric = metric or macro_f1
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y_true), size=(n, len(y_true)))
    scores = np.array([metric(y_true[i], y_pred[i]) for i in idx])
    return float(np.quantile(scores, alpha / 2)), float(np.quantile(scores, 1 - alpha / 2))


def slice_report(df: pd.DataFrame, y_true: str, y_pred: str, slices_col: str = "slices") -> pd.DataFrame:
    """Accuracy and macro-F1 per slice. `slices_col` holds a list of slice names per row."""
    rows = [("ALL", df)]
    exploded = df.explode(slices_col)
    for name, part in exploded.dropna(subset=[slices_col]).groupby(slices_col):
        rows.append((name, df.loc[part.index.unique()]))
    out = []
    for name, part in rows:
        out.append({
            "slice": name,
            "n": len(part),
            "accuracy": accuracy(part[y_true], part[y_pred]),
            "macro_f1": macro_f1(part[y_true], part[y_pred]),
        })
    return pd.DataFrame(out).set_index("slice").round(3)
