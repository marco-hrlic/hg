"""Grounding (information-deficit) discrimination -- spec section 10.3, gate N1.

Torch-free: takes `episodes` plus an `info_deficit_prob` array (e.g. from
`eval.predict.Predictions.info_deficit_prob`), not a model, so it is testable
with synthetic scores and recomputable from a saved predictions array.
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode

from .binary_metrics import summary
from .bootstrap import BootstrapResult, paired_bootstrap_diff


def needs_grounding_labels(episodes: list[Episode]) -> np.ndarray:
    return np.array([float(ep.needs_grounding) for ep in episodes])


def grounding_discrimination(episodes: list[Episode], info_deficit_prob) -> dict:
    """spec section 10.3: AUROC/AUPRC/Brier/ECE, mean score with/without
    deficit, and a breakdown by task subclass (family A's 6 subclasses each
    isolate a different deficit pattern, so a model that discriminates on
    average but fails on e.g. `redundant_hidden` specifically should show it
    here rather than being averaged away)."""
    labels = needs_grounding_labels(episodes)
    scores = np.asarray(info_deficit_prob, dtype=float)
    if len(scores) != len(episodes):
        raise ValueError(f"{len(episodes)} episodes but {len(scores)} scores")

    result = summary(scores, labels)
    result["mean_score_with_deficit"] = float(scores[labels == 1].mean()) if (labels == 1).any() else float("nan")
    result["mean_score_without_deficit"] = float(scores[labels == 0].mean()) if (labels == 0).any() else float("nan")

    by_subclass: dict[str, dict] = {}
    subclasses = {ep.metadata.get("subclass", ep.family) for ep in episodes}
    for subclass in sorted(subclasses):
        mask = np.array([ep.metadata.get("subclass", ep.family) == subclass for ep in episodes])
        by_subclass[subclass] = summary(scores[mask], labels[mask])
    result["by_subclass"] = by_subclass
    return result


def grounding_auroc_improvement(
    episodes: list[Episode],
    hg_info_deficit_prob,
    baseline_info_deficit_prob,
    n_boot: int = 2000,
    seed: int = 0,
) -> BootstrapResult:
    """Gate N1: paired bootstrap CI for HG's AUROC minus a matched baseline's,
    over the *same* episodes for both (spec section 11: "require a
    replicated AUROC improvement with bootstrap 95% CI for the difference
    excluding 0"). The caller checks `result.ci_low > 0`."""
    labels = needs_grounding_labels(episodes)
    from .binary_metrics import auroc

    return paired_bootstrap_diff(
        (hg_info_deficit_prob, labels),
        (baseline_info_deficit_prob, labels),
        metric=auroc,
        n_boot=n_boot,
        seed=seed,
        name="grounding_auroc_improvement",
    )
