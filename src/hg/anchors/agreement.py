"""Panel agreement statistics.

Used for two purposes:

1. Reporting how ambiguous a task's anchors are, which bounds how much of a
   residual can be attributed to models at all.
2. Sanity-checking a panel before an expensive run. A task whose panel is
   mostly contested cannot support a clean-residual claim, and finding that
   out after five model families have been fine-tuned is expensive.

Chance-corrected agreement is computed the simple way (Fleiss-style expected
agreement from marginal label frequencies). That is enough for a design
diagnostic; it is not a substitute for a proper reliability analysis if the
panel structure is unbalanced.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from ..schema import ANCHOR_CONTESTED, ANCHOR_MAJORITY, ANCHOR_UNANIMOUS
from .panel import AnchoredItem


def observed_agreement(judgments: Sequence[str]) -> float:
    """Probability two randomly drawn annotators on this item agree."""
    n = len(judgments)
    if n < 2:
        return 1.0
    counts = Counter(judgments)
    agreeing_pairs = sum(c * (c - 1) for c in counts.values())
    total_pairs = n * (n - 1)
    return agreeing_pairs / total_pairs


def pairwise_agreement(items: Sequence[AnchoredItem]) -> float:
    """Mean per-item observed agreement across the panel."""
    if not items:
        return float("nan")
    return sum(observed_agreement(it.judgments) for it in items) / len(items)


def expected_agreement(items: Sequence[AnchoredItem]) -> float:
    """Chance agreement from pooled marginal label frequencies."""
    counts: Counter[str] = Counter()
    for it in items:
        counts.update(it.judgments)
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    return sum((c / total) ** 2 for c in counts.values())


def chance_corrected_agreement(items: Sequence[AnchoredItem]) -> float:
    """(P_o - P_e) / (1 - P_e). NaN when the panel is degenerate."""
    p_o = pairwise_agreement(items)
    p_e = expected_agreement(items)
    if not (p_e < 1.0):
        return float("nan")
    return (p_o - p_e) / (1.0 - p_e)


def label_marginals(items: Iterable[AnchoredItem]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for it in items:
        counts.update(it.judgments)
    total = sum(counts.values()) or 1
    return {label: c / total for label, c in sorted(counts.items())}


def panel_report(items: Sequence[AnchoredItem]) -> dict[str, object]:
    """A compact diagnostic block, written into every run's report."""
    n = len(items)
    counts = {ANCHOR_UNANIMOUS: 0, ANCHOR_MAJORITY: 0, ANCHOR_CONTESTED: 0}
    panel_sizes: Counter[int] = Counter()
    for it in items:
        counts[it.status] += 1
        panel_sizes[it.n_annotators] += 1
    return {
        "n_items": n,
        "status_counts": counts,
        "status_shares": {k: (v / n if n else float("nan")) for k, v in counts.items()},
        "panel_sizes": dict(sorted(panel_sizes.items())),
        "observed_agreement": pairwise_agreement(items),
        "expected_agreement": expected_agreement(items),
        "chance_corrected_agreement": chance_corrected_agreement(items),
        "label_marginals": label_marginals(items),
    }


def clean_subset_warnings(report: Mapping[str, object], min_clean: int = 200) -> list[str]:
    """Flag panels too ambiguous or too small to support a clean-residual claim."""
    warnings: list[str] = []
    counts = report.get("status_counts") or {}
    n_clean = int(counts.get(ANCHOR_UNANIMOUS, 0)) if isinstance(counts, Mapping) else 0
    if n_clean < min_clean:
        warnings.append(
            f"only {n_clean} unanimous items; C_E^clean confidence intervals will be "
            f"wide (target >= {min_clean} for a usable clean subset)"
        )
    shares = report.get("status_shares")
    if isinstance(shares, Mapping):
        contested = float(shares.get(ANCHOR_CONTESTED, 0.0) or 0.0)
        if contested > 0.5:
            warnings.append(
                f"{contested:.0%} of items are contested; on this task anchor "
                "formation dominates and a surviving residual should not be read "
                "as an information limit (protocol section 18)"
            )
    sizes = report.get("panel_sizes")
    if isinstance(sizes, Mapping) and sizes and max(int(k) for k in sizes) < 3:
        warnings.append(
            "panel size < 3 for every item: 'unanimous' cannot be distinguished "
            "from 'two annotators happened to agree'"
        )
    if isinstance(counts, Mapping) and int(counts.get(ANCHOR_CONTESTED, 0)) == 0:
        n_total = int(report.get("n_items") or 0)
        if n_total:
            warnings.append(
                "no contested items in the panel: Q_contested and prediction P3 "
                "cannot be tested, and a 'clean' subset that is the whole "
                "dataset is not doing any work. Check the majority threshold "
                "against the panel arithmetic -- with an odd panel and two "
                "labels a tie is impossible, so a bare majority (e.g. 3-2) "
                "counts as 'majority' unless majority_threshold is raised "
                "above it."
            )
    return warnings
