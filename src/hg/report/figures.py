"""Figures, with a CSV fallback.

matplotlib is an optional dependency. The protocol's own emphasis -- "the
expensive artifact is the prediction matrix, not the summary plot" -- is the
reason this module is allowed to degrade: if matplotlib is absent, the numbers
behind every figure are still written as CSV, and nothing upstream fails.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)


def _have_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def write_survival_csv(path: str | Path, curve: Sequence[Mapping[str, Any]]) -> None:
    from ..io import write_csv

    write_csv(
        path,
        [dict(p) for p in curve],
        columns=["k", "mean", "sd", "min", "max", "n_subsets"],
    )


def plot_survival(
    path: str | Path,
    curves: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    title: str = "Residual survival curve",
    ylabel: str = "clean consensus-wrong rate",
) -> bool:
    """Plot C_k against k for one or more curves. Returns True if a plot was written.

    The y-axis is forced to include zero: the whole question is whether the
    curve reaches zero, and an autoscaled axis that starts at 0.01 makes a
    surviving residual look like a plateau at a glance.
    """
    csv_path = Path(path).with_suffix(".csv")
    first = next(iter(curves.values()), [])
    if first:
        write_survival_csv(csv_path, first)

    if not _have_matplotlib():
        log.info("matplotlib not installed; wrote %s instead of a figure", csv_path)
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    for name, curve in curves.items():
        ks = [p["k"] for p in curve]
        means = [p["mean"] for p in curve]
        sds = [p.get("sd", 0.0) or 0.0 for p in curve]
        ax.errorbar(ks, means, yerr=sds, marker="o", capsize=3, label=name)

    ax.set_xlabel("k = number of independent model families")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
    if len(curves) > 1:
        ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_race(
    path: str | Path,
    race: Mapping[str, Any],
    *,
    title: str = "Localization race",
) -> bool:
    """Recall of clean consensus failures against audit budget, per strategy."""
    strategies = race.get("strategies") or {}
    rows: list[dict[str, Any]] = []
    for name, r in strategies.items():
        for b, rec in zip(r.get("budgets", []), r.get("recall", [])):
            rows.append({"strategy": name, "budget": b, "recall": rec})
    if rows:
        from ..io import write_csv

        write_csv(Path(path).with_suffix(".csv"), rows, columns=["strategy", "budget", "recall"])

    if not _have_matplotlib() or not strategies:
        return False

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=150)
    for name, r in strategies.items():
        ax.plot(r.get("budgets", []), r.get("recall", []), marker="o", label=name)
    ax.set_xlabel("human-label budget (items audited)")
    ax.set_ylabel("share of clean consensus failures found")
    ax.set_title(title)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return True
