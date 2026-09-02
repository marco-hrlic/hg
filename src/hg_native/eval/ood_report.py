"""Per-condition Q_sel matrix reporting -- M1 kickoff section 10/11/20.

Torch-free: takes precomputed latents + episodes per named condition (e.g.
"iid", "parameter", "lexical", "distractor" -- the `metadata["ood_condition"]`
tags `hg_native.data.ood`'s pool builders already set), not a model, so this
is recomputable from saved `latents.npy`/`episodes.jsonl` per condition
alone, matching every other eval module's convention.

Deliberately thin: `eval.quotient_probes.quotient_recovery` already computes
`Q_sel` for one condition, and `eval.bootstrap.bootstrap_ci` is already
generic over "any scalar metric computed from item-indexed arrays" --
`np.asarray` on a list of `Episode` objects makes an object array that
resamples with ordinary fancy indexing, so `bootstrap_ci(latents, episodes,
metric=condition_q_sel)` works unmodified. Nothing here reimplements either.
"""

from __future__ import annotations

import math

import numpy as np

from hg_native.data.schema import Episode
from hg_native.eval.bootstrap import BootstrapResult, bootstrap_ci
from hg_native.eval.quotient_probes import quotient_recovery


def condition_q_sel(latents, episodes, relevant_vars: tuple[str, ...], irrelevant_vars: tuple[str, ...], **probe_kwargs) -> float:
    """`quotient_recovery(...)["q_sel"]` -- the scalar metric `bootstrap_ci`
    and `condition_matrix` need."""
    return quotient_recovery(latents, episodes, relevant_vars=relevant_vars, irrelevant_vars=irrelevant_vars, **probe_kwargs)["q_sel"]


def condition_matrix(
    latents_by_condition: dict[str, np.ndarray],
    episodes_by_condition: dict[str, list[Episode]],
    relevant_vars: tuple[str, ...],
    irrelevant_vars: tuple[str, ...],
    iid_key: str = "iid",
    n_boot: int = 2000,
    seed: int = 0,
    **probe_kwargs,
) -> dict[str, dict]:
    """Kickoff section 20's per-condition table, one row per key shared by
    both dicts: `quotient_recovery`'s full per-variable probe result plus a
    95% bootstrap CI on `Q_sel`. Requires `iid_key` present in both dicts --
    `rho_ood` below needs an IID point estimate to divide by."""
    if iid_key not in latents_by_condition or iid_key not in episodes_by_condition:
        raise ValueError(f"condition_matrix requires iid_key={iid_key!r} present in both latents_by_condition and episodes_by_condition")
    conditions = set(latents_by_condition) & set(episodes_by_condition)
    matrix: dict[str, dict] = {}
    for condition in conditions:
        latents = latents_by_condition[condition]
        episodes = episodes_by_condition[condition]
        recovery = quotient_recovery(latents, episodes, relevant_vars=relevant_vars, irrelevant_vars=irrelevant_vars, **probe_kwargs)
        ci = bootstrap_ci(
            latents,
            np.asarray(episodes, dtype=object),
            metric=lambda lat, eps: condition_q_sel(lat, list(eps), relevant_vars, irrelevant_vars, **probe_kwargs),
            n_boot=n_boot,
            seed=seed,
            name=f"q_sel[{condition}]",
        )
        matrix[condition] = {"probes": recovery["probes"], "q_sel": recovery["q_sel"], "q_sel_ci": ci}
    return matrix


def rho_ood(matrix: dict[str, dict], condition: str, iid_key: str = "iid") -> float:
    """`Q_sel^condition / Q_sel^iid` (kickoff section 11). `NaN` when the IID
    point estimate is `<= 0` -- the kickoff's own warning that the ratio
    "becomes unstable near zero" (section 11: "do not use this alone"),
    matching how undefined ratios are handled elsewhere in this eval layer
    (e.g. `quotient_probes.quotient_recovery`'s own NaN-guarded mean)."""
    q_sel_iid = matrix[iid_key]["q_sel"]
    q_sel_condition = matrix[condition]["q_sel"]
    if not (q_sel_iid > 0) or math.isnan(q_sel_condition):
        return float("nan")
    return q_sel_condition / q_sel_iid


def format_matrix_table(matrix: dict[str, dict], iid_key: str = "iid") -> str:
    """Plain-text rendering of the section 20 table: condition | Q_sel |
    95% CI | rho_OOD. `iid_key` sorts first; the rest follow alphabetically
    for a stable, diffable report."""
    conditions = [iid_key] + sorted(c for c in matrix if c != iid_key)
    header = f"{'condition':<12} {'Q_sel':>8} {'95% CI':>18} {'rho_OOD':>8}"
    lines = [header, "-" * len(header)]
    for condition in conditions:
        q_sel = matrix[condition]["q_sel"]
        ci: BootstrapResult = matrix[condition]["q_sel_ci"]
        rho = "--" if condition == iid_key else f"{rho_ood(matrix, condition, iid_key=iid_key):.4f}"
        lines.append(f"{condition:<12} {q_sel:>8.4f} [{ci.ci_low:>6.4f}, {ci.ci_high:>6.4f}] {rho:>8}")
    return "\n".join(lines)
