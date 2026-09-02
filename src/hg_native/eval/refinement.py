"""Capability-refinement quotient metric -- M1 kickoff section 8.3/8.4.

`capability_refinement.py` (Family D, reworked for M1 -- see its module
docstring) exposes state `Z=(r,f,u,n,v)` rendered under two capability
tokens `C1`/`C2`, where `v` is decision-irrelevant under `C1` and becomes
relevant only under `C2` (a new action's correctness depends on it). The
theory's prediction (kickoff section 8.3):

    Delta_refine(v) = ProbeAcc(v | S, C2) - ProbeAcc(v | S, C1) > 0

with the stronger control (H-M1-5) that a variable irrelevant under both
capabilities (`u`) shows no such shift:

    Delta_refine(u) ~= 0

Torch-free, same convention as `eval.quotient_probes`: takes a saved latent
array plus episodes, not a model, so it's recomputable from
`latents.npy` + `episodes.jsonl` alone.
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode
from hg_native.eval.quotient_probes import probe_accuracy, variable_label

CAPABILITIES = ("C1", "C2")


def probe_by_capability(latents, episodes: list[Episode], var: str, **probe_kwargs) -> dict[str, dict]:
    """`probe_accuracy` for `var`, fit and scored separately on the
    `capability == "C1"` and `capability == "C2"` subsets of `episodes`
    (matched against `latents` by position; every episode must carry
    `metadata["capability"]`, e.g. capability_refinement's Family D)."""
    latents = np.asarray(latents)
    if len(latents) != len(episodes):
        raise ValueError(f"{len(latents)} latents but {len(episodes)} episodes")
    out: dict[str, dict] = {}
    for capability in CAPABILITIES:
        idx = [i for i, ep in enumerate(episodes) if ep.metadata.get("capability") == capability]
        if not idx:
            raise ValueError(f"no episodes with metadata['capability'] == {capability!r}")
        labels = np.array([variable_label(episodes[i], var) for i in idx])
        out[capability] = probe_accuracy(latents[idx], labels, **probe_kwargs)
    return out


def delta_refine(latents, episodes: list[Episode], var: str, **probe_kwargs) -> float:
    """`ProbeAcc(var | C2) - ProbeAcc(var | C1)`. NaN if either side's probe
    is undefined (e.g. a single-class training split)."""
    by_capability = probe_by_capability(latents, episodes, var, **probe_kwargs)
    return by_capability["C2"]["accuracy"] - by_capability["C1"]["accuracy"]


def refinement_report(latents, episodes: list[Episode], refining_var: str, control_var: str, **probe_kwargs) -> dict:
    """The full H-M1-4/H-M1-5 pair in one call: `refining_var` (predicted
    `Delta_refine > 0`, e.g. `"v"`) alongside `control_var` (predicted
    `Delta_refine ~= 0`, e.g. `"u"`)."""
    return {
        refining_var: probe_by_capability(latents, episodes, refining_var, **probe_kwargs),
        control_var: probe_by_capability(latents, episodes, control_var, **probe_kwargs),
        "delta_refine": {
            refining_var: delta_refine(latents, episodes, refining_var, **probe_kwargs),
            control_var: delta_refine(latents, episodes, control_var, **probe_kwargs),
        },
    }
