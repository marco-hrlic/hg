"""Query efficiency -- spec section 10.4.

Uses `distortion_hat` (the distortion head's own per-action predictions),
not `route_pred`, to determine which single action the model would take:
the natural policy the distortion head's own training target supports is
"take the available action with the lowest predicted distortion", and
unlike the 3-way route, it names a specific `ASK_h<k>`.

Restricted to Family A (`static_hidden.py`), the only family with genuine
per-variable `ASK_h*` actions. Family C's `ASK_h` is a single
undifferentiated action (nothing to be sound/minimal *about* beyond "ask or
don't"); Families B/D have no ASK action at all.

M0's structured, single-hop action space collapses most of this section's
soundness/minimality distinction from the spec's original (free-form query
text) framing: there is no query text to score for extraneous content, so
"sound" here means "resolves the deficit" and "minimal" means "cost matched
the oracle exactly" -- both read directly from generator ground truth
(`distortion_profile`, `min_grounding_bits`), never estimated.
"""

from __future__ import annotations

import numpy as np

from hg_native.data.schema import Episode
from hg_native.training.actions import action_id

# One ASK_k resolves (or fails to resolve) exactly one bit -- matches
# min_grounding_bits' units for a single-variable deficit.
COST_PER_QUERY_BIT = 1.0


def chosen_action(episode: Episode, distortion_hat_row) -> str:
    """The action with lowest *predicted* distortion among this episode's
    `available_actions`."""
    candidates = {a: distortion_hat_row[action_id(a)] for a in episode.available_actions}
    return min(candidates, key=candidates.get)


def query_efficiency(episodes: list[Episode], distortion_hat) -> dict:
    distortion_hat = np.asarray(distortion_hat)
    if len(distortion_hat) != len(episodes):
        raise ValueError(f"{len(episodes)} episodes but {len(distortion_hat)} distortion predictions")

    b_system: list[float] = []
    b_star: list[float] = []
    resolved: list[bool] = []
    sound_asks: list[bool] = []
    n_deficit = 0
    unnecessary_asks = 0
    under_asks = 0

    for ep, row in zip(episodes, distortion_hat):
        action = chosen_action(ep, row)
        asked = action.startswith("ASK")
        this_resolved = ep.distortion_profile[action] == 0.0
        resolved.append(this_resolved)
        b_system.append(COST_PER_QUERY_BIT if asked else 0.0)

        if ep.needs_grounding:
            n_deficit += 1
            b_star.append(ep.min_grounding_bits)
            if asked:
                sound_asks.append(this_resolved)
            else:
                under_asks += 1
        elif asked:
            unnecessary_asks += 1

    b_system_arr = np.array(b_system)
    resolved_arr = np.array(resolved)
    n = len(episodes)
    n_no_deficit = n - n_deficit

    result = {
        "n": n,
        "n_deficit": n_deficit,
        "b_system_mean": float(b_system_arr.mean()) if n > 0 else float("nan"),
        "b_star_mean": float(np.mean(b_star)) if b_star else float("nan"),
        "soundness": float(np.mean(sound_asks)) if sound_asks else float("nan"),
        "unnecessary_ask_rate": (unnecessary_asks / n_no_deficit) if n_no_deficit > 0 else float("nan"),
        "under_asking_rate": (under_asks / n_deficit) if n_deficit > 0 else float("nan"),
        "bits_per_resolved_decision": (float(b_system_arr.sum() / resolved_arr.sum()) if resolved_arr.sum() > 0 else float("nan")),
    }

    # eta_Q = (B_generic - B_system) / (B_generic - B*) (spec section 10.4).
    # B_generic (the "uninformed" baseline cost) is taken as this episode
    # set's largest min_grounding_bits -- a fixed generic query covering the
    # hardest case, matching how the hg-v2 H2 gate compared against a
    # stated-budget baseline rather than a measured one (see
    # [[hg-v2-project-state]]).
    if n_deficit > 0:
        b_generic = max(ep.min_grounding_bits for ep in episodes if ep.needs_grounding)
        b_system_deficit_mean = float(b_system_arr[np.array([ep.needs_grounding for ep in episodes])].mean())
        denom = b_generic - result["b_star_mean"]
        result["b_generic"] = b_generic
        result["eta_q"] = float((b_generic - b_system_deficit_mean) / denom) if denom > 0 else float("nan")
    else:
        result["eta_q"] = float("nan")

    return result
