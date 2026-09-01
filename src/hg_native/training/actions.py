"""The canonical action vocabulary shared across all four M0 task families.

Each generator names its own actions (`static_hidden.py` has `ASK_h1`/
`ASK_h2`; `accessibility.py` has `THINK`; `sequential_quotient.py` has
`ASK_h`; `capability_refinement.py` has `ACT_C`), but the distortion and
transition heads (`HeadsConfig.n_actions`) need one fixed-size, stable
mapping from action name to integer id shared across every family so a
batch mixing episodes from different generators can be collated at all.

Not pinned by the spec (section 3.3 only fixes the *route* set
`{ACT, THINK, ASK}`, not the action vocabulary underneath it) — an
implementation choice, flagged per [[hg-theory-implementation-split]].
"""

from __future__ import annotations

# Order is arbitrary but must stay stable once any run has trained against
# it -- treat this tuple as append-only.
ACTIONS: tuple[str, ...] = (
    "ACT_A",
    "ACT_B",
    "ACT_C",
    "THINK",
    "ASK_h",
    "ASK_h1",
    "ASK_h2",
)

_ACTION_TO_ID = {name: i for i, name in enumerate(ACTIONS)}


def action_id(name: str) -> int:
    try:
        return _ACTION_TO_ID[name]
    except KeyError:
        raise ValueError(f"{name!r} is not in the canonical ACTIONS vocabulary {ACTIONS}") from None


def action_name(idx: int) -> str:
    return ACTIONS[idx]


def route_of(action: str) -> str:
    """Which of the spec section 3.3 routes {ACT, THINK, ASK} an action belongs to."""
    if action.startswith("ACT_"):
        return "ACT"
    if action == "THINK":
        return "THINK"
    if action.startswith("ASK"):
        return "ASK"
    raise ValueError(f"{action!r} does not map to a known route")
