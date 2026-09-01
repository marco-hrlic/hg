"""The canonical `Episode` artifact (spec section 4).

One episode is one decision point with explicit, generator-known ground
truth: what the model can see, what is hidden, what the fully-informed
decision would be, and the exact information-theoretic cost of closing the
gap. Every HG-specific metric in the spec (grounding discrimination, query
efficiency, accessibility, quotient recovery, reuse) is defined in terms of
these fields, so the schema is deliberately flat and JSON-only — the same
rule `hg.grounding.episodes.GroundingEpisode` follows and for the same
reason: the file has to outlive the code that wrote it, and every reported
metric must be recomputable from it without rerunning generation or model
inference (spec section 4, section 16).

`GroundingEpisode` itself is not reused: it is shaped for the HG-v2
route/query/response/cost protocol (one AI turn, one external turn), not for
this schema's generator-known ground truth fields (`needs_grounding`,
`min_grounding_bits`, `minimal_query_set`, `true_quotient_state`, ...).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class Episode:
    """One task instance with full generator-known ground truth.

    Field meanings follow spec section 4 verbatim. `full_state` holds the
    complete generator-side truth (including hidden values the model is not
    shown); `visible_state`/`hidden_state` hold what the model is actually
    shown as KNOWN vs UNKNOWN. `correct_action` is the single right thing to
    do in the unified M0 action space (`ACT_A`/`ACT_B`/`THINK`/`ASK_<k>`),
    not a separate routing vs. business-decision pair — see the worked
    examples in spec section 5.1, where the "correct action" for a
    zero-deficit item is `ACT_A` and for a one-bit-deficit item is `ASK_h3`.
    """

    episode_id: str
    family: str
    visible_state: dict[str, Any]
    hidden_state: dict[str, Any]
    full_state: dict[str, Any]
    available_actions: list[str]
    correct_action: str
    distortion_profile: dict[str, float]
    needs_grounding: bool
    min_grounding_bits: float
    minimal_query_set: list[str] = field(default_factory=list)
    scaffold_variants: list[Any] = field(default_factory=list)
    sequential_state_id: str | None = None
    true_quotient_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Episode":
        # Constructed via keyword args (not `cls(**data)` blindly) so an
        # older or hand-edited artifact missing a field fails loudly at the
        # call site with a clear KeyError rather than silently adopting a
        # dataclass default that may not match what was actually generated.
        known = {f for f in cls.__dataclass_fields__}
        missing = known - data.keys() - _OPTIONAL_FIELDS
        if missing:
            raise ValueError(f"Episode.from_dict missing required fields: {sorted(missing)}")
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json(cls, text: str) -> "Episode":
        return cls.from_dict(json.loads(text))


_OPTIONAL_FIELDS = {
    "minimal_query_set",
    "scaffold_variants",
    "sequential_state_id",
    "true_quotient_state",
    "metadata",
}


def stable_int_key(value: str, digest_size: int = 4) -> int:
    """A small stable integer for a string key, unaffected by PYTHONHASHSEED.

    Spec section 16 explicitly bans Python `hash()` for cross-process
    randomization. Mirrors `hg.queries.synthetic._item_key`.
    """
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=digest_size).digest(), "little")


def rng_for(seed: int, *keys: Any) -> np.random.Generator:
    """A deterministic RNG keyed by `seed` plus arbitrary extra keys.

    Each key is folded into an int-only seed sequence (`np.random.default_rng`
    accepts a sequence of ints), mirroring the pattern used throughout
    `hg.grounding`/`hg.queries` (e.g. `np.random.default_rng([seed, ...])`).
    String keys go through `stable_int_key`; everything else must already be
    an int (raises otherwise, rather than silently truncating a float).
    """
    parts: list[int] = [seed]
    for key in keys:
        if isinstance(key, str):
            parts.append(stable_int_key(key))
        elif isinstance(key, bool):  # bool is an int subclass; keep it explicit
            parts.append(int(key))
        elif isinstance(key, int):
            parts.append(key)
        else:
            raise TypeError(f"rng_for keys must be str or int, got {type(key)!r}")
    return np.random.default_rng(parts)
