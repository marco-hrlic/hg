"""Synthesizers with known behaviour, for validating the query metrics.

The gold synthesizer is the standard (plan section 15). The noisy one is a
*controlled* stand-in for an imperfect synthesizer -- an LM that sometimes asks
about the wrong variable, or hedges by asking about several. It exists so the
query-quality metrics can be checked against known behaviour before an LM's
queries are measured with them: a metric that cannot distinguish a bloated query
from a targeted one would silently flatter gate G2.
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..grounding.channels import QuerySpec
from ..grounding.route_choice import (
    RouteChoiceItem,
    gold_query,
    targeted_preference_query,
)


class GoldQuerySynthesizer:
    """Asks about exactly the decision-relevant variable. One bit."""

    name = "gold"

    def synthesize(self, item: RouteChoiceItem) -> QuerySpec:
        return gold_query(item)


class NoisyQuerySynthesizer:
    """An imperfect synthesizer with three configurable failure modes.

    * with probability `p_correct`, the gold query;
    * else with probability `p_superset`, a hedge: the right variable plus
      `n_extra` decoys, which resolves the decision but overpays;
    * else a miss: a single wrong variable, which is cheap and useless.

    The hedge is the interesting case for gate G2. A synthesizer that hedges
    still achieves full resolution accuracy, so an evaluation that looked only
    at decision quality would call it as good as gold. The cost metrics are what
    separate them, which is the whole claim being tested.
    """

    name = "noisy"

    def __init__(
        self,
        *,
        p_correct: float = 0.7,
        p_superset: float = 0.2,
        n_extra: int = 2,
        seed: int = 0,
    ):
        for label, value in (("p_correct", p_correct), ("p_superset", p_superset)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be in [0, 1], got {value}")
        if p_correct + p_superset > 1.0:
            raise ValueError(
                f"p_correct + p_superset = {p_correct + p_superset} > 1"
            )
        if n_extra < 1:
            raise ValueError(f"n_extra must be >= 1, got {n_extra}")
        self.p_correct = float(p_correct)
        self.p_superset = float(p_superset)
        self.n_extra = int(n_extra)
        self.seed = int(seed)

    def synthesize(self, item: RouteChoiceItem) -> QuerySpec:
        # Keyed on the item, not drawn from a walking stream, so a synthesizer's
        # behaviour on one item does not depend on how many items preceded it.
        #
        # The key is a *stable* digest rather than `hash(item.item_id)`. Python
        # salts string hashing per process, so the earlier version produced a
        # different query set on every run -- the arms stayed internally
        # consistent and the saved-artifact round-trip still reproduced, so
        # nothing failed loudly, but two runs of the same config disagreed. Cost
        # comparisons between runs are the whole point here, and the repository
        # requires deterministic inference for exactly this reason.
        rng = np.random.default_rng([self.seed, _item_key(item.item_id)])
        u = float(rng.random())
        others = [v for v in item.variables if v != item.requirement]

        if u < self.p_correct or not others:
            return gold_query(item)

        if u < self.p_correct + self.p_superset:
            n_extra = min(self.n_extra, len(others))
            decoys = list(rng.choice(others, size=n_extra, replace=False))
            return targeted_preference_query([item.requirement, *decoys])

        wrong = str(rng.choice(others))
        return targeted_preference_query([wrong])


def _item_key(item_id: str) -> int:
    """A stable small integer key for an item id, unaffected by PYTHONHASHSEED."""
    return int.from_bytes(
        hashlib.blake2b(item_id.encode("utf-8"), digest_size=4).digest(), "little"
    )
