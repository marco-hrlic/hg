"""Query quality, decomposed the way the improved plan insists it must be.

Section 2.2: soundness and minimality are different failure modes and must not
be summarized by one number.

    soundness    does the answer carry enough to resolve the decision?
    minimality   conditional on soundness, how much more than B* did it cost?

The previous run makes the point concretely. On Qwen2.5-3B-Instruct the
synthesized query was sound on essentially every item and still cost 2.52 bits
against an optimum of 1.00, because the cost distribution was bimodal: a
one-variable query most of the time, a full five-variable escalation the rest.
An aggregate "query quality" score would have averaged those into a number that
describes neither mode. So this module reports the distribution, and estimates
the weight of each mode explicitly (section 10).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

# Cost modes, for the mixture in plan section 10. `minimal` is a query at the
# residual deficit, `full_escalation` is one that named everything the source
# holds -- generic escalation arrived at by a different route -- and
# `intermediate` is a hedge between them.
MODE_NONE = "no_query"
MODE_MINIMAL = "minimal"
MODE_INTERMEDIATE = "intermediate"
MODE_FULL = "full_escalation"


@dataclass(frozen=True, slots=True)
class QueryOutcome:
    """One item's query, as the quality metrics need to see it.

    Deliberately not a `QuerySpec`: an outcome exists for items where no query
    was sent, and "asked nothing on an item that needed nothing" is a *correct*
    outcome that has to be scoreable rather than skipped.
    """

    item_id: str
    needs_grounding: bool
    asked: bool
    n_variables: int
    response_bits: float
    # Whether a truthful answer resolves the decision (plan section 2.2).
    sound: bool
    # Whether the variable set is exactly the gold one. Stricter than
    # soundness: a superset is sound and not exact.
    exact_gold: bool
    response_alphabet_size: int | None = None

    @property
    def optimal_bits(self) -> float:
        """B*: one bit where the withheld variable decides, zero where it does not."""
        return 1.0 if self.needs_grounding else 0.0

    @property
    def excess_bits(self) -> float:
        """B_Q - B*, the minimality gap. Defined for any outcome; read only for sound ones."""
        return float(self.response_bits) - self.optimal_bits

    def mode(self, n_available: int) -> str:
        if not self.asked:
            return MODE_NONE
        if self.n_variables >= n_available:
            return MODE_FULL
        if self.response_bits <= self.optimal_bits + 1e-9:
            return MODE_MINIMAL
        return MODE_INTERMEDIATE


def oracle_gap_closure(
    generic_bits: float, system_bits: float, oracle_bits: float
) -> float | None:
    """eta_Q = (B_generic - B_system) / (B_generic - B_oracle) (plan section 8).

    None when the denominator vanishes -- a task where generic escalation is
    already optimal has no gap to close, and dividing by zero there would report
    an arbitrarily large or small closure for an effect that does not exist.

    Interpretable only at matched final distortion. This function cannot check
    that, so its caller must.
    """
    denominator = float(generic_bits) - float(oracle_bits)
    if abs(denominator) < 1e-12:
        return None
    return (float(generic_bits) - float(system_bits)) / denominator


def summarize_queries(
    outcomes: Sequence[QueryOutcome], *, n_available: int
) -> dict[str, Any]:
    """Soundness, minimality, waste and the cost distribution for one arm."""
    if not outcomes:
        return {"n": 0}

    n = len(outcomes)
    asked = [o for o in outcomes if o.asked]
    needs = [o for o in outcomes if o.needs_grounding]
    not_needs = [o for o in outcomes if not o.needs_grounding]
    sound = [o for o in outcomes if o.sound]
    sound_asks = [o for o in asked if o.sound]

    bits = np.asarray([o.response_bits for o in outcomes], dtype=float)
    modes = Counter(o.mode(n_available) for o in outcomes)

    def rate(subset: Sequence[QueryOutcome], predicate) -> float | None:
        return float(np.mean([predicate(o) for o in subset])) if subset else None

    return {
        "n": n,
        "n_asked": len(asked),
        # --- routing behaviour, split the way plan section 8 requires -------
        "ask_rate": len(asked) / n,
        "ask_rate_when_grounding_needed": rate(needs, lambda o: o.asked),
        "ask_rate_when_not_needed": rate(not_needs, lambda o: o.asked),
        # --- what the source had to send -----------------------------------
        "mean_response_bits": float(bits.mean()),
        "response_bits_distribution": {
            str(value): int(count)
            for value, count in sorted(Counter(bits.tolist()).items())
        },
        "mean_response_bits_when_asked": (
            float(np.mean([o.response_bits for o in asked])) if asked else 0.0
        ),
        "mean_variables_when_asked": (
            float(np.mean([o.n_variables for o in asked])) if asked else 0.0
        ),
        "variables_distribution": {
            str(value): int(count)
            for value, count in sorted(
                Counter(o.n_variables for o in asked).items()
            )
        },
        # --- soundness (plan section 2.2) -----------------------------------
        "soundness": len(sound) / n,
        "soundness_when_asked": (len(sound_asks) / len(asked)) if asked else None,
        "soundness_when_grounding_needed": rate(needs, lambda o: o.sound),
        "exact_gold_rate_when_asked": (
            rate(asked, lambda o: o.exact_gold) if asked else None
        ),
        # --- minimality, conditional on soundness ---------------------------
        # Conditional because an unsound query's cheapness is not economy: the
        # two failure modes have to stay separable, and averaging an unsound
        # 1-bit query into the excess would make a broken synthesizer look
        # minimal.
        "mean_excess_bits_given_sound": (
            float(np.mean([o.excess_bits for o in sound])) if sound else None
        ),
        "mean_excess_bits_given_sound_and_asked": (
            float(np.mean([o.excess_bits for o in sound_asks])) if sound_asks else None
        ),
        "mean_excess_bits_all_items": float(
            np.mean([o.excess_bits for o in outcomes])
        ),
        # --- waste (plan section 8: "a central integrated-routing metric") ---
        # Bits spent on items the AI-visible prompt already determined. Zero is
        # attainable: the information was never needed.
        "waste_bits_per_item": float(
            np.sum([o.response_bits for o in not_needs]) / n
        ) if not_needs else 0.0,
        "waste_share_of_bits": (
            float(np.sum([o.response_bits for o in not_needs]) / bits.sum())
            if bits.sum() > 0
            else None
        ),
        "unnecessary_ask_rate": rate(not_needs, lambda o: o.asked),
        # --- the mixture of plan section 10 ---------------------------------
        "cost_modes": {mode: modes.get(mode, 0) / n for mode in
                       (MODE_NONE, MODE_MINIMAL, MODE_INTERMEDIATE, MODE_FULL)},
        # alpha is the weight of the minimal mode among items where a query was
        # sent. H-C predicts that minimality interventions raise it by draining
        # the full-escalation mode, so the two are reported side by side.
        "alpha_minimal": (
            modes.get(MODE_MINIMAL, 0) / len(asked) if asked else None
        ),
        "share_full_escalation": (
            modes.get(MODE_FULL, 0) / len(asked) if asked else None
        ),
    }
