"""Phase 2's task: a controlled natural-language decision with one bit withheld.

Plan section 14. The requirements are specific, and each one is here for a
reason that shows up in the metrics:

* the prompt provides partial information -- so acting now is possible;
* one decision-relevant variable is intentionally hidden -- so there is a
  genuine residual deficit rather than a reasoning gap;
* that variable is available to a simulated source -- so it is acquirable;
* the correct action is computable exactly -- so distortion is measured, not
  judged;
* the optimal query is known -- so a synthesized query has a gold standard to
  be scored against (plan section 15).

The task
--------
Two routes. Route A is faster or slower than B, and A requires exactly one
traveller concession -- a mountain road, a toll, a ferry, night driving, an
unpaved stretch. Route B requires none. The rule, which travels *in the prompt*
so the correct action never depends on the model's own preferences:

    if route A is slower              -> B
    else if A's requirement is acceptable to the traveller -> A
    else                                                   -> B

The traveller's acceptances are private to the source. This yields a residual
deficit that varies per item, which is the point:

    A slower  ->  H(J | M_A) = 0.  B wins either way; asking is waste.
    A faster  ->  H(J | M_A) = 1 bit, and exactly one of the source's
                  variables carries it.

So the same task exercises Phase 1's routing question and Phase 2's query
question at once, and the two are separable: routing decides *whether* to ask,
the query decides *what* to ask.

Why the source holds several variables
--------------------------------------
Generic escalation has to be more expensive than a targeted query or gate G2 is
untestable. The source therefore holds an acceptance bit for every preference
variable, of which exactly one is decision-relevant for a given item. A generic
"tell me everything about the traveller" answer costs log2 of the full
preference alphabet -- `len(PREFERENCE_VARIABLES)` bits -- while the gold
targeted query costs 1. That is Phase 0's structure again, now in natural
language and with the query no longer handed to the system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..inference.base import InferenceItem
from ..inference.prompts import render_prompt
from .channels import QuerySpec, Response, log2_alphabet

# v2 changed two display phrases (see VARIABLE_PHRASES) and so changed the
# rendered prompt. The items themselves are unchanged, but a run is only
# comparable to another at the same version.
ROUTE_CHOICE_TASK_VERSION = "route-choice-v2"
TASK_NAME = "route_choice"
PROMPT_VERSION = "route_choice.v1"

ACTIONS = ("A", "B")

# The traveller concessions the source holds an opinion about. One of these is
# decision-relevant per item; the rest are decoys that a generic escalation
# pays for and a targeted query does not.
PREFERENCE_VARIABLES = (
    "mountain_roads",
    "toll_roads",
    "ferry_crossings",
    "night_driving",
    "unpaved_roads",
)

# Human-readable phrasing, used both in the prompt and in the query text.
#
# Uniform noun phrases, one distinctive head word each. The v1 phrasings for
# night driving and unpaved roads were both "... stretch ...", which made them
# indistinguishable to any parser keying on the phrase and unmatchable from a
# natural rendering like "night driving".
VARIABLE_PHRASES = {
    "mountain_roads": "a mountain road",
    "toll_roads": "a toll road",
    "ferry_crossings": "a ferry crossing",
    "night_driving": "driving at night",
    "unpaved_roads": "an unpaved road",
}

# Surface forms a model might use for each variable, for parsing a synthesized
# query back into a QuerySpec. Listed explicitly rather than derived from the
# display phrase: derivation cannot know that "tolls", "ferries" and "night
# driving" name these variables, and a parser that silently fails to match is
# indistinguishable from a model that failed to ask.
VARIABLE_ALIASES = {
    "mountain_roads": ("mountain road", "mountain roads", "mountain"),
    "toll_roads": ("toll road", "toll roads", "tolls", "toll"),
    "ferry_crossings": ("ferry crossing", "ferry crossings", "ferries", "ferry"),
    "night_driving": ("night driving", "driving at night", "night-driving", "night"),
    "unpaved_roads": ("unpaved road", "unpaved roads", "unpaved", "gravel"),
}

DECISION_RULE = (
    "- If route A's travel time is longer than route B's, choose B.\n"
    "- Otherwise, if route A's requirement is acceptable to the traveller, "
    "choose A.\n"
    "- Otherwise choose B."
)

NO_CONSTRAINTS = "(nothing known about the traveller's preferences)"

# Meta keys the grounding simulator reads. Named on the task side so the
# backend and the task cannot drift apart.
META_REFERENCE_ACTION = "gs_reference_action"
META_SUFFICIENT = "gs_sufficient"
META_HINT = "gs_prior_hint"


@dataclass(frozen=True, slots=True)
class RouteChoiceSpec:
    n_items: int = 400
    # Share of items where route A is faster, i.e. where the hidden variable
    # actually decides. The rest are answerable from the prompt alone.
    p_a_faster: float = 0.5
    # Share of items where the traveller accepts the relevant concession.
    p_accepts: float = 0.5
    variables: tuple[str, ...] = PREFERENCE_VARIABLES
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {self.n_items}")
        for name, value in (("p_a_faster", self.p_a_faster), ("p_accepts", self.p_accepts)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if len(self.variables) < 2:
            raise ValueError(
                f"need at least two preference variables, got {self.variables}: "
                "with one, a generic ask and a targeted query cost the same and "
                "gate G2 is untestable by construction"
            )
        unknown = set(self.variables) - set(VARIABLE_PHRASES)
        if unknown:
            raise ValueError(f"no phrasing registered for {sorted(unknown)}")
        missing_aliases = set(self.variables) - set(VARIABLE_ALIASES)
        if missing_aliases:
            raise ValueError(
                f"no parse aliases registered for {sorted(missing_aliases)}: a "
                "synthesized query naming them could never be parsed"
            )

    @property
    def generic_bits(self) -> float:
        """Cost of a generic ask: one bit per preference variable."""
        return float(len(self.variables))


@dataclass(frozen=True, slots=True)
class RouteChoiceItem:
    """One route-choice instance, holding both private states."""

    item_id: str
    # --- AI-visible -----------------------------------------------------
    time_a: int
    time_b: int
    distance_a: int
    distance_b: int
    toll_a: int
    toll_b: int
    requirement: str  # which concession route A needs
    variables: tuple[str, ...]
    # --- source-private -------------------------------------------------
    acceptances: Mapping[str, bool]

    @property
    def a_faster(self) -> bool:
        return self.time_a < self.time_b

    @property
    def needs_grounding(self) -> bool:
        """True when H(J | M_A) = 1 bit rather than 0."""
        return self.a_faster

    @property
    def reference_action(self) -> str:
        """The exactly-computable correct action."""
        if not self.a_faster:
            return "B"
        return "A" if self.acceptances[self.requirement] else "B"

    @property
    def residual_deficit_bits(self) -> float:
        """H(J | M_A), exact for this construction."""
        return 1.0 if self.needs_grounding else 0.0

    # ------------------------------------------------------------ prompt ---
    def route_text(self) -> str:
        return (
            f"Route A: {self.distance_a} km, toll {self.toll_a} EUR, "
            f"{self.time_a} minutes. Requires {VARIABLE_PHRASES[self.requirement]}.\n"
            f"Route B: {self.distance_b} km, toll {self.toll_b} EUR, "
            f"{self.time_b} minutes. No special requirements."
        )

    def prompt(self, constraints: str = NO_CONSTRAINTS) -> str:
        return render_prompt(
            PROMPT_VERSION,
            self.route_text(),
            ACTIONS,
            extra={"rule": DECISION_RULE, "constraints": constraints},
        )

    def inference_item(
        self,
        constraints: str = NO_CONSTRAINTS,
        revealed: Sequence[str] = (),
    ) -> InferenceItem:
        """The item as handed to any backend, simulated or real.

        `meta` carries what the *simulator* needs and a real model never sees:
        the reference action, and whether the prompt in hand determines it.

        `revealed` names the variables the constraints block discloses, and the
        caller passes it explicitly rather than having this method look for
        phrases in the rendered text. Substring matching against prose is
        fragile in a way that fails silently and in the flattering direction:
        an earlier version compared a lowercase phrase against a capitalized
        sentence, never matched, and so reported every grounded baseline as
        having learned nothing -- a real effect erased by a rendering detail.
        """
        sufficient = self.constraints_suffice(revealed)
        return InferenceItem(
            item_id=self.item_id,
            prompt=self.prompt(constraints),
            labels=list(ACTIONS),
            meta={
                META_REFERENCE_ACTION: self.reference_action,
                META_SUFFICIENT: sufficient,
                # What a model guessing without the missing variable would be
                # drawn towards: route A looks better on every visible number
                # when it is faster.
                META_HINT: "A" if self.a_faster else "B",
                "requirement": self.requirement,
                "needs_grounding": self.needs_grounding,
            },
        )

    def constraints_suffice(self, revealed: Sequence[str]) -> bool:
        """Whether the disclosed variables determine the correct action.

        Trivially true when route A is slower: the rule then closes without any
        preference at all.
        """
        if not self.needs_grounding:
            return True
        return self.requirement in set(revealed)


def generate_route_choice_items(spec: RouteChoiceSpec) -> list[RouteChoiceItem]:
    """Generate the task. Deterministic given the spec."""
    rng = np.random.default_rng([spec.seed, 7714, len(spec.variables)])
    n = spec.n_items
    a_faster = rng.random(n) < spec.p_a_faster
    requirement_idx = rng.integers(0, len(spec.variables), size=n)
    accept = rng.random((n, len(spec.variables))) < spec.p_accepts

    times_b = rng.integers(60, 240, size=n)
    # A margin of at least five minutes, so "faster" is never a rounding
    # artifact a reader could dispute.
    deltas = rng.integers(5, 40, size=n)
    times_a = np.where(a_faster, times_b - deltas, times_b + deltas)

    items: list[RouteChoiceItem] = []
    for i in range(n):
        items.append(
            RouteChoiceItem(
                item_id=f"route-{i:06d}",
                time_a=int(times_a[i]),
                time_b=int(times_b[i]),
                distance_a=int(rng.integers(40, 400)),
                distance_b=int(rng.integers(40, 400)),
                toll_a=int(rng.integers(0, 30)),
                toll_b=int(rng.integers(0, 30)),
                requirement=spec.variables[requirement_idx[i]],
                variables=tuple(spec.variables),
                acceptances={
                    name: bool(accept[i, j]) for j, name in enumerate(spec.variables)
                },
            )
        )
    return items


# ------------------------------------------------------------------ queries ---
def gold_query(item: RouteChoiceItem) -> QuerySpec:
    """The optimal QuerySpec, produced algorithmically (plan section 15).

    One variable, binary answer, one bit -- which equals H(J | M_A) for an item
    that needs grounding, so its coordination overhead is zero. This is the gold
    standard an LM-synthesized query is scored against.
    """
    return targeted_preference_query((item.requirement,))


def targeted_preference_query(variables: Sequence[str]) -> QuerySpec:
    """A query about a specific subset of the traveller's preferences.

    Any subset is expressible, which is what lets a *wrong* or *bloated* query
    be measured rather than merely rejected: asking about three variables when
    one would do costs three bits and shows up as excess response cost.
    """
    names = tuple(variables)
    if not names:
        raise ValueError("a query must name at least one variable")
    unknown = [n for n in names if n not in VARIABLE_PHRASES]
    if unknown:
        raise ValueError(f"unknown preference variables: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError(f"query repeats a variable: {names}")

    phrases = [VARIABLE_PHRASES[n] for n in names]
    if len(names) == 1:
        question = f"Is {phrases[0]} acceptable to you?"
    else:
        question = (
            "Which of these are acceptable to you: "
            + ", ".join(phrases)
            + "?"
        )
    return QuerySpec(
        variable_names=names,
        response_bits=log2_alphabet(2 ** len(names)),
        response_alphabet_size=2 ** len(names),
        candidate_values=None if len(names) > 3 else _enumerate_answers(len(names)),
        candidate_actions_affected=ACTIONS,
        natural_language_query=question,
        metadata={"kind": "targeted" if len(names) == 1 else "subset"},
    )


def generic_query(variables: Sequence[str]) -> QuerySpec:
    """Generic human escalation: ask for the traveller's whole preference set."""
    names = tuple(variables)
    return QuerySpec(
        variable_names=names,
        response_bits=log2_alphabet(2 ** len(names)),
        response_alphabet_size=2 ** len(names),
        candidate_values=None,
        candidate_actions_affected=ACTIONS,
        natural_language_query=(
            "Tell me everything relevant about the traveller: which of "
            + ", ".join(VARIABLE_PHRASES[n] for n in names)
            + " are acceptable?"
        ),
        metadata={"kind": "generic"},
    )


def _enumerate_answers(n: int) -> tuple[str, ...]:
    return tuple(
        "".join(bits)
        for bits in __import__("itertools").product("01", repeat=n)
    )


class TravellerSource:
    """The simulated human. Answers truthfully about variables it holds.

    Truthful and complete by construction: Phase 2 is measuring the *cost* of
    communication, and an unreliable source would confound that with the value
    of the information. Source reliability is a later experiment.
    """

    def __init__(self, item: RouteChoiceItem):
        self.item = item

    def respond(self, query: QuerySpec) -> Response:
        unknown = [n for n in query.variable_names if n not in self.item.acceptances]
        if unknown:
            raise ValueError(f"the source holds no opinion on {unknown}")
        answers = {
            name: bool(self.item.acceptances[name]) for name in query.variable_names
        }
        return Response(
            content=answers,
            raw_bits=query.response_bits,
            informative=True,
            metadata={"n_variables": len(answers)},
        )


def constraints_text(answers: Mapping[str, bool]) -> str:
    """Render a source's answers into the prompt's constraints block."""
    if not answers:
        return NO_CONSTRAINTS
    lines = []
    for name, ok in answers.items():
        phrase = VARIABLE_PHRASES[name]
        lines.append(f"- {phrase.capitalize()} is {'acceptable' if ok else 'not acceptable'}.")
    return "\n".join(lines)


def resolves_decision(item: RouteChoiceItem, query: QuerySpec) -> bool:
    """Whether a query's answer determines the correct action.

    The primary query-quality question (plan section 16): resolution accuracy is
    about the decision, not about whether the query looked sensible.
    """
    if not item.needs_grounding:
        return True
    return item.requirement in query.variable_names


def excess_response_cost(item: RouteChoiceItem, query: QuerySpec) -> float:
    """Omega-hat = B_query - B_optimal (plan section 16).

    Zero for the gold query on an item that needs grounding. Positive for a
    bloated query, and positive for *any* query on an item that needed no
    grounding at all -- where the optimal cost is zero bits, because the prompt
    already determined the answer.
    """
    return float(query.response_bits) - item.residual_deficit_bits
