"""Ask the primary LM what it needs to know (plan sections 15 and 17, B4).

This is the piece that makes baseline B4 a test of the architecture rather than
of a stand-in: the query is produced by the same frozen model that has to make
the decision, and it is scored against the algorithmically-produced gold
`QuerySpec`.

Design decisions that affect what the numbers mean
--------------------------------------------------
**The output space is variable names, not free text.** The model picks from a
listed set of available facts. That makes a synthesized query directly
comparable to the gold one -- same representation, same cost function -- instead
of requiring someone to interpret a sentence and decide what it "really asked".
It also means the measured quantity is *targeting*, not phrasing.

**The instruction asks for as few facts as possible.** Response cost is the
thing under test, and a prompt that merely asked "what do you need?" invites
hedging. A hedge resolves the decision and overpays, which is precisely the
failure mode gate G2 exists to detect, so the model is told the cost exists.

**A query that cannot be parsed falls back to generic escalation.** This is the
conservative choice and it is deliberately unflattering: the alternative --
asking nothing -- would let a model that failed to produce a query score as
cheap. A real system that cannot form a targeted question has to escalate, and
the fallback rate is reported so the cost is attributable.

**NONE is honoured.** On items whose prompt already settles the rule the right
query is no query, and a synthesizer that recognizes that pays zero bits. This
is the one place B4 can legitimately beat the always-ask oracle B3 on cost, and
it is worth being able to observe: it is routing and query synthesis arriving in
the same decision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..grounding.channels import QuerySpec
from ..grounding.route_choice import (
    ACTIONS,
    VARIABLE_ALIASES,
    VARIABLE_PHRASES,
    RouteChoiceItem,
    generic_query,
    targeted_preference_query,
)
from ..inference.base import Backend, GenerationSettings, InferenceItem
from ..inference.prompts import render_prompt

log = logging.getLogger(__name__)

QUERY_SYNTH_PROMPT_VERSION = "query_synth.v1"
NONE_TOKEN = "NONE"

# What to do when the model's answer names no valid variable.
ON_FAILURE_GENERIC = "generic"
ON_FAILURE_NONE = "none"


@dataclass(slots=True)
class SynthesisOutcome:
    """One synthesized query plus how it was arrived at."""

    item_id: str
    query: QuerySpec | None
    raw_output: str
    parse_ok: bool
    named_variables: list[str] = field(default_factory=list)
    said_none: bool = False
    fell_back: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "raw_output": self.raw_output,
            "parse_ok": self.parse_ok,
            "named_variables": list(self.named_variables),
            "said_none": self.said_none,
            "fell_back": self.fell_back,
            "response_bits": self.query.response_bits if self.query else 0.0,
        }


class LMQuerySynthesizer:
    """Synthesizes a query by prompting a frozen LM for the facts it needs."""

    name = "lm"

    def __init__(
        self,
        backend: Backend,
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        seed: int = 0,
        on_parse_failure: str = ON_FAILURE_GENERIC,
        # Refuse a query naming more than this many variables. A model that
        # names everything has produced generic escalation with extra steps,
        # and letting it through as a "targeted" query would misattribute the
        # cost. Capped at the full set by default, i.e. no refusal.
        max_variables: int | None = None,
    ):
        if on_parse_failure not in (ON_FAILURE_GENERIC, ON_FAILURE_NONE):
            raise ValueError(
                f"on_parse_failure must be {ON_FAILURE_GENERIC!r} or "
                f"{ON_FAILURE_NONE!r}, got {on_parse_failure!r}"
            )
        self.backend = backend
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.on_parse_failure = on_parse_failure
        self.max_variables = max_variables
        self.outcomes: dict[str, SynthesisOutcome] = {}

    # ------------------------------------------------------------ prompting ---
    def prompt_for(self, item: RouteChoiceItem) -> str:
        available = "\n".join(
            f"- {name}: whether {VARIABLE_PHRASES[name]} is acceptable"
            for name in item.variables
        )
        return render_prompt(
            QUERY_SYNTH_PROMPT_VERSION,
            item.route_text(),
            ACTIONS,
            extra={"rule": item.decision_rule(), "available": available},
        )

    def _inference_items(self, items: Sequence[RouteChoiceItem]) -> list[InferenceItem]:
        return [
            InferenceItem(
                item_id=item.item_id,
                prompt=self.prompt_for(item),
                # The answer space is variable names, so the label list is not
                # a forced choice here; it travels for provenance only.
                labels=list(item.variables) + [NONE_TOKEN],
                meta={},
            )
            for item in items
        ]

    # ----------------------------------------------------------- synthesis ---
    def synthesize_batch(
        self, items: Sequence[RouteChoiceItem]
    ) -> list[QuerySpec | None]:
        """Synthesize for many items in one backend call.

        Batched because the Backend protocol is batch-oriented and a per-item
        call on a real model wastes most of the GPU.
        """
        if not items:
            return []
        settings = GenerationSettings(
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            seed=self.seed,
        )
        predictions = self.backend.predict(self._inference_items(items), settings)

        out: list[QuerySpec | None] = []
        for item, prediction in zip(items, predictions):
            raw = prediction.raw_output or (prediction.label or "")
            outcome = self._parse(item, raw)
            self.outcomes[item.item_id] = outcome
            out.append(outcome.query)
        return out

    def synthesize(self, item: RouteChoiceItem) -> QuerySpec | None:
        return self.synthesize_batch([item])[0]

    def _parse(self, item: RouteChoiceItem, raw: str) -> SynthesisOutcome:
        text = (raw or "").strip()
        named = extract_variables(text, item.variables)
        said_none = bool(re.search(rf"\b{NONE_TOKEN}\b", text, flags=re.IGNORECASE))

        if named and self.max_variables is not None and len(named) > self.max_variables:
            log.debug(
                "%s: query named %d variables, above max_variables=%d",
                item.item_id, len(named), self.max_variables,
            )
            named = []

        if named:
            return SynthesisOutcome(
                item_id=item.item_id,
                query=targeted_preference_query(named),
                raw_output=text,
                parse_ok=True,
                named_variables=named,
            )

        # "NONE" is a real answer: on an item the prompt already settles, the
        # correct query is no query at all.
        if said_none:
            return SynthesisOutcome(
                item_id=item.item_id,
                query=None,
                raw_output=text,
                parse_ok=True,
                said_none=True,
            )

        if self.on_parse_failure == ON_FAILURE_GENERIC:
            return SynthesisOutcome(
                item_id=item.item_id,
                query=generic_query(item.variables),
                raw_output=text,
                parse_ok=False,
                fell_back=True,
            )
        return SynthesisOutcome(
            item_id=item.item_id, query=None, raw_output=text, parse_ok=False
        )

    # ------------------------------------------------------------ reporting ---
    def summary(self) -> dict[str, Any]:
        if not self.outcomes:
            return {"n": 0}
        n = len(self.outcomes)
        return {
            "n": n,
            "parse_ok_rate": sum(o.parse_ok for o in self.outcomes.values()) / n,
            "fallback_rate": sum(o.fell_back for o in self.outcomes.values()) / n,
            "said_none_rate": sum(o.said_none for o in self.outcomes.values()) / n,
            "mean_variables_named": sum(
                len(o.named_variables) for o in self.outcomes.values()
            ) / n,
            "prompt_version": QUERY_SYNTH_PROMPT_VERSION,
            "on_parse_failure": self.on_parse_failure,
            "examples": [
                o.to_dict() for o in list(self.outcomes.values())[:5]
            ],
        }


def extract_variables(text: str, available: Sequence[str]) -> list[str]:
    """Pull known variable names out of a model's answer.

    Matches the canonical names and the registered aliases, deduplicates, and
    returns them in the order the *task* declares rather than the order the
    model happened to mention them -- so two models naming the same set produce
    the same query and therefore the same cost.

    Longer aliases are tried before shorter ones so "mountain road" is not
    consumed by the bare "mountain" of another entry; matching is on whole words
    throughout.

    Only the first line is considered. Small instruct models often continue past
    the answer with an explanation, and a later sentence mentioning every option
    ("...I do not need to know about tolls or ferries") would otherwise be read
    as asking about all of them.
    """
    if not text:
        return []
    head = text.strip().splitlines()[0]
    lowered = head.lower()

    found: set[str] = set()
    for name in available:
        surface = [name.lower(), *(a.lower() for a in VARIABLE_ALIASES.get(name, ()))]
        for alias in sorted(surface, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", lowered):
                found.add(name)
                break
    return [name for name in available if name in found]
