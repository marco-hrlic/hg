"""Versioned prompt templates.

Protocol section 18 lists "inference-format artifacts" as a confound:
prompting differences across model families can manufacture disagreement that
looks like representational diversity. Two consequences, both enforced here:

* One template is used across all families in a condition. Family-specific
  chat formatting is applied by the backend (via the tokenizer's chat
  template), not by rewriting the instruction.
* The template version is recorded on every row, so a mid-experiment prompt
  change cannot be silently averaged into the results.

Adding a template: register a new version rather than editing an existing one.
Old rows must stay interpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    version: str
    instruction: str
    # `{labels}` and `{x}` are substituted; everything else is literal.
    body: str
    # Text the model is expected to emit. Kept beside the template so the
    # canonicalizer and the prompt cannot drift apart.
    answer_prefix: str = "Answer:"

    def render(self, x: str, labels: Sequence[str], extra: Mapping[str, str] | None = None) -> str:
        label_list = ", ".join(labels)
        text = self.body.format(
            instruction=self.instruction,
            labels=label_list,
            x=x,
            answer_prefix=self.answer_prefix,
            **(extra or {}),
        )
        return text


# v1: single-token forced-choice classification. Deliberately terse -- long
# rationales widen the output distribution and make canonicalization the
# dominant source of variance across families.
CLASSIFY_V1 = PromptTemplate(
    version="classify.v1",
    instruction=(
        "Classify the input. Reply with exactly one of the allowed labels and nothing else."
    ),
    body=(
        "{instruction}\n"
        "Allowed labels: {labels}\n\n"
        "Input:\n{x}\n\n"
        "{answer_prefix}"
    ),
)

# v1 for task family B (policy application, section 5.2): the policy travels in
# the prompt so the target is fixed and external knowledge is not required.
POLICY_V1 = PromptTemplate(
    version="policy.v1",
    instruction=(
        "Apply the policy below to the case. Reply with exactly one of the "
        "allowed labels and nothing else. Follow the policy even where your own "
        "judgment would differ."
    ),
    body=(
        "{instruction}\n\n"
        "Policy:\n{policy}\n\n"
        "Allowed labels: {labels}\n\n"
        "Case:\n{x}\n\n"
        "{answer_prefix}"
    ),
)

# v1 for task family D (preference ranking): A/B choice over two candidates.
PREFERENCE_V1 = PromptTemplate(
    version="preference.v1",
    instruction=(
        "Two candidate responses are shown. Reply with exactly one of the "
        "allowed labels naming the better response, and nothing else."
    ),
    body=(
        "{instruction}\n"
        "Allowed labels: {labels}\n\n"
        "Prompt:\n{x}\n\n"
        "{answer_prefix}"
    ),
)

# v1 for the HG-v2 route-choice task (plan section 14). The decision rule
# travels in the prompt so the correct action is machine-verifiable and does not
# depend on the model's own preferences; what is deliberately *withheld* is one
# variable the rule depends on, which the external source holds.
#
# `{constraints}` carries whatever the source has already told us -- empty on
# the ungrounded baseline. Keeping it in the template rather than appending to
# the case text means the grounded and ungrounded prompts differ in exactly one
# place, so a cost difference between them cannot come from reformatting.
ROUTE_CHOICE_V1 = PromptTemplate(
    version="route_choice.v1",
    instruction=(
        "Choose a route by applying the rule below. Reply with exactly one of "
        "the allowed labels and nothing else. Apply the rule even where your "
        "own judgment would differ."
    ),
    body=(
        "{instruction}\n\n"
        "Rule:\n{rule}\n\n"
        "Allowed labels: {labels}\n\n"
        "Routes:\n{x}\n\n"
        "Known traveller constraints:\n{constraints}\n\n"
        "{answer_prefix}"
    ),
)

# v1 for HG-v2 query synthesis (plan section 15). The model is asked what it
# needs to know, not what the answer is.
#
# Two properties are load-bearing. The output space is the *names* of available
# variables, so a synthesized query is directly comparable to the gold
# QuerySpec instead of needing free-text interpretation. And the instruction
# asks for the fewest variables that settle the decision, because the quantity
# under test is response cost: a prompt that merely said "what do you need"
# invites hedging, and a hedge is the failure mode gate G2 is meant to detect.
QUERY_SYNTH_V1 = PromptTemplate(
    version="query_synth.v1",
    instruction=(
        "You must choose between the options below by applying the rule, but "
        "some information is missing. Decide what to ask the traveller.\n"
        "Reply with the names of the facts you need, comma-separated, taken "
        "exactly from the list of available facts. Name as few as possible -- "
        "each one costs the traveller effort. If the rule already settles the "
        "choice with what you know, reply with the single word NONE."
    ),
    body=(
        "{instruction}\n\n"
        "Rule:\n{rule}\n\n"
        "Options:\n{x}\n\n"
        "Available facts you could ask about:\n{available}\n\n"
        "{answer_prefix}"
    ),
    answer_prefix="Facts needed:",
)

# E3's minimality interventions (improved plan section 9). Q1 and Q2 differ
# from `query_synth.v1` only in the instruction: the rule, the options and the
# list of available facts are byte-identical, so a cost difference between them
# is attributable to the instruction rather than to a reformatting.
#
# The previous run's failure mode is the target. The 3B model was sound on
# essentially every item and still overpaid, because its cost distribution was
# bimodal -- a one-variable query most of the time and a full escalation the
# rest. These two instructions attack that from opposite ends: v2 argues from
# decision-relevance, v3 states a hard budget.
QUERY_SYNTH_V2_MINIMAL = PromptTemplate(
    version="query_synth.v2_minimal",
    instruction=(
        "You must choose between the options below by applying the rule, but "
        "some information is missing. Decide what to ask the traveller.\n"
        "Ask only for information that can change the final choice. Do not "
        "request anything that becomes irrelevant once a narrower answer would "
        "already determine the choice.\n"
        "Reply with the names of the facts you need, comma-separated, taken "
        "exactly from the list of available facts. If the rule already settles "
        "the choice with what you know, reply with the single word NONE."
    ),
    body=(
        "{instruction}\n\n"
        "Rule:\n{rule}\n\n"
        "Options:\n{x}\n\n"
        "Available facts you could ask about:\n{available}\n\n"
        "{answer_prefix}"
    ),
    answer_prefix="Facts needed:",
)

QUERY_SYNTH_V3_BUDGET = PromptTemplate(
    version="query_synth.v3_budget",
    instruction=(
        "You must choose between the options below by applying the rule, but "
        "some information is missing. Decide what to ask the traveller.\n"
        "You may ask about at most one of the traveller's preferences. Choose "
        "the single fact that decides the choice.\n"
        "Reply with the name of that one fact, taken exactly from the list of "
        "available facts. If the rule already settles the choice with what you "
        "know, reply with the single word NONE."
    ),
    body=(
        "{instruction}\n\n"
        "Rule:\n{rule}\n\n"
        "Options:\n{x}\n\n"
        "Available facts you could ask about:\n{available}\n\n"
        "{answer_prefix}"
    ),
    answer_prefix="Facts needed:",
)

PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    t.version: t
    for t in (
        CLASSIFY_V1,
        POLICY_V1,
        PREFERENCE_V1,
        ROUTE_CHOICE_V1,
        QUERY_SYNTH_V1,
        QUERY_SYNTH_V2_MINIMAL,
        QUERY_SYNTH_V3_BUDGET,
    )
}
# Short aliases so dataset configs can say `prompt_version: classify.v1` or `v1`.
PROMPT_REGISTRY["v1"] = CLASSIFY_V1


def get_template(version: str) -> PromptTemplate:
    if version not in PROMPT_REGISTRY:
        raise KeyError(
            f"unknown prompt version {version!r}; registered: "
            f"{sorted(v for v in PROMPT_REGISTRY if '.' in v)}"
        )
    return PROMPT_REGISTRY[version]


def render_prompt(
    version: str,
    x: str,
    labels: Sequence[str],
    extra: Mapping[str, str] | None = None,
) -> str:
    return get_template(version).render(x, labels, extra=extra)
