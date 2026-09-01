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

PROMPT_REGISTRY: dict[str, PromptTemplate] = {
    t.version: t
    for t in (CLASSIFY_V1, POLICY_V1, PREFERENCE_V1, ROUTE_CHOICE_V1)
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
