"""The LM query synthesizer (plan sections 15 and 17, baseline B4).

Driven by a stub backend that returns scripted text, so every parsing and
fallback path is covered without a GPU. What a real model actually says is a
separate question, answered by running Phase 2 on a real checkpoint; these tests
fix what the *system* does with whatever it says.
"""

from __future__ import annotations

import pytest

from hg.grounding.route_choice import (
    PREFERENCE_VARIABLES,
    RouteChoiceSpec,
    generate_route_choice_items,
)
from hg.inference.base import GenerationSettings, InferenceItem, Prediction
from hg.inference.prompts import get_template
from hg.queries.lm_query_synthesizer import (
    ON_FAILURE_GENERIC,
    ON_FAILURE_NONE,
    QUERY_SYNTH_PROMPT_VERSION,
    LMQuerySynthesizer,
    extract_variables,
)


class ScriptedBackend:
    """Returns fixed text per item, in order. Records the prompts it saw."""

    name = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []
        self.settings: list[GenerationSettings] = []

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def predict(self, items, settings):
        self.settings.append(settings)
        out = []
        for k, item in enumerate(items):
            self.prompts.append(item.prompt)
            text = self.replies[k % len(self.replies)]
            out.append(
                Prediction(item_id=item.item_id, label=None, raw_output=text)
            )
        return out


def items(n=6):
    return generate_route_choice_items(RouteChoiceSpec(n_items=n, seed=0))


# ------------------------------------------------------------------ parsing ---
@pytest.mark.parametrize(
    "text,expected",
    [
        ("mountain_roads", ["mountain_roads"]),
        ("mountain roads", ["mountain_roads"]),
        ("a mountain road", ["mountain_roads"]),
        ("toll_roads, ferry_crossings", ["toll_roads", "ferry_crossings"]),
        ("ferry_crossings, toll_roads", ["toll_roads", "ferry_crossings"]),
        ("I need to know about the ferry crossing", ["ferry_crossings"]),
        ("night driving", ["night_driving"]),
        ("unpaved roads", ["unpaved_roads"]),
        ("NONE", []),
        ("", []),
        ("the weather", []),
    ],
)
def test_extract_variables(text, expected):
    assert extract_variables(text, PREFERENCE_VARIABLES) == expected


def test_extraction_is_order_independent():
    """Two models naming the same set must produce the same query, and
    therefore the same cost."""
    a = extract_variables("toll_roads, mountain_roads", PREFERENCE_VARIABLES)
    b = extract_variables("mountain_roads, toll_roads", PREFERENCE_VARIABLES)
    assert a == b


def test_only_the_first_line_is_read():
    """Small instruct models continue past the answer. A later sentence
    mentioning every option would otherwise read as asking for all of them."""
    text = "mountain_roads\nI do not need toll_roads, ferry_crossings or night_driving."
    assert extract_variables(text, PREFERENCE_VARIABLES) == ["mountain_roads"]


def test_extraction_is_restricted_to_the_offered_variables():
    offered = ("toll_roads", "ferry_crossings")
    assert extract_variables("mountain_roads, toll_roads", offered) == ["toll_roads"]


# ---------------------------------------------------------------- the prompt ---
def test_the_synthesis_prompt_offers_the_variables_and_hides_the_answers():
    item = items(1)[0]
    syn = LMQuerySynthesizer(ScriptedBackend(["NONE"]))
    prompt = syn.prompt_for(item)

    for name in item.variables:
        assert name in prompt
    # It must not disclose any acceptance -- that is what it is asking about.
    assert "is acceptable" not in prompt.split("Rule:")[0]
    assert "NONE" in prompt
    # And it must say that asking costs something, or hedging is free.
    assert "as few as possible" in prompt or "few as possible" in prompt


def test_the_synthesis_prompt_is_a_registered_version():
    """Prompt versions are a contract: old rows have to stay interpretable."""
    template = get_template(QUERY_SYNTH_PROMPT_VERSION)
    assert template.version == QUERY_SYNTH_PROMPT_VERSION


# ------------------------------------------------------------------ behaviour ---
def test_a_named_variable_becomes_a_one_bit_query():
    pool = items(4)
    replies = [pool[0].requirement] * 4
    syn = LMQuerySynthesizer(ScriptedBackend(replies))
    queries = syn.synthesize_batch(pool)

    assert all(q is not None for q in queries)
    assert all(q.response_bits == 1.0 for q in queries)
    assert syn.summary()["parse_ok_rate"] == 1.0
    assert syn.summary()["fallback_rate"] == 0.0


def test_naming_several_variables_costs_several_bits():
    """A hedge resolves the decision and overpays, which is the failure mode
    gate G2 exists to detect."""
    pool = items(2)
    syn = LMQuerySynthesizer(
        ScriptedBackend(["mountain_roads, toll_roads, ferry_crossings"])
    )
    queries = syn.synthesize_batch(pool)
    assert all(q.response_bits == 3.0 for q in queries)
    assert syn.summary()["mean_variables_named"] == 3.0


def test_none_means_no_query_and_no_cost():
    """On an item the prompt already settles, the right query is no query."""
    pool = items(3)
    syn = LMQuerySynthesizer(ScriptedBackend(["NONE"]))
    assert syn.synthesize_batch(pool) == [None, None, None]
    summary = syn.summary()
    assert summary["said_none_rate"] == 1.0
    assert summary["parse_ok_rate"] == 1.0
    assert summary["fallback_rate"] == 0.0


def test_an_unparseable_answer_falls_back_to_generic_escalation():
    """The conservative default, and deliberately unflattering: asking nothing
    would let a model that failed to form a query score as cheap."""
    pool = items(3)
    syn = LMQuerySynthesizer(ScriptedBackend(["I'm not sure what you mean."]))
    queries = syn.synthesize_batch(pool)

    assert all(q is not None for q in queries)
    assert all(q.response_bits == float(len(PREFERENCE_VARIABLES)) for q in queries)
    summary = syn.summary()
    assert summary["parse_ok_rate"] == 0.0
    assert summary["fallback_rate"] == 1.0


def test_the_no_fallback_policy_asks_nothing_instead():
    pool = items(2)
    syn = LMQuerySynthesizer(
        ScriptedBackend(["no idea"]), on_parse_failure=ON_FAILURE_NONE
    )
    assert syn.synthesize_batch(pool) == [None, None]
    assert syn.summary()["fallback_rate"] == 0.0
    assert syn.summary()["parse_ok_rate"] == 0.0


def test_an_unknown_failure_policy_is_rejected():
    with pytest.raises(ValueError, match="on_parse_failure"):
        LMQuerySynthesizer(ScriptedBackend(["x"]), on_parse_failure="improvise")


def test_max_variables_rejects_a_query_that_names_everything():
    """A model naming every variable has produced generic escalation with extra
    steps; letting it through as 'targeted' would misattribute the cost."""
    pool = items(2)
    syn = LMQuerySynthesizer(
        ScriptedBackend([", ".join(PREFERENCE_VARIABLES)]), max_variables=3
    )
    queries = syn.synthesize_batch(pool)
    # Refused, so it falls back rather than being counted as a targeted query.
    assert all(q.response_bits == float(len(PREFERENCE_VARIABLES)) for q in queries)
    assert syn.summary()["fallback_rate"] == 1.0


def test_max_variables_allows_a_query_within_the_cap():
    pool = items(2)
    syn = LMQuerySynthesizer(
        ScriptedBackend(["mountain_roads, toll_roads"]), max_variables=3
    )
    assert all(q.response_bits == 2.0 for q in syn.synthesize_batch(pool))
    assert syn.summary()["fallback_rate"] == 0.0


def test_synthesis_is_one_backend_call_for_many_items():
    """Per-item calls on a real model waste most of the GPU."""
    backend = ScriptedBackend(["NONE"])
    syn = LMQuerySynthesizer(backend)
    syn.synthesize_batch(items(20))
    assert len(backend.settings) == 1
    assert len(backend.prompts) == 20


def test_synthesis_defaults_to_greedy_decoding():
    backend = ScriptedBackend(["NONE"])
    LMQuerySynthesizer(backend).synthesize_batch(items(2))
    assert backend.settings[0].greedy


def test_the_single_item_path_matches_the_batch_path():
    pool = items(1)
    a = LMQuerySynthesizer(ScriptedBackend(["toll_roads"])).synthesize(pool[0])
    b = LMQuerySynthesizer(ScriptedBackend(["toll_roads"])).synthesize_batch(pool)[0]
    assert a.variable_names == b.variable_names


def test_an_empty_batch_is_not_a_backend_call():
    backend = ScriptedBackend(["NONE"])
    assert LMQuerySynthesizer(backend).synthesize_batch([]) == []
    assert backend.settings == []


def test_summary_is_empty_before_any_synthesis():
    assert LMQuerySynthesizer(ScriptedBackend(["x"])).summary() == {"n": 0}


def test_summary_records_the_provenance_a_result_needs():
    syn = LMQuerySynthesizer(ScriptedBackend(["mountain_roads"]))
    syn.synthesize_batch(items(4))
    summary = syn.summary()
    for key in ("parse_ok_rate", "fallback_rate", "said_none_rate",
                "mean_variables_named", "prompt_version", "on_parse_failure"):
        assert key in summary
    assert summary["prompt_version"] == QUERY_SYNTH_PROMPT_VERSION
    assert summary["on_parse_failure"] == ON_FAILURE_GENERIC
    assert len(summary["examples"]) == 4
