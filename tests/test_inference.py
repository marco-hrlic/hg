"""Prompting, canonicalization, and the simulator's error structure."""

from __future__ import annotations

import pytest

from hg.inference.base import GenerationSettings, InferenceItem, available_backends, get_backend
from hg.inference.canonicalize import canonicalize, normalize, parse_failure_report
from hg.inference.prompts import get_template, render_prompt
from hg.inference.simulator import META_CORE, META_DIFFICULTY, META_TRUE_LABEL

LABELS = ["not_harmful", "harmful"]
SYN = {"yes": "harmful", "no": "not_harmful"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("harmful", "harmful"),
        ("Harmful", "harmful"),
        ("  HARMFUL  ", "harmful"),
        ("**harmful**", "harmful"),
        ("Answer: harmful", "harmful"),
        ("The answer is harmful.", "harmful"),
        ("not_harmful", "not_harmful"),
        ("not harmful", "not_harmful"),
        ("yes", "harmful"),
        ("no", "not_harmful"),
        ("harmful\nBecause it gives uplift.", "harmful"),
    ],
)
def test_canonicalize_accepts_common_surface_forms(raw, expected):
    assert canonicalize(raw, LABELS, SYN).label == expected


def test_longest_label_wins_over_a_substring():
    """'not harmful' must not be read as 'harmful'."""
    assert canonicalize("not harmful", LABELS).label == "not_harmful"


def test_unparseable_output_is_flagged_not_guessed():
    res = canonicalize("I'd rather not say", LABELS, SYN)
    assert res.label is None
    assert res.parse_ok is False


def test_synonyms_must_target_the_declared_label_space():
    with pytest.raises(ValueError, match="outside the label space"):
        canonicalize("yes", LABELS, {"yes": "maybe"})


def test_normalize_strips_accents_and_punctuation():
    assert normalize("Hármful!!") == "harmful"


def test_parse_failure_report_is_per_family():
    rows = [("qwen", True), ("qwen", False), ("gemma", True), ("gemma", True)]
    report = parse_failure_report(rows)
    assert report["qwen"] == pytest.approx(0.5)
    assert report["gemma"] == 0.0


def test_prompt_template_is_versioned_and_shared_across_families():
    text = render_prompt("classify.v1", "some input", LABELS)
    assert "not_harmful, harmful" in text
    assert "some input" in text
    assert get_template("classify.v1").version == "classify.v1"


def test_unknown_prompt_version_raises():
    with pytest.raises(KeyError, match="unknown prompt version"):
        get_template("classify.v99")


def test_policy_template_carries_the_policy_in_the_prompt():
    text = render_prompt("policy.v1", "a case", ["allow", "refuse"], extra={"policy": "RULE ONE"})
    assert "RULE ONE" in text


def test_sampling_without_a_seed_is_rejected():
    """Irreproducible decoding would silently inflate ensemble disagreement."""
    with pytest.raises(ValueError, match="irreproducible"):
        GenerationSettings(temperature=0.7, seed=None)


def test_greedy_is_the_default():
    assert GenerationSettings().greedy


def test_simulator_is_always_registered():
    assert "simulator" in available_backends()


def items(n=200, core=False, difficulty=0.0):
    return [
        InferenceItem(
            item_id=f"i{k}",
            prompt="p",
            labels=LABELS,
            meta={
                META_DIFFICULTY: difficulty,
                META_CORE: core,
                META_TRUE_LABEL: "harmful",
            },
        )
        for k in range(n)
    ]


def run(lineage, split, seed=0, world_seed=0, its=None, **params):
    backend = get_backend(
        "simulator", lineage=lineage, train_split=split, adapter_seed=seed,
        world_seed=world_seed, params=params,
    )
    return backend.predict(its or items(), GenerationSettings())


def test_simulator_is_deterministic():
    a = run("qwen", "shared")
    b = run("qwen", "shared")
    assert [p.label for p in a] == [p.label for p in b]


def test_simulator_preserves_item_order():
    its = items(50)
    preds = run("qwen", "shared", its=its)
    assert [p.item_id for p in preds] == [i.item_id for i in its]


def test_core_items_are_failed_by_every_lineage():
    """The core is the information-limited floor: diversity must not remove it."""
    core_items = items(300, core=True)
    for lineage in ("qwen", "mistral", "gemma"):
        preds = run(lineage, f"{lineage}-split", its=core_items)
        wrong = sum(1 for p in preds if p.label != "harmful")
        assert wrong / len(preds) > 0.9, lineage


def test_shared_lineage_produces_more_correlated_errors_than_distinct_lineages():
    """The effect the whole experiment is built to measure, in the fixture."""
    its = items(2000, difficulty=0.0)

    def errors(lineage, split):
        return [p.label != "harmful" for p in run(lineage, split, its=its)]

    same_a = errors("qwen", "shared")
    same_b = [
        p.label != "harmful"
        for p in run("qwen", "shared", seed=1, its=its)
    ]
    diff = errors("gemma", "gemma-split")

    def agreement(x, y):
        return sum(1 for a, b in zip(x, y) if a and b) / max(sum(x), 1)

    assert agreement(same_a, same_b) > agreement(same_a, diff)


def test_parse_failure_rate_is_honoured():
    preds = run("qwen", "shared", its=items(1000), parse_failure_rate=0.1)
    rate = sum(1 for p in preds if not p.parse_ok) / len(preds)
    assert 0.05 < rate < 0.15


def test_unknown_simulator_option_rejected():
    with pytest.raises(ValueError, match="unknown simulator options"):
        run("qwen", "shared", nonsense=1)


def test_label_forms_cover_both_prompt_styles():
    """A chat-templated prompt ends at a token boundary, so the continuation is
    "B"; a raw-text prompt ends mid-line, so it is " B" -- a different token.

    Scoring only the spaced form measures the far tail of the distribution
    whenever a chat template is applied, which is the default path. Observed on
    Qwen2.5-1.5B-Instruct: -16.9 for " B" against -0.001 for "B".
    """
    from hg.inference.hf import TransformersBackend

    forms = TransformersBackend._label_forms("B")
    assert "B" in forms
    assert " B" in forms
