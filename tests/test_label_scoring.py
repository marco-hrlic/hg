"""Scoring-infrastructure regression for rank-scored labels (plan section 16).

The improved plan asks for a regression test per supported chat template, and
the reason is a bug this repository already paid for. `scoring_mode: rank`
scores each allowed label's continuation log-probability. A chat-templated
prompt ends at a token boundary, so the continuation the model expects is `"B"`;
a raw-text prompt ends mid-line after `"Answer:"`, so it is `" B"` -- a
different token id. Scoring only the spaced form measured the far tail of the
distribution on every chat-templated run: on Qwen2.5-1.5B-Instruct, -16.9 for
`" B"` against -0.001 for `"B"`. Every label then scores near-uniformly badly
and the argmax is noise, which is indistinguishable from a model that cannot do
the task.

These tests need a real tokenizer, so they skip where `transformers` or the
checkpoint is unavailable -- on the Spark, where the runs happen, they execute.
Set `HG_TEST_CHECKPOINTS` to a comma-separated list to cover more templates.
"""

from __future__ import annotations

import os

import pytest

CHECKPOINTS = [
    c.strip()
    for c in os.environ.get(
        "HG_TEST_CHECKPOINTS", "Qwen/Qwen2.5-3B-Instruct"
    ).split(",")
    if c.strip()
]


@pytest.fixture(scope="module")
def tokenizer_for():
    transformers = pytest.importorskip("transformers")

    cache: dict[str, object] = {}

    def load(checkpoint: str):
        if checkpoint not in cache:
            try:
                cache[checkpoint] = transformers.AutoTokenizer.from_pretrained(
                    checkpoint, local_files_only=True
                )
            except Exception as exc:  # noqa: BLE001 - absence is the skip reason
                pytest.skip(f"{checkpoint} is not available locally: {exc}")
        return cache[checkpoint]

    return load


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_chat_template_ends_at_a_token_boundary(checkpoint, tokenizer_for):
    """After a chat template, the bare label must be the cheaper continuation.

    This is the property `_label_forms` exists to cover. If a new template ever
    ends mid-token, scoring the bare form alone would be as wrong as scoring the
    spaced form alone was -- which is why both are scored and the better one
    taken.
    """
    from hg.inference.hf import TransformersBackend

    tok = tokenizer_for(checkpoint)
    if getattr(tok, "chat_template", None) is None:
        pytest.skip(f"{checkpoint} has no chat template")

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Choose A or B.\n\nAnswer:"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tok(prompt, return_tensors=None)["input_ids"]
    for label in ("A", "B"):
        bare = tok(label, add_special_tokens=False)["input_ids"]
        spaced = tok(" " + label, add_special_tokens=False)["input_ids"]
        assert bare, f"{checkpoint}: {label!r} tokenizes to nothing"
        assert bare != spaced, (
            f"{checkpoint}: {label!r} and {' ' + label!r} are the same token, "
            "so the bug this test guards cannot occur -- but a template change "
            "could reintroduce it, so the assertion is kept as documentation "
            "of the assumption"
        )
        # Concatenating the prompt and the bare label must not merge tokens at
        # the seam: if it did, the scored continuation would not be the label.
        joined = tok(prompt + label, add_special_tokens=False)["input_ids"]
        assert joined[: len(prompt_ids)] == prompt_ids, (
            f"{checkpoint}: appending {label!r} to the chat-templated prompt "
            "re-tokenizes the prompt, so a rank score would be reading a "
            "different sequence than the one that was scored"
        )
        assert joined[len(prompt_ids):] == bare

    assert set(TransformersBackend._label_forms("B")) == {"B", " B"}


@pytest.mark.parametrize("checkpoint", CHECKPOINTS)
def test_raw_prompt_needs_the_spaced_form(checkpoint, tokenizer_for):
    """Without a chat template the prompt ends mid-line, so " B" is the label.

    The mirror of the test above, and the reason `_label_forms` returns both
    rather than switching on whether a template was applied: the two paths need
    different forms, and taking the better-scoring one is correct for both.
    """
    tok = tokenizer_for(checkpoint)
    prompt = "Choose A or B.\n\nAnswer:"
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    spaced = tok(" B", add_special_tokens=False)["input_ids"]
    joined = tok(prompt + " B", add_special_tokens=False)["input_ids"]
    assert joined[: len(prompt_ids)] == prompt_ids
    assert joined[len(prompt_ids):] == spaced
