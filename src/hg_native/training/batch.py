"""Turn `Episode`s into model-ready batches (spec section 26 checklist item 10).

Deliberately torch-free (plain lists, ints, floats, bools): every field here
converts to a tensor with a one-line `torch.tensor(...)` in the trainer, but
keeping this layer torch-free means it's testable on any machine and the
tokenizer/serialization logic can be checked independently of any model code.

`ByteTokenizer` is an explicit **placeholder**, not the spec section 3.1
16k-vocabulary BPE tokenizer -- that needs a real training corpus decision
(spec section 6) which hasn't been made yet. Byte-level tokenization needs
no training and no corpus, so batches can be built and the loss registry
exercised today; swapping in a trained BPE later only touches `encode`/
`decode`/`vocab_size`, not `serialize_episode`/`collate`.
"""

from __future__ import annotations

from typing import Any, Protocol

from hg_native.data import accessibility, capability_refinement, sequential_quotient, static_hidden
from hg_native.data.schema import Episode

from . import targets
from .actions import ACTIONS, action_id


class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class ByteTokenizer:
    """Raw UTF-8 bytes + BOS/EOS/PAD. No training, no corpus, no vocab file."""

    PAD = 256
    BOS = 257
    EOS = 258
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return [self.BOS, *text.encode("utf-8"), self.EOS]

    def decode(self, ids: list[int]) -> str:
        raw = bytes(i for i in ids if i < 256)
        return raw.decode("utf-8", errors="replace")


# One render_text per family (spec section 5's worked text-block formats).
_RENDERERS = {
    "static_hidden": static_hidden.render_text,
    "accessibility": accessibility.render_text,
    "sequential_quotient": sequential_quotient.render_text,
    "capability_refinement": capability_refinement.render_text,
}


def render(episode: Episode) -> str:
    try:
        renderer = _RENDERERS[episode.family]
    except KeyError:
        raise ValueError(f"no render_text registered for family {episode.family!r}") from None
    return renderer(episode)


def serialize_episode(episode: Episode, tokenizer: Tokenizer, max_seq_len: int) -> dict[str, Any]:
    """One episode -> fixed-length token ids, attention mask, and every head's target.

    The decision position is always the EOS token: every family's rendered
    text ends right at the "DECIDE ..." line, so reading `h_t` off the final
    real token is exactly "after having seen the whole prompt" (spec section
    3.2's "at selected decision positions").
    """
    if max_seq_len < 2:
        raise ValueError(f"max_seq_len must allow at least BOS+EOS, got {max_seq_len}")

    ids = tokenizer.encode(render(episode))
    if len(ids) > max_seq_len:
        eos = ids[-1]
        ids = [*ids[: max_seq_len - 1], eos]  # keep BOS + truncate the middle, EOS always present

    decision_position = len(ids) - 1  # index of EOS, before padding
    pad_id = getattr(tokenizer, "PAD", 0)
    n_pad = max_seq_len - len(ids)
    input_ids = ids + [pad_id] * n_pad
    attention_mask = [1] * len(ids) + [0] * n_pad

    decision_label, decision_defined = targets.decision_target(episode)
    distortion_vec, distortion_mask = targets.distortion_target(episode)
    accessibility_value, accessibility_defined = targets.accessibility_target(episode)
    route_label, route_defined = targets.route_target(episode)

    return {
        "episode_id": episode.episode_id,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decision_position": decision_position,
        "action_id": action_id(episode.correct_action),
        "targets": {
            "decision_label": decision_label if decision_defined else 0,
            "decision_defined": decision_defined,
            "distortion_vec": distortion_vec,
            "distortion_mask": distortion_mask,
            "info_deficit": targets.info_deficit_target(episode),
            "accessibility": accessibility_value,
            "accessibility_defined": accessibility_defined,
            "route_label": route_label if route_defined else 0,
            "route_defined": route_defined,
        },
    }


def collate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of `serialize_episode` outputs into batch-shaped lists.

    Still torch-free: `{k: torch.tensor(v) for k, v in out.items()}` (and the
    nested `targets` dict likewise) is the entire remaining step for a
    trainer, left there rather than here so this module has no torch import.
    """
    if not examples:
        raise ValueError("collate requires at least one example")
    return {
        "episode_id": [ex["episode_id"] for ex in examples],
        "input_ids": [ex["input_ids"] for ex in examples],
        "attention_mask": [ex["attention_mask"] for ex in examples],
        "decision_position": [ex["decision_position"] for ex in examples],
        "action_id": [ex["action_id"] for ex in examples],
        "targets": {
            key: [ex["targets"][key] for ex in examples] for key in examples[0]["targets"]
        },
    }
