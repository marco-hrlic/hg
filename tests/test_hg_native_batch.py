"""`ByteTokenizer` + `serialize_episode`/`collate` -- torch-free, runs anywhere."""

from __future__ import annotations

import pytest

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.capability_refinement import generate_pair
from hg_native.data.sequential_quotient import generate_trajectory
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.training.actions import ACTIONS
from hg_native.training.batch import ByteTokenizer, collate, collate_text, render, serialize_episode, serialize_text

ALL_FAMILY_EPISODES = [
    gen_static_hidden(0, "one_bit_deficit", 0),
    gen_static_hidden(0, "no_deficit", 0),
    gen_accessibility(0, "reordered_form", 0),
    generate_trajectory(0, 0, 3)[0],
    generate_pair(0, 0)["z_t0_C1"],
]


# --- ByteTokenizer -----------------------------------------------------


def test_byte_tokenizer_round_trips_ascii():
    tok = ByteTokenizer()
    text = "TASK STATIC-CHOICE\nA.cost = 37"
    ids = tok.encode(text)
    assert ids[0] == tok.BOS
    assert ids[-1] == tok.EOS
    assert tok.decode(ids) == text


def test_byte_tokenizer_vocab_size_covers_all_specials_and_bytes():
    assert ByteTokenizer.vocab_size == 259
    assert ByteTokenizer.PAD < ByteTokenizer.vocab_size
    assert ByteTokenizer.BOS < ByteTokenizer.vocab_size
    assert ByteTokenizer.EOS < ByteTokenizer.vocab_size


# --- render dispatch -----------------------------------------------------


def test_render_dispatches_by_family():
    for ep in ALL_FAMILY_EPISODES:
        text = render(ep)
        assert isinstance(text, str) and text


def test_render_rejects_unknown_family():
    ep = gen_static_hidden(0, "no_deficit", 0)
    bad = ep.__class__(**{**ep.to_dict(), "family": "not_a_family"})
    with pytest.raises(ValueError, match="no render_text registered"):
        render(bad)


# --- serialize_episode -----------------------------------------------------


@pytest.mark.parametrize("episode", ALL_FAMILY_EPISODES)
def test_serialize_episode_shapes(episode):
    tok = ByteTokenizer()
    max_seq_len = 64
    out = serialize_episode(episode, tok, max_seq_len)
    assert len(out["input_ids"]) == max_seq_len
    assert len(out["attention_mask"]) == max_seq_len
    assert out["input_ids"][0] == tok.BOS
    assert out["input_ids"][out["decision_position"]] == tok.EOS
    assert all(m == 1 for m in out["attention_mask"][: out["decision_position"] + 1])
    assert all(m == 0 for m in out["attention_mask"][out["decision_position"] + 1 :])
    assert all(t == tok.PAD for t in out["input_ids"][out["decision_position"] + 1 :])
    assert 0 <= out["action_id"] < len(ACTIONS)


def test_serialize_episode_truncates_and_keeps_eos():
    ep = gen_static_hidden(0, "one_bit_deficit", 0)
    tok = ByteTokenizer()
    full = serialize_episode(ep, tok, max_seq_len=200)
    real_len = full["decision_position"] + 1
    truncated = serialize_episode(ep, tok, max_seq_len=real_len - 5)
    assert len(truncated["input_ids"]) == real_len - 5
    assert truncated["input_ids"][0] == tok.BOS
    assert truncated["input_ids"][-1] == tok.EOS
    assert truncated["decision_position"] == real_len - 5 - 1


def test_serialize_episode_rejects_too_small_max_seq_len():
    ep = gen_static_hidden(0, "no_deficit", 0)
    with pytest.raises(ValueError, match="max_seq_len"):
        serialize_episode(ep, ByteTokenizer(), max_seq_len=1)


def test_serialize_episode_targets_present_for_every_family():
    tok = ByteTokenizer()
    for ep in ALL_FAMILY_EPISODES:
        out = serialize_episode(ep, tok, max_seq_len=64)
        t = out["targets"]
        assert set(t) == {
            "decision_label",
            "decision_defined",
            "distortion_vec",
            "distortion_mask",
            "info_deficit",
            "accessibility",
            "accessibility_defined",
            "route_label",
            "route_defined",
        }
        assert len(t["distortion_vec"]) == len(t["distortion_mask"]) == len(ACTIONS)


# --- collate -----------------------------------------------------


def test_collate_rejects_empty():
    with pytest.raises(ValueError, match="at least one example"):
        collate([])


def test_serialize_text_shape_and_no_targets():
    tok = ByteTokenizer()
    out = serialize_text("Once upon a time there was a small robot.", tok, max_seq_len=32)
    assert set(out) == {"input_ids", "attention_mask"}
    assert len(out["input_ids"]) == 32
    assert out["input_ids"][0] == tok.BOS


def test_serialize_text_truncates_and_pads_like_episodes_do():
    tok = ByteTokenizer()
    long_text = "word " * 100
    out = serialize_text(long_text, tok, max_seq_len=16)
    assert len(out["input_ids"]) == 16
    assert out["input_ids"][-1] in (tok.PAD, tok.EOS)  # either padded, or EOS lands exactly at the end
    assert out["input_ids"][0] == tok.BOS


def test_collate_text_rejects_empty():
    with pytest.raises(ValueError, match="at least one example"):
        collate_text([])


def test_collate_text_stacks_fields():
    tok = ByteTokenizer()
    examples = [serialize_text(t, tok, max_seq_len=16) for t in ["hello", "goodbye", "a longer sentence here"]]
    batch = collate_text(examples)
    assert set(batch) == {"input_ids", "attention_mask"}
    assert len(batch["input_ids"]) == 3
    assert all(len(row) == 16 for row in batch["input_ids"])


def test_collate_stacks_every_field():
    tok = ByteTokenizer()
    examples = [serialize_episode(ep, tok, max_seq_len=64) for ep in ALL_FAMILY_EPISODES]
    batch = collate(examples)
    n = len(ALL_FAMILY_EPISODES)
    assert len(batch["input_ids"]) == n
    assert len(batch["decision_position"]) == n
    assert len(batch["episode_id"]) == n
    assert batch["episode_id"] == [ep.episode_id for ep in ALL_FAMILY_EPISODES]
    for key in ("decision_label", "distortion_vec", "info_deficit", "route_label"):
        assert len(batch["targets"][key]) == n
