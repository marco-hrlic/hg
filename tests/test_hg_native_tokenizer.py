"""BPETokenizer training/loading -- needs `tokenizers`; skips here if absent."""

from __future__ import annotations

import pytest

pytest.importorskip("tokenizers")

from hg_native.data.tokenizer import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, BPETokenizer, train_bpe_tokenizer

TOY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "A.cost = 37\nB.cost = 29\nRULE: choose B iff B is cheaper AND h1 = YES",
    "once upon a time there was a small robot who liked to help",
] * 20  # repeated so the trainer has enough frequency signal to merge anything


def test_train_and_load_round_trips_text(tmp_path):
    path = tmp_path / "tok.json"
    train_bpe_tokenizer(TOY_CORPUS, vocab_size=300, save_path=path)
    tok = BPETokenizer(path)

    text = "the quick brown fox"
    ids = tok.encode(text)
    assert ids[0] == tok.BOS
    assert ids[-1] == tok.EOS
    assert tok.decode(ids) == text


def test_vocab_size_matches_requested_budget(tmp_path):
    path = tmp_path / "tok.json"
    train_bpe_tokenizer(TOY_CORPUS, vocab_size=300, save_path=path)
    tok = BPETokenizer(path)
    # A byte-level BPE trainer can't always hit the target exactly on a tiny
    # corpus (it stops early once no more merges help), but must not exceed it.
    assert tok.vocab_size <= 300


def test_special_tokens_get_distinct_ids(tmp_path):
    path = tmp_path / "tok.json"
    train_bpe_tokenizer(TOY_CORPUS, vocab_size=300, save_path=path)
    tok = BPETokenizer(path)
    assert len({tok.PAD, tok.BOS, tok.EOS}) == 3


def test_byte_level_handles_unseen_characters(tmp_path):
    # Byte-level BPE has no <unk>: any UTF-8 text round-trips, even
    # characters absent from the training corpus.
    path = tmp_path / "tok.json"
    train_bpe_tokenizer(TOY_CORPUS, vocab_size=300, save_path=path)
    tok = BPETokenizer(path)
    text = "éèê中文\U0001f600"  # accented Latin, CJK, emoji
    ids = tok.encode(text)
    assert tok.decode(ids) == text


def test_missing_special_token_raises(tmp_path):
    from tokenizers import Tokenizer as HFTokenizer
    from tokenizers.models import BPE

    path = tmp_path / "no_specials.json"
    bare = HFTokenizer(BPE(unk_token=None))
    bare.save(str(path))
    with pytest.raises(ValueError, match="missing required special token"):
        BPETokenizer(path)


def test_load_nonexistent_file_raises():
    with pytest.raises(Exception):
        BPETokenizer("/nonexistent/path/tok.json")
