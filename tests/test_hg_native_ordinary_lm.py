"""Ordinary-LM text sampling. `sample_ordinary_lm_texts` is pure Python and
runs anywhere; `load_tinystories_texts` needs `datasets` and a network
connection, so it's exercised on the Spark, not asserted on here beyond a
skip-guarded smoke check.
"""

from __future__ import annotations

import pytest

from hg_native.training.ordinary_lm import sample_ordinary_lm_texts


def test_rejects_empty_texts():
    with pytest.raises(ValueError, match="non-empty"):
        sample_ordinary_lm_texts([], n=1, start_index=0)


def test_rejects_non_positive_n():
    with pytest.raises(ValueError, match="n must be"):
        sample_ordinary_lm_texts(["a"], n=0, start_index=0)


def test_cycles_through_texts_deterministically():
    texts = ["a", "b", "c"]
    assert sample_ordinary_lm_texts(texts, n=5, start_index=0) == ["a", "b", "c", "a", "b"]


def test_start_index_advances_the_cycle():
    texts = ["a", "b", "c"]
    assert sample_ordinary_lm_texts(texts, n=3, start_index=1) == ["b", "c", "a"]


def test_deterministic_given_same_arguments():
    texts = ["x", "y", "z", "w"]
    a = sample_ordinary_lm_texts(texts, n=10, start_index=3)
    b = sample_ordinary_lm_texts(texts, n=10, start_index=3)
    assert a == b


def test_load_tinystories_rejects_non_positive_n_stories():
    from hg_native.training.ordinary_lm import load_tinystories_texts

    with pytest.raises(ValueError, match="n_stories"):
        load_tinystories_texts(0)
