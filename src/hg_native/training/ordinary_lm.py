"""Load and sample ordinary-LM text -- spec section 6's `ordinary_lm`
component of the data mix (currently TinyStories only, the corpus decision
recorded in [[hg-native-m0-project-state]] and what
`experiments/hg_native/train_tokenizer.py` trained the vocabulary on).

Needs `datasets` (the `gpu` extra); nothing else in `training/` imports it,
so a run that never passes `ordinary_lm_texts` to `TrainConfig` (the
default) never needs this module or a network connection.
"""

from __future__ import annotations

import random


def load_tinystories_texts(n_stories: int, seed: int = 0) -> list[str]:
    """Load a bounded, shuffled slice of TinyStories into memory.

    Non-streaming (`split=f"train[:{n_stories}]"`): the trainer re-samples
    from this same in-memory list every step (`sample_ordinary_lm_texts`),
    so a bounded, cached slice is what's actually wanted, not a live stream
    re-fetched per step.
    """
    if n_stories < 1:
        raise ValueError(f"n_stories must be >= 1, got {n_stories}")
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{n_stories}]")
    texts = list(dataset["text"])
    random.Random(seed).shuffle(texts)
    return texts


def sample_ordinary_lm_texts(texts: list[str], n: int, start_index: int) -> list[str]:
    """Deterministically cycle through `texts` (already shuffled once at load
    time by `load_tinystories_texts`, so cycling order is as good as a fresh
    random draw, without needing a second RNG here)."""
    if not texts:
        raise ValueError("texts must be non-empty")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return [texts[(start_index + i) % len(texts)] for i in range(n)]
