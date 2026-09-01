"""A trained byte-level BPE tokenizer (spec section 3.1's 16k-vocab target).

Uses the `tokenizers` library (HF's Rust-backed BPE trainer) -- the natural
pairing with `transformers`, already a `gpu`-extra dependency, and what
`experiments/hg_native/train_tokenizer.py` actually runs against a real
corpus (TinyStories -- the corpus decision recorded in
[[hg-native-m0-project-state]]).

`BPETokenizer` implements the same `encode`/`decode`/`vocab_size`/`PAD`
surface as `training.batch.ByteTokenizer`, so swapping this in for that
placeholder (once a real vocab file exists) touches nothing else in the
training pipeline -- exactly what `ByteTokenizer`'s own docstring promises.
Needs `tokenizers` (the `gpu` extra); nothing else in this module imports
torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer as _HFTokenizer
from tokenizers.implementations import ByteLevelBPETokenizer

# Matches training.batch.ByteTokenizer's PAD/BOS/EOS convention so the two
# tokenizers are interchangeable from the training code's point of view.
PAD_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<s>", "</s>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


def train_bpe_tokenizer(texts: Iterable[str], vocab_size: int, save_path: str | Path, min_frequency: int = 2) -> None:
    """Train a byte-level BPE tokenizer over `texts` and save it to `save_path` (one .json file).

    Byte-level (not word-level): no `<unk>` token is needed, since every
    input byte is representable, matching `ByteTokenizer`'s own guarantee.
    """
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=SPECIAL_TOKENS)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))


class BPETokenizer:
    """Loads a tokenizer file written by `train_bpe_tokenizer`."""

    def __init__(self, path: str | Path):
        self._tok = _HFTokenizer.from_file(str(path))
        self.vocab_size = self._tok.get_vocab_size()
        self.PAD = self._tok.token_to_id(PAD_TOKEN)
        self.BOS = self._tok.token_to_id(BOS_TOKEN)
        self.EOS = self._tok.token_to_id(EOS_TOKEN)
        missing = [name for name, i in [(PAD_TOKEN, self.PAD), (BOS_TOKEN, self.BOS), (EOS_TOKEN, self.EOS)] if i is None]
        if missing:
            raise ValueError(f"{path}: tokenizer is missing required special token(s) {missing}")

    def encode(self, text: str) -> list[int]:
        return [self.BOS, *self._tok.encode(text).ids, self.EOS]

    def decode(self, ids: list[int]) -> str:
        specials = {self.PAD, self.BOS, self.EOS}
        return self._tok.decode([i for i in ids if i not in specials])
