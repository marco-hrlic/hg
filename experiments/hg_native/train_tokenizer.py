#!/usr/bin/env python3
"""Train the M0 BPE tokenizer on TinyStories.

TinyStories (a small HF-hosted dataset of simple GPT-generated short
stories, built for training tiny LMs) is the corpus decision recorded in
[[hg-native-m0-project-state]] -- it matches spec section 6's "does not
require encyclopedic knowledge" goal, is public (no HF token needed, unlike
the gated checkpoints noted in [[dgx-spark-access]]), and small enough to
stream and tokenize quickly.

Usage:
    python experiments/hg_native/train_tokenizer.py \
        --vocab-size 16000 --n-stories 100000 \
        --output experiments/hg_native/tokenizer/hg_native_bpe.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hg_native.data.tokenizer import BPETokenizer, train_bpe_tokenizer  # noqa: E402


def iter_tinystories(n_stories: int):
    from datasets import load_dataset

    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, row in enumerate(dataset):
        if i >= n_stories:
            break
        yield row["text"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--n-stories", type=int, default=100_000)
    parser.add_argument("--output", default="experiments/hg_native/tokenizer/hg_native_bpe.json")
    args = parser.parse_args()

    print(f"streaming {args.n_stories} TinyStories examples...", file=sys.stderr)
    texts = list(iter_tinystories(args.n_stories))
    print(f"training BPE (vocab_size={args.vocab_size}) on {len(texts)} stories...", file=sys.stderr)
    train_bpe_tokenizer(texts, vocab_size=args.vocab_size, save_path=args.output)

    tokenizer = BPETokenizer(args.output)
    sample = texts[0][:200]
    ids = tokenizer.encode(sample)
    print(f"wrote {args.output}", file=sys.stderr)
    print(f"vocab_size={tokenizer.vocab_size}", file=sys.stderr)
    print(f"sample text: {sample!r}", file=sys.stderr)
    print(f"encoded ({len(ids)} tokens): {ids[:30]}{'...' if len(ids) > 30 else ''}", file=sys.stderr)
    print(f"round-trip: {tokenizer.decode(ids)!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
