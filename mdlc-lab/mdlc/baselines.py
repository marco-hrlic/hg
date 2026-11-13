"""Baselines: the simplest things that could possibly work. Every model must beat these."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from .text import clean_text

POSITIVE_WORDS = frozenset("""
good great amazing awesome excellent love loved lovely best fantastic wonderful delicious
perfect friendly nice tasty fresh helpful recommend recommended favorite favourite enjoy enjoyed
happy pleasant clean beautiful fast attentive outstanding incredible superb yummy glad impressed
""".split())

NEGATIVE_WORDS = frozenset("""
bad terrible awful horrible worst rude disgusting bland dirty disappointed disappointing poor
slow cold overpriced mediocre gross nasty stale greasy unprofessional waste annoying avoid
sick wrong broken unfriendly ignored hate hated meh dry burnt soggy lukewarm
""".split())

NEGATORS = frozenset("not no never isn't wasn't don't didn't doesn't aren't weren't won't can't couldn't hardly".split())

_WORD_RE = re.compile(r"[a-z']+")


def tokenize(text: object) -> list[str]:
    return _WORD_RE.findall(clean_text(text).lower())


class MajorityClassifier:
    """Always predicts the most common training label."""

    def fit(self, texts: Iterable, labels: Iterable[str]) -> "MajorityClassifier":
        self.label_ = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, texts: Iterable) -> list[str]:
        return [self.label_ for _ in texts]


class LexiconClassifier:
    """Counts positive and negative words. No learning involved.

    score = (#positive words) - (#negative words)
    score >  margin -> positive, score < -margin -> negative, otherwise neutral.

    With handle_negation=True, a sentiment word within `window` tokens after a
    negator ("not", "never", "didn't", ...) has its polarity flipped.
    """

    def __init__(self, margin: int = 0, handle_negation: bool = False, window: int = 3):
        self.margin = margin
        self.handle_negation = handle_negation
        self.window = window

    def fit(self, texts=None, labels=None) -> "LexiconClassifier":
        return self  # nothing to learn

    def score(self, text: object) -> int:
        tokens = tokenize(text)
        total, flip_left = 0, 0
        for tok in tokens:
            sign = 1 if tok in POSITIVE_WORDS else -1 if tok in NEGATIVE_WORDS else 0
            if self.handle_negation and flip_left > 0:
                sign = -sign
            total += sign
            if self.handle_negation:
                flip_left = self.window if tok in NEGATORS else max(flip_left - 1, 0)
        return total

    def predict(self, texts: Iterable) -> list[str]:
        out = []
        for t in texts:
            s = self.score(t)
            out.append("positive" if s > self.margin else "negative" if s < -self.margin else "neutral")
        return out
