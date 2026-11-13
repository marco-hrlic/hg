"""Heuristic slice tags: cheap rules that flag reviews likely to be hard.

Slices are how we look *inside* an aggregate metric. A model with 85% accuracy
overall can still be at 50% on reviews that use negation.
"""

from __future__ import annotations

import re

from .text import clean_text

POSITIVE_HINTS = re.compile(
    r"\b(great|amazing|awesome|love|loved|excellent|best|delicious|perfect|fantastic|wonderful|friendly)\b",
    re.I,
)
NEGATIVE_HINTS = re.compile(
    r"\b(terrible|awful|horrible|worst|rude|disgusting|bland|dirty|disappointed|disappointing|never again|waste)\b",
    re.I,
)

SLICE_RULES: dict[str, re.Pattern] = {
    "negation": re.compile(r"\b(not|no|never|nothing|nobody|hardly|none|\w+n't)\b", re.I),
    "contrast": re.compile(r"\b(but|however|although|though|except|yet|on the other hand)\b", re.I),
    "comparison": re.compile(r"\b(better than|worse than|compared to|not as good)\b", re.I),
    "shouting": re.compile(r"(!!+|\b[A-Z]{4,}\b.*\b[A-Z]{4,}\b)"),
    "sarcasm_cue": re.compile(r"\b(yeah right|thanks a lot|oh great|so much for|what a joke|gee|sure,)\b", re.I),
}

SLICE_NAMES = ("short", "long", *SLICE_RULES, "mixed_signals")


def tag_slices(text: object) -> list[str]:
    """Return the list of slice names that apply to a review."""
    t = clean_text(text)
    n_words = len(t.split())
    tags = []
    if n_words <= 15:
        tags.append("short")
    if n_words >= 200:
        tags.append("long")
    for name, rx in SLICE_RULES.items():
        if rx.search(t):
            tags.append(name)
    if POSITIVE_HINTS.search(t) and NEGATIVE_HINTS.search(t):
        tags.append("mixed_signals")
    return tags
