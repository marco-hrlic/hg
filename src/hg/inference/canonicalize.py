"""Mapping raw model output onto the task's label space.

This is the other half of the "inference-format artifacts" mitigation
(protocol section 18). Families differ in how chatty they are, how they
capitalize, and whether they wrap answers in punctuation or markdown. If that
variation reaches the error matrix, D_E measures formatting diversity.

The rules are deliberately conservative and applied in a fixed order:

1. exact match after normalization;
2. the label appears as a whole word in the output;
3. a configured synonym maps to a label;
4. otherwise the row is marked `parse_ok=False`.

Unparsed rows are *not* silently scored wrong. They are counted and surfaced,
because a family with a high parse-failure rate looks like a weak model and
would bias the "unequal base capability" adjustment in section 18.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Mapping, Sequence

_PUNCT = re.compile(r"[^\w\s-]+", flags=re.UNICODE)
_WS = re.compile(r"\s+")


@dataclass(slots=True)
class CanonicalizeResult:
    label: str | None
    parse_ok: bool
    # How the match was made, recorded for auditing parser behaviour per family.
    rule: str


def normalize(text: str) -> str:
    """Lowercase, strip accents, drop punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("_", " ")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip().lower()


def _strip_scaffolding(text: str) -> str:
    """Remove the answer prefix and common chat scaffolding before matching."""
    out = text.strip()
    # Drop a leading "Answer:"-style prefix, and any assistant role marker.
    out = re.sub(
        r"^\s*(?:<\|[^>]*\|>\s*)?(?:assistant\s*[:\n])?\s*"
        r"(?:the\s+)?(?:answer|label|classification|verdict)\s*(?:is)?\s*[:\-]?\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    # Take the first non-empty line: a label followed by unsolicited rationale
    # is still an unambiguous label.
    for line in out.splitlines():
        if line.strip():
            return line.strip()
    return out


def canonicalize(
    raw: str,
    labels: Sequence[str],
    synonyms: Mapping[str, str] | None = None,
) -> CanonicalizeResult:
    """Map `raw` onto one of `labels`.

    `synonyms` maps a surface form to a label, e.g. {"yes": "harmful",
    "1": "harmful"}. Keys are normalized before comparison.
    """
    if not labels:
        raise ValueError("labels must be non-empty")

    norm_labels = {normalize(l): l for l in labels}
    norm_syn = {normalize(k): v for k, v in (synonyms or {}).items()}
    unknown_targets = sorted(set(norm_syn.values()) - set(labels))
    if unknown_targets:
        raise ValueError(f"synonyms map to labels outside the label space: {unknown_targets}")

    stripped = _strip_scaffolding(raw)
    norm = normalize(stripped)

    if norm in norm_labels:
        return CanonicalizeResult(norm_labels[norm], True, "exact")
    if norm in norm_syn:
        return CanonicalizeResult(norm_syn[norm], True, "synonym-exact")

    # Whole-word containment. Longest label first so "not harmful" wins over
    # "harmful" when both appear as substrings.
    for norm_label in sorted(norm_labels, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(norm_label)}(?!\w)", norm):
            return CanonicalizeResult(norm_labels[norm_label], True, "contains")
    for norm_key in sorted(norm_syn, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(norm_key)}(?!\w)", norm):
            return CanonicalizeResult(norm_syn[norm_key], True, "synonym-contains")

    return CanonicalizeResult(None, False, "unparsed")


def parse_failure_report(
    rows: Sequence[tuple[str, bool]],
) -> dict[str, float]:
    """Per-family parse-failure rate from `(model_family, parse_ok)` pairs.

    A family whose outputs the parser cannot read is a measurement problem,
    not a weak model, and the two must not be confused when matching
    standalone accuracy across the panel.
    """
    totals: dict[str, int] = {}
    failures: dict[str, int] = {}
    for family, ok in rows:
        totals[family] = totals.get(family, 0) + 1
        if not ok:
            failures[family] = failures.get(family, 0) + 1
    return {f: failures.get(f, 0) / n for f, n in sorted(totals.items())}
