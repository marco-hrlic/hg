"""Text cleaning and duplicate detection."""

from __future__ import annotations

import html
import re
import zlib
from collections import defaultdict

import numpy as np

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def clean_text(text: object) -> str:
    """Make review text readable: unescape HTML, drop tags, fix newlines, collapse whitespace.

    This changes how the text *looks*, not what it *says*. It is safe to apply
    before both training and prediction.
    """
    if not isinstance(text, str):
        return ""
    t = html.unescape(text)
    t = _TAG_RE.sub(" ", t)
    t = t.replace("\\n", " ")  # the source escapes newlines as a literal backslash-n
    return _WS_RE.sub(" ", t).strip()


def normalize_for_dedup(text: object) -> str:
    """Aggressive normalisation used only to compare texts: lowercase, letters and digits only."""
    t = clean_text(text).lower()
    t = _NON_ALNUM_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def shingles(text: str, k: int = 3) -> set[str]:
    """Overlapping word k-grams of an already-normalised text."""
    words = text.split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


_MERSENNE = (1 << 61) - 1


def find_near_duplicates(
    texts: list[str],
    threshold: float = 0.7,
    num_perm: int = 64,
    bands: int = 16,
    seed: int = 0,
) -> list[tuple[int, int, float]]:
    """Find pairs of texts whose word-3-gram Jaccard similarity is >= threshold.

    Uses MinHash + locality-sensitive hashing so we never compare all n^2 pairs:
    each text gets a short "fingerprint", texts whose fingerprints agree on a
    whole band become candidates, and only candidates are checked exactly.

    Returns (i, j, similarity) with i < j, indexes into `texts`.
    """
    rows = num_perm // bands
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE, size=num_perm, dtype=np.uint64)

    norm = [normalize_for_dedup(t) for t in texts]
    sh = [shingles(t) for t in norm]

    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, s in enumerate(sh):
        if not s:
            continue
        h = np.fromiter((zlib.crc32(x.encode()) for x in s), dtype=np.uint64, count=len(s))
        # (a*h + b) mod p, computed in uint64 with wraparound; good enough for teaching scale.
        sig = ((a[:, None] * h[None, :] + b[:, None]) % np.uint64(_MERSENNE)).min(axis=1)
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows].tobytes())
            buckets[key].append(idx)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) > 1:
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    candidates.add((members[x], members[y]))

    out = []
    for i, j in sorted(candidates):
        sim = jaccard(sh[i], sh[j])
        if sim >= threshold:
            out.append((i, j, sim))
    return out


def cluster_pairs(n: int, pairs: list[tuple[int, int, float]]) -> np.ndarray:
    """Union-find: turn duplicate pairs into a cluster id per item (0..n-1)."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, *_ in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.array([find(i) for i in range(n)])
