"""Training-set overlap O_jk = |D_j n D_k| / |D_j u D_k| (protocol section 13).

The protocol's point: this "turns 'information diversity' into a measured
variable rather than a verbal label". A condition labelled `disjoint` whose
overlap matrix is not zero off-diagonal is mislabelled, and the analysis would
attribute data-sharing dependence to representation sharing. So the matrix is
computed from the actual item ids each member trained on and written into the
run manifest, where a later reader can check it.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard overlap of two training slices. Two empty slices count as 0."""
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def overlap_matrix(splits: Mapping[str, Sequence[str]]) -> dict[str, float]:
    """Pairwise O_jk keyed `"j|k"` for j < k, plus diagonal entries at 1.0.

    Flat string keys rather than nested dicts so the matrix round-trips
    through JSON without a custom decoder.
    """
    keys = sorted(splits)
    out: dict[str, float] = {}
    for j in keys:
        out[f"{j}|{j}"] = 1.0 if splits[j] else 0.0
    for j, k in combinations(keys, 2):
        out[f"{j}|{k}"] = jaccard(splits[j], splits[k])
    return out


def mean_overlap(matrix: Mapping[str, float]) -> float:
    """Mean off-diagonal overlap. NaN when there is no pair to average."""
    offdiag = [
        v for key, v in matrix.items()
        if "|" in key and key.split("|")[0] != key.split("|")[1]
    ]
    if not offdiag:
        return float("nan")
    return sum(offdiag) / len(offdiag)


def overlap_report(
    matrix: Mapping[str, float], declared: str, tolerance: float = 1e-9
) -> dict[str, object]:
    """Check a measured overlap matrix against the condition's declared policy.

    `declared` is "shared" or "disjoint". Mismatches are returned as warnings
    rather than raised: a partially overlapping design is legitimate (it is how
    you trace out the O_jk axis), it just must not be *described* as disjoint.
    """
    mean = mean_overlap(matrix)
    offdiag = {
        k: v for k, v in matrix.items()
        if "|" in k and k.split("|")[0] != k.split("|")[1]
    }
    warnings: list[str] = []
    if declared == "disjoint":
        leaks = {k: v for k, v in offdiag.items() if v > tolerance}
        if leaks:
            worst = max(leaks.items(), key=lambda kv: kv[1])
            warnings.append(
                f"condition declares disjoint training slices but {len(leaks)} "
                f"pair(s) overlap (worst {worst[0]}={worst[1]:.4f}); "
                "data-sharing dependence would be misread as representation sharing"
            )
    elif declared == "shared":
        gaps = {k: v for k, v in offdiag.items() if v < 1.0 - 1e-6}
        if gaps:
            worst = min(gaps.items(), key=lambda kv: kv[1])
            warnings.append(
                f"condition declares shared training data but {len(gaps)} "
                f"pair(s) are not identical (lowest {worst[0]}={worst[1]:.4f})"
            )
    return {
        "declared": declared,
        "mean_offdiagonal_overlap": mean,
        "n_pairs": len(offdiag),
        "warnings": warnings,
    }
