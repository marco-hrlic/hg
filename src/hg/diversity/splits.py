"""Training-slice construction for the four conditions (protocol section 4).

    HS  same family, shared slice,   seeds vary
    HD  same family, disjoint slices
    DS  five families, shared slice
    DD  five families, disjoint slices

`build_splits` is the only place slices are made, so "shared" and "disjoint"
mean exactly one thing across the experiment, and the item ids it assigns are
recorded per member for the O_jk computation.

Held-out evaluation items are carved out before any training slice is cut, and
that boundary is asserted rather than assumed: leakage of an evaluation item
into a training slice would make the residual look smaller than it is, in the
one direction that would produce a false "diversity closes the gap" result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(slots=True)
class SplitPlan:
    """Item ids assigned to evaluation and to each member's training slice."""

    eval_ids: list[str]
    train_by_member: dict[str, list[str]]
    policy: str  # "shared" or "disjoint"
    pool_size: int = 0
    unused_train_ids: list[str] = field(default_factory=list)

    def validate(self) -> None:
        eval_set = set(self.eval_ids)
        for member, ids in self.train_by_member.items():
            leak = eval_set & set(ids)
            if leak:
                raise ValueError(
                    f"member {member}: {len(leak)} evaluation items leaked into "
                    f"its training slice (first: {sorted(leak)[:3]})"
                )
            if len(set(ids)) != len(ids):
                raise ValueError(f"member {member}: duplicate items in training slice")
        if self.policy == "disjoint":
            members = sorted(self.train_by_member)
            for i, a in enumerate(members):
                for b in members[i + 1 :]:
                    shared = set(self.train_by_member[a]) & set(self.train_by_member[b])
                    if shared:
                        raise ValueError(
                            f"disjoint policy violated: members {a} and {b} share "
                            f"{len(shared)} training items"
                        )


def _stable_shuffle(ids: Sequence[str], seed: int) -> list[str]:
    """Deterministic permutation that does not depend on Python's hash seed."""
    def key(item_id: str) -> str:
        return hashlib.blake2b(
            f"{seed}:{item_id}".encode("utf-8"), digest_size=16
        ).hexdigest()

    return sorted(ids, key=key)


def build_splits(
    item_ids: Sequence[str],
    member_ids: Sequence[str],
    *,
    policy: str,
    n_eval: int | None = None,
    n_train_per_member: int | None = None,
    seed: int = 0,
) -> SplitPlan:
    """Carve evaluation items out first, then cut training slices.

    Under `policy="shared"` every member gets the *same* slice, sized
    `n_train_per_member`. Under `policy="disjoint"` the training pool is cut
    into non-overlapping slices of that size, which requires
    `n_train_per_member * len(member_ids)` items to be available.

    A design note worth stating: to compare HS with HD honestly, the per-member
    training budget must be equal, not the total. Giving the disjoint condition
    one fifth of the data per model would confound information diversity with
    training-set size, and the resulting drop in shared error would be partly a
    weaker-models artifact rather than a diversity effect.
    """
    if policy not in ("shared", "disjoint"):
        raise ValueError("policy must be 'shared' or 'disjoint'")
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("member ids must be unique")

    shuffled = _stable_shuffle(item_ids, seed)
    n_total = len(shuffled)
    n_eval_actual = n_total if n_eval is None else min(n_eval, n_total)
    eval_ids = shuffled[:n_eval_actual]
    pool = shuffled[n_eval_actual:]

    n_members = len(member_ids)
    if n_train_per_member is None:
        per = len(pool) if policy == "shared" else len(pool) // max(n_members, 1)
    else:
        per = int(n_train_per_member)

    train_by_member: dict[str, list[str]] = {}
    if policy == "shared":
        if per > len(pool):
            raise ValueError(
                f"training pool has {len(pool)} items, need {per} per member "
                f"(reduce n_eval or n_train_per_member)"
            )
        slice_ids = pool[:per]
        for m in member_ids:
            train_by_member[m] = list(slice_ids)
        unused = pool[per:]
    else:
        needed = per * n_members
        if needed > len(pool):
            raise ValueError(
                f"disjoint policy needs {needed} training items "
                f"({per} x {n_members} members) but the pool has {len(pool)}. "
                "Equal per-member budget is required so information diversity "
                "is not confounded with training-set size."
            )
        for i, m in enumerate(member_ids):
            train_by_member[m] = pool[i * per : (i + 1) * per]
        unused = pool[needed:]

    plan = SplitPlan(
        eval_ids=eval_ids,
        train_by_member=train_by_member,
        policy=policy,
        pool_size=len(pool),
        unused_train_ids=list(unused),
    )
    plan.validate()
    return plan


def split_summary(plan: SplitPlan) -> dict[str, object]:
    from .overlap import mean_overlap, overlap_matrix

    matrix = overlap_matrix(plan.train_by_member)
    sizes = {m: len(ids) for m, ids in sorted(plan.train_by_member.items())}
    return {
        "policy": plan.policy,
        "n_eval": len(plan.eval_ids),
        "train_pool": plan.pool_size,
        "train_sizes": sizes,
        "equal_budget": len(set(sizes.values())) <= 1,
        "mean_offdiagonal_overlap": mean_overlap(matrix),
    }
