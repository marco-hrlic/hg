"""Synthetic panel generation for Phase 0.

Phase 0's job is to validate the pipeline, not to produce evidence, and it
needs a dataset with two properties no real dataset has: known item difficulty,
and a known information-limited core. Both are written into item metadata,
where `hg.inference.simulator` reads them.

The generated panel also carries realistic anchor structure -- a mix of
unanimous, majority and contested items -- because a pipeline that has only
ever seen unanimous anchors will not have exercised the clean-subset logic that
every headline metric depends on.

Nothing here is used once real tasks are wired in. `configs/datasets/` is where
a real task's panel file is pointed at.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..anchors.panel import PanelItem
from ..inference.simulator import META_CORE, META_DIFFICULTY, META_TRUE_LABEL


@dataclass(slots=True)
class SyntheticSpec:
    n_items: int = 2000
    labels: tuple[str, ...] = ("not_harmful", "harmful")
    n_annotators: int = 5
    # Share of items where annotators genuinely disagree. Real judgment tasks
    # sit well above zero here; a task with no contested items cannot exercise
    # the distinction the experiment is built on.
    contested_rate: float = 0.15
    majority_rate: float = 0.20
    # SD of latent item difficulty u_i.
    difficulty_sd: float = 1.2
    # Share of items forming an information-limited core: every family gets
    # these wrong regardless of lineage or training slice. Set 0.0 to simulate
    # decision-gate outcome A, >0 for outcome C.
    core_rate: float = 0.02
    # Whether the core clusters in representation space, which is what makes it
    # targetable by failure memory (prediction P6).
    core_clustered: bool = True
    seed: int = 0


def generate_synthetic_panel(spec: SyntheticSpec) -> list[PanelItem]:
    """Generate panel items with latent difficulty, core flags, and anchors."""
    rng = np.random.default_rng(spec.seed)
    labels = list(spec.labels)
    n = spec.n_items
    r = spec.n_annotators

    difficulty = rng.normal(0.0, spec.difficulty_sd, n)
    is_core = rng.random(n) < spec.core_rate
    true_idx = rng.integers(0, len(labels), n)

    # Anchor structure is assigned independently of the core, so a contested
    # item can also be a core item. Coupling them would build the conclusion
    # ("the residual is really anchor disagreement") into the fixture.
    u = rng.random(n)
    contested = u < spec.contested_rate
    majority = (u >= spec.contested_rate) & (u < spec.contested_rate + spec.majority_rate)

    items: list[PanelItem] = []
    for i in range(n):
        true_label = labels[true_idx[i]]
        other = [l for l in labels if l != true_label]

        if contested[i]:
            # No clear majority: split the panel as evenly as the size allows.
            half = r // 2
            judgments = [true_label] * half + [other[0]] * (r - half)
            rng.shuffle(judgments)
        elif majority[i]:
            n_dissent = max(1, r // 3)
            judgments = [true_label] * (r - n_dissent) + [
                other[rng.integers(0, len(other))] for _ in range(n_dissent)
            ]
            rng.shuffle(judgments)
        else:
            judgments = [true_label] * r

        item_id = f"syn-{i:06d}"
        items.append(
            PanelItem(
                item_id=item_id,
                x=_synthetic_text(item_id, difficulty[i], bool(is_core[i])),
                judgments=judgments,
                annotator_ids=[f"ann-{j}" for j in range(r)],
                meta={
                    META_DIFFICULTY: float(difficulty[i]),
                    META_CORE: bool(is_core[i]),
                    META_TRUE_LABEL: true_label,
                    "synthetic": True,
                    "core_cluster": (
                        int(hashlib.blake2b(item_id.encode(), digest_size=2).hexdigest(), 16) % 3
                        if (is_core[i] and spec.core_clustered)
                        else None
                    ),
                },
            )
        )
    return items


def _synthetic_text(item_id: str, difficulty: float, core: bool) -> str:
    """Placeholder input text.

    Deliberately obvious as synthetic. A fixture that reads like real task data
    invites someone to quote a Phase 0 number as a result.
    """
    tag = "core" if core else "ordinary"
    return (
        f"[SYNTHETIC FIXTURE {item_id}] {tag} item, latent difficulty "
        f"{difficulty:+.2f}. This text is a placeholder for pipeline validation "
        "and carries no real content."
    )
