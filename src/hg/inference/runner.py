"""Running an ensemble condition and emitting item-level rows.

This is the only place that turns model output into `ItemResult` rows, so
every backend, every condition and every phase produce a matrix with the same
columns and the same scoring convention.

Two scoring decisions are made explicitly here because both can quietly bias
consensus metrics:

* **Unparsed output.** Default `unparsed_policy="wrong"`: a model that did not
  answer got the item wrong. The alternative, dropping the cell, makes the
  matrix ragged, and an item scored by four of five members cannot be "all
  wrong" on the same footing as one scored by five. The parse-failure rate is
  reported per family regardless, since a parser problem masquerading as a
  weak model would corrupt the capability matching in section 18.
* **Anchor status travels with the row.** `anchor_label` and `anchor_status`
  are stamped from a single anchor map, so a row's status can never disagree
  with the subset the analysis conditions on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..anchors.panel import AnchoredItem
from ..config import ExperimentConfig, ModelConfig
from ..schema import ItemResult, RunManifest, SchemaError
from .base import (
    Backend,
    GenerationSettings,
    InferenceItem,
    get_backend,
    validate_predictions,
)
from .canonicalize import parse_failure_report
from .prompts import get_template
from .simulator import META_CORE, META_DIFFICULTY, META_TRUE_LABEL

log = logging.getLogger(__name__)

UNPARSED_POLICIES = ("wrong", "exclude")


@dataclass(slots=True)
class MemberSpec:
    """One ensemble member: a model, an adapter, and a training slice."""

    member_id: str
    model: ModelConfig
    condition: str
    adapter_seed: int
    train_split: str
    adapter_path: str | None = None
    adapter_hash: str | None = None
    # Item ids this member trained on, used to compute O_jk (section 13).
    train_item_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "model": self.model.name,
            "family": self.model.family,
            "lineage": self.model.lineage,
            "checkpoint": self.model.checkpoint,
            "backend": self.model.backend,
            "condition": self.condition,
            "adapter_seed": self.adapter_seed,
            "train_split": self.train_split,
            "adapter_path": self.adapter_path,
            "adapter_hash": self.adapter_hash,
            "n_train_items": len(self.train_item_ids),
        }


def build_inference_items(
    items: Sequence[AnchoredItem],
    labels: Sequence[str],
    prompt_version: str,
    prompt_extra: Mapping[str, str] | None = None,
) -> list[InferenceItem]:
    """Render prompts once, for every member, from one template."""
    template = get_template(prompt_version)
    out: list[InferenceItem] = []
    for it in items:
        meta = dict(it.meta)
        # Latent simulator fields are derived from the item, not the member, so
        # item difficulty is genuinely shared across the ensemble.
        meta.setdefault(META_TRUE_LABEL, it.reference)
        out.append(
            InferenceItem(
                item_id=it.item_id,
                prompt=template.render(it.x, labels, extra=prompt_extra),
                labels=list(labels),
                meta=meta,
            )
        )
    return out


def make_backend(spec: MemberSpec, world_seed: int, synonyms: Mapping[str, str] | None) -> Backend:
    """Instantiate the backend for a member.

    The simulator is handed the member's *lineage* and *train_split* rather
    than its name, because those are what decide which error components are
    shared (see `hg.inference.simulator`).
    """
    mc = spec.model
    options = dict(mc.options)
    if mc.backend == "simulator":
        return get_backend(
            "simulator",
            lineage=mc.lineage or mc.family,
            train_split=spec.train_split,
            adapter_seed=spec.adapter_seed,
            world_seed=world_seed,
            params=options,
            checkpoint=mc.checkpoint,
        )
    return get_backend(
        mc.backend,
        checkpoint=mc.checkpoint,
        dtype=mc.dtype,
        adapter_path=spec.adapter_path,
        synonyms=dict(synonyms or {}),
        **options,
    )


def run_member(
    spec: MemberSpec,
    items: Sequence[AnchoredItem],
    inference_items: Sequence[InferenceItem],
    *,
    task: str,
    prompt_version: str,
    settings: GenerationSettings,
    world_seed: int = 0,
    synonyms: Mapping[str, str] | None = None,
    unparsed_policy: str = "wrong",
    keep_backend_loaded: bool = False,
    backend: Backend | None = None,
) -> list[ItemResult]:
    """Score one member over the evaluation set, returning item-level rows."""
    if unparsed_policy not in UNPARSED_POLICIES:
        raise ValueError(f"unparsed_policy must be one of {UNPARSED_POLICIES}")

    own_backend = backend is None
    backend = backend or make_backend(spec, world_seed, synonyms)
    try:
        backend.load()
        preds = backend.predict(list(inference_items), settings)
        validate_predictions(inference_items, preds)
    finally:
        if own_backend and not keep_backend_loaded:
            backend.unload()

    by_id = {it.item_id: it for it in items}
    rows: list[ItemResult] = []
    for pred in preds:
        anchored = by_id[pred.item_id]
        if not pred.parse_ok and unparsed_policy == "exclude":
            continue
        correct = (pred.label == anchored.reference) if pred.label is not None else False
        rows.append(
            ItemResult(
                item_id=pred.item_id,
                task=task,
                model_family=spec.model.family,
                model_version=spec.model.checkpoint,
                member_id=spec.member_id,
                condition=spec.condition,
                adapter_seed=spec.adapter_seed,
                adapter_hash=spec.adapter_hash,
                train_split=spec.train_split,
                prompt_version=prompt_version,
                prediction=pred.label,
                raw_output=pred.raw_output,
                parse_ok=pred.parse_ok,
                score=(1.0 if correct else 0.0),
                confidence=pred.confidence,
                probs=pred.probs,
                logprob=pred.logprob,
                embedding=pred.embedding,
                inference_seed=settings.seed,
                decoding=settings.to_dict(),
                anchor_label=anchored.reference,
                anchor_status=anchored.status,
                correct=correct,
            )
        )
    return rows


def run_condition(
    cfg: ExperimentConfig,
    condition: str,
    items: Sequence[AnchoredItem],
    specs: Sequence[MemberSpec],
    *,
    settings: GenerationSettings | None = None,
    unparsed_policy: str = "wrong",
) -> tuple[list[ItemResult], RunManifest]:
    """Run every member of a condition and assemble rows plus provenance."""
    ds = cfg.dataset
    synonyms = dict(ds.options.get("synonyms") or {})
    prompt_extra = {
        k: str(v) for k, v in (ds.options.get("prompt_extra") or {}).items()
    }
    inference_items = build_inference_items(
        items, ds.labels, ds.prompt_version, prompt_extra=prompt_extra
    )
    if settings is None:
        max_new = max((m.model.max_new_tokens for m in specs), default=8)
        settings = GenerationSettings(max_new_tokens=max_new, seed=cfg.seed)

    rows: list[ItemResult] = []
    for spec in specs:
        log.info(
            "condition=%s member=%s family=%s split=%s seed=%s",
            condition, spec.member_id, spec.model.family, spec.train_split, spec.adapter_seed,
        )
        rows.extend(
            run_member(
                spec,
                items,
                inference_items,
                task=ds.task,
                prompt_version=ds.prompt_version,
                settings=settings,
                world_seed=cfg.seed,
                synonyms=synonyms,
                unparsed_policy=unparsed_policy,
            )
        )

    failures = parse_failure_report([(r.model_family, r.parse_ok) for r in rows])
    bad = {f: rate for f, rate in failures.items() if rate > 0.02}
    if bad:
        log.warning(
            "parse-failure rate above 2%% for %s -- check prompt/parser before "
            "attributing the gap to model capability", bad,
        )

    from ..diversity.overlap import overlap_matrix

    manifest = RunManifest(
        run_id=f"{cfg.name}:{condition}",
        condition=condition,
        task=ds.task,
        dataset_config=ds.name,
        experiment_config=cfg.name,
        members=[s.to_dict() for s in specs],
        backend=",".join(sorted({s.model.backend for s in specs})),
        train_overlap=overlap_matrix(
            {s.member_id: s.train_item_ids for s in specs}
        ),
        notes=(
            f"parse_failure_rate={failures}; unparsed_policy={unparsed_policy}"
        ),
    )
    return rows, manifest
