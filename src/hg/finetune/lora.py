"""LoRA adaptation (protocol section 13).

The protocol's framing matters for how this is written: "The objective is not
to maximize benchmark performance but to compare dependence structure cheaply
and reproducibly across families." So the defaults are deliberately modest and
uniform -- rank 16, limited tuning, one recipe for every family -- and the
module spends its effort on *recording* what happened rather than on squeezing
out accuracy.

Four protocol requirements are enforced here rather than left to discipline:

* rank stays modest and hyperparameters are not searched per family (a per-
  family search would make families differ by tuning effort, not lineage);
* training-set overlaps and seeds are recorded explicitly, in the adapter card;
* adapters are never shared across families -- `train_lora` refuses to write
  into a directory whose card names a different base checkpoint;
* the adapter hash covers the data, the seed and the hyperparameters, so a row
  in the prediction matrix can be traced to exactly one training run.

torch/peft are imported lazily so the analysis pipeline stays installable
without them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

log = logging.getLogger(__name__)

# Attention/MLP projections present in every family in the section 3 panel.
# One target set across families, so adapter capacity does not silently differ
# by lineage.
DEFAULT_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


@dataclass(slots=True)
class LoRASettings:
    """One adaptation recipe, applied unchanged to every family."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 1e-4
    n_epochs: float = 1.0
    batch_size: int = 4
    grad_accum: int = 4
    max_seq_len: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    seed: int = 0
    # Kept explicit so a deviation from the protocol's "keep hyperparameter
    # tuning limited" is a recorded decision rather than an accident.
    tuning_notes: str = "protocol default; no per-family search"

    def __post_init__(self) -> None:
        if self.rank not in (8, 16, 32, 64):
            log.warning(
                "LoRA rank %d is outside the protocol's suggested modest range "
                "(16 or 32); record why in tuning_notes", self.rank,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "max_seq_len": self.max_seq_len,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "target_modules": list(self.target_modules),
            "seed": self.seed,
            "tuning_notes": self.tuning_notes,
        }

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.max_seq_len


def adapter_hash(
    *,
    checkpoint: str,
    train_item_ids: Sequence[str],
    settings: LoRASettings,
    adapter_seed: int,
    prompt_version: str,
) -> str:
    """A content hash identifying one training run.

    Covers the base checkpoint, the exact training items, the recipe, the seed
    and the prompt version -- everything that can change what the adapter
    learned. Item ids are sorted, so slice *order* does not change the hash
    while slice *membership* does.

    This is what lets a prediction row be traced to one training run months
    later, and what makes an accidental adapter reuse across families
    detectable instead of invisible.
    """
    payload = json.dumps(
        {
            "checkpoint": checkpoint,
            "train_items": sorted(train_item_ids),
            "settings": settings.to_dict(),
            "adapter_seed": adapter_seed,
            "prompt_version": prompt_version,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def write_adapter_card(
    path: str | Path,
    *,
    member_id: str,
    checkpoint: str,
    family: str,
    lineage: str,
    train_item_ids: Sequence[str],
    settings: LoRASettings,
    adapter_seed: int,
    prompt_version: str,
    budget: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> str:
    """Write `adapter_card.json` beside an adapter and return its hash."""
    h = adapter_hash(
        checkpoint=checkpoint,
        train_item_ids=train_item_ids,
        settings=settings,
        adapter_seed=adapter_seed,
        prompt_version=prompt_version,
    )
    card = {
        "adapter_hash": h,
        "member_id": member_id,
        "base_checkpoint": checkpoint,
        "family": family,
        "lineage": lineage,
        "adapter_seed": adapter_seed,
        "prompt_version": prompt_version,
        "n_train_items": len(train_item_ids),
        # The full id list, not a count: O_jk cannot be recomputed from a count,
        # and recomputing it later is the only way to audit a "disjoint" claim.
        "train_item_ids": sorted(train_item_ids),
        "settings": settings.to_dict(),
        "budget": dict(budget or {}),
        "train_metrics": dict(metrics or {}),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(card, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return h


def read_adapter_card(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_no_adapter_sharing(adapter_dir: str | Path, checkpoint: str) -> None:
    """Refuse to reuse an adapter trained on a different base model.

    Protocol section 13: "Do not share adapters across model families." A
    silent reuse would make two members far more similar than the design says
    they are, biasing every dependence measure toward the homogeneous result.
    """
    card_path = Path(adapter_dir) / "adapter_card.json"
    if not card_path.exists():
        return
    card = read_adapter_card(card_path)
    existing = card.get("base_checkpoint")
    if existing and existing != checkpoint:
        raise RuntimeError(
            f"{adapter_dir} holds an adapter for base checkpoint {existing!r}, "
            f"not {checkpoint!r}. Adapters must not be shared across model "
            "families (protocol section 13). Use a per-member output directory."
        )


def build_training_texts(
    items: Sequence[Any],
    labels: Sequence[str],
    prompt_version: str,
    prompt_extra: Mapping[str, str] | None = None,
) -> list[str]:
    """Render prompt+target strings for supervised adaptation.

    The training prompt uses the *same* template as evaluation. A train/eval
    template mismatch is the cheapest way to manufacture a family-specific
    performance gap, and it would show up as representational diversity.
    """
    from ..inference.prompts import get_template

    template = get_template(prompt_version)
    texts: list[str] = []
    for it in items:
        prompt = template.render(it.x, labels, extra=prompt_extra)
        texts.append(f"{prompt} {it.reference}")
    return texts


def train_lora(
    *,
    checkpoint: str,
    output_dir: str | Path,
    train_items: Sequence[Any],
    labels: Sequence[str],
    prompt_version: str,
    settings: LoRASettings | None = None,
    member_id: str = "member",
    family: str = "unknown",
    lineage: str = "unknown",
    adapter_seed: int = 0,
    dtype: str = "bfloat16",
    prompt_extra: Mapping[str, str] | None = None,
    trust_remote_code: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train one LoRA adapter and write its card.

    `dry_run=True` plans the budget and writes the card without touching a GPU,
    which is how Phase 0 validates the schema and the overlap bookkeeping
    before any real training time is spent.

    Returns a dict with `adapter_path`, `adapter_hash` and `budget`, which is
    exactly what `MemberSpec` needs.
    """
    settings = settings or LoRASettings(seed=adapter_seed)
    output_dir = Path(output_dir)
    check_no_adapter_sharing(output_dir, checkpoint)

    train_item_ids = [it.item_id for it in train_items]
    texts = build_training_texts(train_items, labels, prompt_version, prompt_extra)

    from .budget import plan_budget

    if dry_run:
        # Whitespace tokens are a stand-in only for planning; the real count
        # comes from the family's own tokenizer below.
        count = lambda t: max(1, len(t.split()))  # noqa: E731
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
        count = lambda t: len(tok(t).input_ids)  # noqa: E731

    budget = plan_budget(
        member_id,
        texts,
        count,
        mode="equal_examples",
        n_epochs=settings.n_epochs,
        max_seq_len=settings.max_seq_len,
    )

    metrics: dict[str, Any] = {}
    if not dry_run:
        metrics = _run_training(
            checkpoint=checkpoint,
            output_dir=output_dir,
            texts=texts[: budget.n_examples],
            settings=settings,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    h = write_adapter_card(
        output_dir / "adapter_card.json",
        member_id=member_id,
        checkpoint=checkpoint,
        family=family,
        lineage=lineage,
        train_item_ids=train_item_ids[: budget.n_examples],
        settings=settings,
        adapter_seed=adapter_seed,
        prompt_version=prompt_version,
        budget=budget.to_dict(),
        metrics=metrics,
    )
    return {
        "adapter_path": str(output_dir) if not dry_run else None,
        "adapter_hash": h,
        "budget": budget.to_dict(),
        "train_item_ids": train_item_ids[: budget.n_examples],
        "train_metrics": metrics,
    }


def _run_training(
    *,
    checkpoint: str,
    output_dir: Path,
    texts: Sequence[str],
    settings: LoRASettings,
    dtype: str,
    trust_remote_code: bool,
) -> dict[str, Any]:
    """The actual peft/transformers training loop. GPU path."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(settings.seed)
    tok = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tok(
            batch["text"],
            truncation=True,
            max_length=settings.max_seq_len,
            padding=False,
        )

    ds = Dataset.from_dict({"text": list(texts)}).map(
        tokenize, batched=True, remove_columns=["text"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    # Some families ship with cache enabled, which conflicts with gradient
    # checkpointing and silently slows training rather than failing.
    model.config.use_cache = False

    present = _present_target_modules(model, settings.target_modules)
    if not present:
        raise RuntimeError(
            f"none of the target modules {settings.target_modules} exist in "
            f"{checkpoint}; adapter capacity would be zero"
        )
    if set(present) != set(settings.target_modules):
        log.warning(
            "%s: adapting %s; %s absent from this architecture. Adapter capacity "
            "differs from other families -- record this before comparing.",
            checkpoint, sorted(present),
            sorted(set(settings.target_modules) - set(present)),
        )

    peft_cfg = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(present),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.n_epochs,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=settings.seed,
        data_seed=settings.seed,
        bf16=(dtype == "bfloat16"),
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    )
    result = trainer.train()
    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))

    return {
        "train_loss": float(result.training_loss),
        "n_steps": int(result.global_step),
        "trainable_params": int(trainable),
        "adapted_modules": sorted(present),
    }


def _present_target_modules(model: Any, wanted: Sequence[str]) -> list[str]:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    return [w for w in wanted if w in names]
