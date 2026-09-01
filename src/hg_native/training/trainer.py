"""The training loop (spec sections 8, 9). Needs `torch`.

Minimal by design for the smoke/pilot stages (spec section 8.1/8.2): plain
AdamW, cosine-with-warmup LR, gradient-norm clipping, the HG loss warm-up
multiplier, no activation checkpointing, no distributed training, and (see
`dataset.py`) no `ordinary_lm` corpus mixed in yet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn

from hg_native.model.config import HGNativeConfig
from hg_native.model.hg_model import HGNativeModel

from .batch import ByteTokenizer, Tokenizer, collate, serialize_episode
from .dataset import SyntheticMix, sample_episodes, sample_transition_pairs
from .losses import compute_losses, next_token_labels, targets_to_tensors, total_loss, transition_loss


@dataclass(slots=True)
class OptimConfig:
    """Spec section 9's optimizer/schedule defaults."""

    lr: float = 3e-4
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    min_lr_ratio: float = 0.1
    clip_norm: float = 1.0


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay
    )


def lr_at_step(cfg: OptimConfig, step: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to `min_lr_ratio * lr` (spec section 9)."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    if step < warmup_steps:
        return cfg.lr * (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    floor = cfg.min_lr_ratio * cfg.lr
    return floor + (cfg.lr - floor) * cosine


@dataclass(slots=True)
class StepMetrics:
    step: int
    lr: float
    hg_multiplier: float
    grad_norm: float
    losses: dict[str, float]
    total: float


@dataclass(slots=True)
class TrainConfig:
    total_steps: int
    batch_size: int
    max_seq_len: int
    seed: int
    mix: SyntheticMix = field(default_factory=SyntheticMix)
    device: str = "cpu"


def _transition_loss_for_step(
    model: HGNativeModel, tokenizer: Tokenizer, train_cfg: TrainConfig, step: int, device: str
) -> torch.Tensor:
    """The transition head's loss for this step: a dedicated batch of
    consecutive Family-C `(episode_t, episode_t+1)` pairs (spec section 3.3),
    separate from the main mixed batch (which has no guaranteed consecutive
    pairs). `episode_t`'s own `correct_action` stands in for "the action
    actually taken" -- teacher-forced, like every other head's target, not
    sampled from the model's own (not-yet-trained) router.

    `S_{t+1}`'s target comes from a `torch.no_grad()` forward pass on
    `episode_t+1`, so the transition head chases the trunk's own current
    representation of the next step rather than a fixed external label --
    exactly spec section 3.3's `sg(S_{t+1})`.
    """
    pairs = sample_transition_pairs(
        seed=train_cfg.seed, n=train_cfg.batch_size, start_index=step * train_cfg.batch_size
    )
    cur_batch = collate([serialize_episode(ep, tokenizer, train_cfg.max_seq_len) for ep, _ in pairs])
    next_batch = collate([serialize_episode(ep, tokenizer, train_cfg.max_seq_len) for _, ep in pairs])

    cur_ids = torch.tensor(cur_batch["input_ids"], device=device)
    cur_positions = torch.tensor(cur_batch["decision_position"], device=device)
    cur_action_id = torch.tensor(cur_batch["action_id"], device=device)
    out_cur = model(cur_ids, decision_positions=cur_positions, action_id=cur_action_id)

    next_ids = torch.tensor(next_batch["input_ids"], device=device)
    next_positions = torch.tensor(next_batch["decision_position"], device=device)
    with torch.no_grad():
        s_next_target = model(next_ids, decision_positions=next_positions)["s_t"]

    return transition_loss(out_cur["s_next_hat"], s_next_target)


def train_step(
    model: HGNativeModel,
    config: HGNativeConfig,
    optimizer: torch.optim.Optimizer,
    optim_cfg: OptimConfig,
    tokenizer: Tokenizer,
    train_cfg: TrainConfig,
    step: int,
) -> StepMetrics:
    """One optimizer step: sample a fresh batch, forward, backward, clip, step."""
    episodes = sample_episodes(
        seed=train_cfg.seed, n=train_cfg.batch_size, mix=train_cfg.mix, start_index=step * train_cfg.batch_size
    )
    raw_batch = collate([serialize_episode(ep, tokenizer, train_cfg.max_seq_len) for ep in episodes])

    device = train_cfg.device
    ids = torch.tensor(raw_batch["input_ids"], device=device)
    attention_mask = torch.tensor(raw_batch["attention_mask"], device=device)
    labels = next_token_labels(ids, attention_mask)

    if config.variant == "hg":
        positions = torch.tensor(raw_batch["decision_position"], device=device)
        action_id = torch.tensor(raw_batch["action_id"], device=device)
        out = model(ids, decision_positions=positions, action_id=action_id)
        target_tensors = targets_to_tensors(raw_batch["targets"], device=torch.device(device))
    else:
        out = model(ids)
        target_tensors = {}

    losses = compute_losses(out, target_tensors, labels)

    if config.variant == "hg":
        losses["transition"] = _transition_loss_for_step(model, tokenizer, train_cfg, step, device)
    hg_mult = config.loss_schedule.hg_multiplier(step / train_cfg.total_steps) if config.variant == "hg" else 0.0
    loss = total_loss(losses, config.loss_weights, hg_multiplier=hg_mult)

    lr = lr_at_step(optim_cfg, step, train_cfg.total_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optim_cfg.clip_norm)
    optimizer.step()

    return StepMetrics(
        step=step,
        lr=lr,
        hg_multiplier=hg_mult,
        grad_norm=float(grad_norm),
        losses={k: float(v.detach()) for k, v in losses.items()},
        total=float(loss.detach()),
    )


def train(
    model: HGNativeModel,
    config: HGNativeConfig,
    train_cfg: TrainConfig,
    optim_cfg: OptimConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> list[StepMetrics]:
    """Run `train_cfg.total_steps` optimizer steps; returns per-step metrics."""
    optim_cfg = optim_cfg or OptimConfig()
    tokenizer = tokenizer or ByteTokenizer()
    optimizer = build_optimizer(model, optim_cfg)
    model.to(train_cfg.device)
    model.train()
    return [
        train_step(model, config, optimizer, optim_cfg, tokenizer, train_cfg, step)
        for step in range(train_cfg.total_steps)
    ]
