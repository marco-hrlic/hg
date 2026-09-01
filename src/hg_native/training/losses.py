"""The HG loss registry and warm-up schedule (spec sections 3.3 and 7).

Needs `torch` (the `gpu` extra) -- unlike `batch.py`/`targets.py`, which stay
torch-free. Every per-head loss takes already-defined-mask tensors (from
`batch.collate`'s `targets` dict, converted via `targets_to_tensors`) so an
episode whose family doesn't pin an answer for a head (spec section 2.3:
"do not collapse [info deficit and accessibility] into a single confidence
score", and more generally Family D has no decision-head target, etc.)
contributes exactly zero gradient to that head rather than training toward a
fabricated label.

The transition head is the one exception: its target is `sg(S_{t+1})`, a
stop-gradient of the *next* timestep's own forward pass (spec section 3.3),
which only a trainer pairing up Family C's consecutive `sequential_state_id`
steps can produce -- `transition_loss` takes that target as a plain argument
rather than trying to derive it from one episode's batch alone.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from hg_native.model.config import HGLossSchedule, LossWeights


def targets_to_tensors(batch_targets: dict[str, list], device: torch.device | None = None) -> dict[str, torch.Tensor]:
    """Convert `batch.collate(...)["targets"]` (plain lists) to tensors."""
    long_keys = {"decision_label", "route_label"}
    bool_keys = {"decision_defined", "distortion_mask", "accessibility_defined", "route_defined"}
    out: dict[str, torch.Tensor] = {}
    for key, values in batch_targets.items():
        if key in long_keys:
            out[key] = torch.tensor(values, dtype=torch.long, device=device)
        elif key in bool_keys:
            out[key] = torch.tensor(values, dtype=torch.bool, device=device)
        else:
            out[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return out


def _masked_mean(per_example: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over `per_example` where `mask` is true; 0 (not NaN) if nothing is defined.

    Multiplying by the mask (rather than boolean-indexing) keeps this
    differentiable and well-defined even when every example in the batch is
    masked out, which happens routinely (e.g. no Family D episodes in a
    batch means the decision head sees nothing to mask *out*, but a
    THINK-only batch masks every `route` target from an ACT-vs-ASK
    perspective -- either way, zero contribution, not a crash).
    """
    mask = mask.to(per_example.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (per_example * mask).sum() / denom


def next_token_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Build `labels` such that `labels[:, t]` is the token *after* position `t`.

    Without this shift, `lm_loss(logits, input_ids)` asks position `t`'s
    output to reproduce position `t`'s own input token -- with tied
    embeddings, the residual stream already carries `embed(x_t)` straight
    through to the final norm, so the LM head can "predict" it almost
    perfectly at random init by simple self-similarity, collapsing the loss
    to ~0 with ~0 gradient before any real training happens. Caught by a
    smoke run on the Spark returning exactly `0.0` loss and `0.0` grad norm
    from step 0 -- the spec section 8.1 smoke stage's whole job.

    The last position and any position whose *target* token is padding are
    set to `ignore_index` so padding never contributes to the loss.
    """
    labels = torch.full_like(input_ids, ignore_index)
    labels[:, :-1] = input_ids[:, 1:]
    pad_target = attention_mask[:, 1:] == 0
    labels[:, :-1] = labels[:, :-1].masked_fill(pad_target, ignore_index)
    return labels


def lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Cross-entropy against already-next-token-aligned `labels` (see `next_token_labels`).

    logits: (batch, seq, vocab); labels: (batch, seq).
    """
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=ignore_index)


def decision_loss(decision_logits: torch.Tensor, decision_label: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
    per_example = F.cross_entropy(decision_logits, decision_label, reduction="none")
    return _masked_mean(per_example, defined)


def quotient_loss(distortion_hat: torch.Tensor, distortion_vec: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The distortion-profile head's loss (spec section 7 names it L_quotient)."""
    per_element = F.mse_loss(distortion_hat, distortion_vec, reduction="none")
    return _masked_mean(per_element, mask)


def info_deficit_loss(info_deficit_logit: torch.Tensor, info_deficit_target: torch.Tensor) -> torch.Tensor:
    """Always defined (spec section 3.3): no masking needed."""
    return F.binary_cross_entropy_with_logits(info_deficit_logit, info_deficit_target)


def accessibility_loss(accessibility_logit: torch.Tensor, accessibility_target: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
    per_example = F.binary_cross_entropy_with_logits(accessibility_logit, accessibility_target, reduction="none")
    return _masked_mean(per_example, defined)


def route_loss(route_logits: torch.Tensor, route_label: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
    per_example = F.cross_entropy(route_logits, route_label, reduction="none")
    return _masked_mean(per_example, defined)


def transition_loss(s_next_hat: torch.Tensor, s_next_target: torch.Tensor) -> torch.Tensor:
    """MSE against `sg(S_{t+1})` (spec section 3.3). Caller must already have
    detached `s_next_target` -- this function does not call `.detach()`
    itself, so a caller that forgets to stop the gradient gets exactly the
    (buggy) behavior it asked for rather than a silent fix."""
    return F.mse_loss(s_next_hat, s_next_target)


def compute_losses(
    output: dict[str, torch.Tensor],
    target_tensors: dict[str, torch.Tensor],
    lm_labels: torch.Tensor,
    s_next_target: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble every applicable loss term from one forward pass's output.

    `output` is an `HGNativeModel` forward-pass dict; for `variant="baseline"`
    only `"lm"` is returned. `s_next_target`, if given, must already be a
    stop-gradient tensor (see `transition_loss`).
    """
    losses = {"lm": lm_loss(output["logits"], lm_labels)}
    if "decision_logits" not in output:  # baseline variant
        return losses

    losses["decision"] = decision_loss(output["decision_logits"], target_tensors["decision_label"], target_tensors["decision_defined"])
    losses["quotient"] = quotient_loss(output["distortion_hat"], target_tensors["distortion_vec"], target_tensors["distortion_mask"])
    losses["info_deficit"] = info_deficit_loss(output["info_deficit_logit"], target_tensors["info_deficit"])
    losses["accessibility"] = accessibility_loss(output["accessibility_logit"], target_tensors["accessibility"], target_tensors["accessibility_defined"])
    losses["route"] = route_loss(output["route_logits"], target_tensors["route_label"], target_tensors["route_defined"])
    if s_next_target is not None and "s_next_hat" in output:
        losses["transition"] = transition_loss(output["s_next_hat"], s_next_target)
    return losses


def total_loss(losses: dict[str, torch.Tensor], loss_weights: LossWeights, hg_multiplier: float = 1.0) -> torch.Tensor:
    """Sum losses per spec section 7's L_HG, with the warm-up multiplier (section 7)
    applied to every HG term but not L_LM."""
    total = loss_weights.lm * losses["lm"]
    hg_weight_by_name = {
        "decision": loss_weights.decision,
        "quotient": loss_weights.quotient,
        "transition": loss_weights.transition,
        "info_deficit": loss_weights.info_deficit,
        "accessibility": loss_weights.accessibility,
        "route": loss_weights.route,
    }
    for name, weight in hg_weight_by_name.items():
        if name in losses:
            total = total + hg_multiplier * weight * losses[name]
    return total


def hg_multiplier_for_progress(schedule: HGLossSchedule, step: int, total_steps: int) -> float:
    """Convenience: `schedule.hg_multiplier` from a (step, total_steps) pair."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    return schedule.hg_multiplier(step / total_steps)
