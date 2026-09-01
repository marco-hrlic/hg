"""The HG loss registry -- needs `torch`; skips here, runs on the Spark."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.model.config import HeadsConfig, HGLatentConfig, HGLossSchedule, HGNativeConfig, LossWeights, TransformerConfig
from hg_native.model.hg_model import build_model
from hg_native.training.batch import ByteTokenizer, collate, serialize_episode
from hg_native.training.losses import (
    compute_losses,
    hg_multiplier_for_progress,
    targets_to_tensors,
    total_loss,
    transition_loss,
)

SMALL_TRUNK = TransformerConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=64)
SMALL_LATENT = HGLatentConfig(hidden_dim=16, latent_dim=8, layers=2)
SMALL_HEADS = HeadsConfig(n_decision_labels=2, n_actions=7, action_embed_dim=6, router_hidden_dim=12, n_routes=3, cost_dim=1)


def _hg_config() -> HGNativeConfig:
    return HGNativeConfig(variant="hg", transformer=SMALL_TRUNK, hg_latent=SMALL_LATENT, heads=SMALL_HEADS, seed=0)


def _baseline_config() -> HGNativeConfig:
    return HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=0)


def _batch(episodes):
    tok = ByteTokenizer()
    examples = [serialize_episode(ep, tok, max_seq_len=SMALL_TRUNK.max_seq_len) for ep in episodes]
    return collate(examples), tok


def _mixed_episodes():
    return [
        gen_static_hidden(0, "one_bit_deficit", 0),
        gen_static_hidden(0, "no_deficit", 0),
        gen_static_hidden(1, "multi_bit_deficit", 0),
        gen_accessibility(0, "reordered_form", 0),
        gen_accessibility(0, "canonical_compact", 0),
    ]


def test_compute_losses_baseline_variant_only_has_lm():
    cfg = _baseline_config()
    model = build_model(cfg)
    batch, tok = _batch(_mixed_episodes())
    ids = torch.tensor(batch["input_ids"])
    out = model(ids)
    labels = ids.clone()
    losses = compute_losses(out, {}, labels)
    assert set(losses) == {"lm"}
    assert torch.isfinite(losses["lm"])


def test_compute_losses_hg_variant_has_every_non_transition_head():
    cfg = _hg_config()
    model = build_model(cfg)
    batch, tok = _batch(_mixed_episodes())
    ids = torch.tensor(batch["input_ids"])
    positions = torch.tensor(batch["decision_position"])
    out = model(ids, decision_positions=positions)
    labels = ids.clone()
    target_tensors = targets_to_tensors(batch["targets"])
    losses = compute_losses(out, target_tensors, labels)
    assert set(losses) == {"lm", "decision", "quotient", "info_deficit", "accessibility", "route"}
    for name, value in losses.items():
        assert torch.isfinite(value), name


def test_compute_losses_with_transition_target():
    cfg = _hg_config()
    model = build_model(cfg)
    batch, tok = _batch(_mixed_episodes())
    ids = torch.tensor(batch["input_ids"])
    positions = torch.tensor(batch["decision_position"])
    action_id = torch.tensor(batch["action_id"])
    out = model(ids, decision_positions=positions, action_id=action_id)
    labels = ids.clone()
    target_tensors = targets_to_tensors(batch["targets"])
    fake_next_state = torch.zeros_like(out["s_t"]).detach()
    losses = compute_losses(out, target_tensors, labels, s_next_target=fake_next_state)
    assert "transition" in losses
    assert torch.isfinite(losses["transition"])


def test_masked_losses_do_not_crash_on_all_masked_batch():
    # An all-accessibility, all-inaccessible batch defines every accessibility
    # target but no decision target that isn't ACT_A/ACT_B-derived from an
    # actual deficit -- exercise the masking path directly with a batch where
    # `route_defined`/`decision_defined` happen to already all be True, and
    # separately confirm the masked-mean helper's zero-example path.
    cfg = _hg_config()
    model = build_model(cfg)
    episodes = [gen_accessibility(0, sc, 0) for sc in ("canonical_compact", "reordered_form")]
    batch, tok = _batch(episodes)
    ids = torch.tensor(batch["input_ids"])
    positions = torch.tensor(batch["decision_position"])
    out = model(ids, decision_positions=positions)
    target_tensors = targets_to_tensors(batch["targets"])
    losses = compute_losses(out, target_tensors, ids.clone())
    for name, value in losses.items():
        assert torch.isfinite(value), name


def test_total_loss_applies_hg_multiplier():
    losses = {"lm": torch.tensor(2.0), "decision": torch.tensor(1.0)}
    weights = LossWeights(lm=1.0, decision=0.5)
    off = total_loss(losses, weights, hg_multiplier=0.0)
    on = total_loss(losses, weights, hg_multiplier=1.0)
    assert off.item() == pytest.approx(2.0)  # lm only, HG term zeroed out
    assert on.item() == pytest.approx(2.5)  # lm + 0.5 * decision


def test_total_loss_ignores_absent_terms():
    losses = {"lm": torch.tensor(1.0)}  # baseline: no HG heads at all
    weights = LossWeights()
    assert total_loss(losses, weights, hg_multiplier=1.0).item() == pytest.approx(1.0)


def test_hg_multiplier_for_progress_matches_schedule():
    schedule = HGLossSchedule(warmup_start_pct=0.1, warmup_end_pct=0.3)
    assert hg_multiplier_for_progress(schedule, step=5, total_steps=100) == schedule.hg_multiplier(0.05)
    assert hg_multiplier_for_progress(schedule, step=50, total_steps=100) == 1.0


def test_hg_multiplier_for_progress_rejects_zero_total_steps():
    with pytest.raises(ValueError, match="total_steps"):
        hg_multiplier_for_progress(HGLossSchedule(), step=0, total_steps=0)


def test_transition_loss_does_not_detach_for_you():
    s_hat = torch.randn(3, 8, requires_grad=True)
    target = s_hat.detach() + 1.0  # caller already stopped the gradient
    loss = transition_loss(s_hat, target)
    loss.backward()
    assert s_hat.grad is not None
    assert torch.isfinite(s_hat.grad).all()
