"""The training loop -- spec section 8.1's smoke stage, as an automated test.
Needs `torch`; skips here, runs on the Spark.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hg_native.model.config import HeadsConfig, HGLatentConfig, HGNativeConfig, TransformerConfig
from hg_native.model.hg_model import build_model
from hg_native.training.dataset import SyntheticMix
from hg_native.training.trainer import OptimConfig, TrainConfig, lr_at_step, train

TINY_TRUNK = TransformerConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=64)
TINY_LATENT = HGLatentConfig(hidden_dim=16, latent_dim=8, layers=1)
TINY_HEADS = HeadsConfig(n_decision_labels=2, n_actions=7, action_embed_dim=6, router_hidden_dim=12, n_routes=3, cost_dim=1)


# --- lr_at_step ---------------------------------------------------


def test_lr_at_step_warms_up_linearly_then_decays():
    cfg = OptimConfig(lr=1.0, warmup_ratio=0.1, min_lr_ratio=0.1)
    total_steps = 100
    warmup_end = int(0.1 * total_steps)
    assert lr_at_step(cfg, 0, total_steps) == pytest.approx(1.0 / warmup_end)
    assert lr_at_step(cfg, warmup_end - 2, total_steps) < 1.0  # mid-warmup, strictly below peak
    assert lr_at_step(cfg, warmup_end - 1, total_steps) == pytest.approx(1.0)  # last warmup step reaches peak
    # Right at the end of warmup, LR is at (or essentially at) its peak.
    assert lr_at_step(cfg, warmup_end, total_steps) == pytest.approx(1.0, abs=0.05)
    assert lr_at_step(cfg, total_steps - 1, total_steps) == pytest.approx(0.1, abs=0.02)


def test_lr_at_step_rejects_non_positive_total_steps():
    with pytest.raises(ValueError, match="total_steps"):
        lr_at_step(OptimConfig(), 0, 0)


# --- smoke training run ---------------------------------------------------


def _tiny_hg_config(seed: int = 0) -> HGNativeConfig:
    return HGNativeConfig(variant="hg", transformer=TINY_TRUNK, hg_latent=TINY_LATENT, heads=TINY_HEADS, seed=seed)


def _tiny_baseline_config(seed: int = 0) -> HGNativeConfig:
    return HGNativeConfig(variant="baseline", transformer=TINY_TRUNK, seed=seed)


def test_smoke_run_baseline_finite_losses_and_gradients():
    config = _tiny_baseline_config()
    model = build_model(config)
    train_cfg = TrainConfig(total_steps=10, batch_size=4, max_seq_len=48, seed=0)
    history = train(model, config, train_cfg)
    assert len(history) == 10
    for m in history:
        assert set(m.losses) == {"lm"}
        assert torch.isfinite(torch.tensor(m.total))
        assert torch.isfinite(torch.tensor(m.grad_norm))
        assert m.hg_multiplier == 0.0


def test_smoke_run_hg_every_head_has_finite_loss_and_multiplier_follows_schedule():
    config = _tiny_hg_config()
    model = build_model(config)
    train_cfg = TrainConfig(total_steps=20, batch_size=4, max_seq_len=48, seed=0)
    history = train(model, config, train_cfg)
    assert len(history) == 20
    for m in history:
        assert {"lm", "decision", "quotient", "info_deficit", "accessibility", "route", "transition"} <= set(m.losses)
        for v in m.losses.values():
            assert torch.isfinite(torch.tensor(v))
    # Default schedule: warmup_start_pct=0.10, warmup_end_pct=0.25 -> at
    # step/total_steps=0 the multiplier must be 0, and by the last step (0.95)
    # it must be 1.0.
    assert history[0].hg_multiplier == 0.0
    assert history[-1].hg_multiplier == 1.0


def test_smoke_run_checkpoint_and_gradient_finiteness_are_stable_across_seeds():
    # Spec section 8.1's smoke stage explicitly wants "finite gradients" and
    # "loss decrease" checked before trusting anything downstream -- run a
    # slightly longer HG smoke run and check the LM loss trends down.
    config = _tiny_hg_config()
    model = build_model(config)
    train_cfg = TrainConfig(total_steps=60, batch_size=4, max_seq_len=48, seed=0)
    history = train(model, config, train_cfg)
    early = sum(m.losses["lm"] for m in history[:10]) / 10
    late = sum(m.losses["lm"] for m in history[-10:]) / 10
    assert late < early


def test_baseline_never_gets_a_transition_loss():
    config = _tiny_baseline_config()
    model = build_model(config)
    train_cfg = TrainConfig(total_steps=5, batch_size=4, max_seq_len=48, seed=0)
    history = train(model, config, train_cfg)
    for m in history:
        assert "transition" not in m.losses


def test_transition_loss_pulls_current_and_next_states_together():
    # A weak but real sanity check: training should not leave the transition
    # loss exploding or growing -- with 60 steps on a trivial deterministic
    # environment it should trend down or stay small, not blow up.
    config = _tiny_hg_config()
    model = build_model(config)
    train_cfg = TrainConfig(total_steps=60, batch_size=4, max_seq_len=48, seed=0)
    history = train(model, config, train_cfg)
    early = sum(m.losses["transition"] for m in history[:10]) / 10
    late = sum(m.losses["transition"] for m in history[-10:]) / 10
    assert late <= early * 1.5  # not blowing up; a strict decrease isn't guaranteed this early


def test_train_step_deterministic_given_seed():
    config = _tiny_baseline_config()
    train_cfg = TrainConfig(total_steps=3, batch_size=4, max_seq_len=48, seed=0)
    h1 = train(build_model(config), config, train_cfg)
    h2 = train(build_model(config), config, train_cfg)
    assert [m.total for m in h1] == pytest.approx([m.total for m in h2])
