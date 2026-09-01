"""Checkpoint save/load round-trip -- needs `torch`; skips here, runs on the Spark."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hg_native.model.config import HGNativeConfig, TransformerConfig
from hg_native.model.hg_model import build_model
from hg_native.training.checkpoint import load_checkpoint, save_checkpoint
from hg_native.training.trainer import OptimConfig, build_optimizer

SMALL_TRUNK = TransformerConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=32, max_seq_len=32)


def test_save_and_load_restores_weights_and_step(tmp_path):
    cfg = HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=0)
    model = build_model(cfg)
    optimizer = build_optimizer(model, OptimConfig())

    # Take one fake step so the optimizer has real state to round-trip.
    ids = torch.randint(0, SMALL_TRUNK.vocab_size, (2, 5))
    loss = model(ids)["logits"].sum()
    loss.backward()
    optimizer.step()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=7, extra={"note": "smoke"})

    fresh_model = build_model(HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=999))  # different seed init
    fresh_optimizer = build_optimizer(fresh_model, OptimConfig())
    before = [p.clone() for p in fresh_model.parameters()]

    info = load_checkpoint(path, fresh_model, fresh_optimizer)

    assert info["step"] == 7
    assert info["extra"] == {"note": "smoke"}
    after = list(fresh_model.parameters())
    assert not any(torch.equal(b, a) for b, a in zip(before, after))  # weights actually changed
    for p_orig, p_loaded in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p_orig, p_loaded)


def test_load_without_optimizer_only_restores_model(tmp_path):
    cfg = HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=0)
    model = build_model(cfg)
    optimizer = build_optimizer(model, OptimConfig())
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, optimizer, step=3)

    fresh_model = build_model(HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=1))
    info = load_checkpoint(path, fresh_model, optimizer=None)
    assert info["step"] == 3
    for p_orig, p_loaded in zip(model.parameters(), fresh_model.parameters()):
        assert torch.equal(p_orig, p_loaded)
