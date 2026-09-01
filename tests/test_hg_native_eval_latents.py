"""`eval.latents.extract_latents` -- needs `torch`; skips here, runs on the Spark."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hg_native.data.sequential_quotient import generate_batch
from hg_native.eval.latents import extract_latents
from hg_native.model.config import HeadsConfig, HGLatentConfig, HGNativeConfig, TransformerConfig
from hg_native.model.hg_model import build_model
from hg_native.training.batch import ByteTokenizer

SMALL_TRUNK = TransformerConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=64)
SMALL_LATENT = HGLatentConfig(hidden_dim=16, latent_dim=8, layers=1)
SMALL_HEADS = HeadsConfig(n_decision_labels=2, n_actions=7, action_embed_dim=6, router_hidden_dim=12, n_routes=3, cost_dim=1)


def _episodes(n=10):
    return [traj[0] for traj in generate_batch(seed=0, n_trajectories=n, n_steps=1)]


def test_extract_latents_rejects_baseline_model():
    cfg = HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=0)
    model = build_model(cfg)
    with pytest.raises(ValueError, match="requires an HG-variant model"):
        extract_latents(model, ByteTokenizer(), _episodes(), max_seq_len=48)


def test_extract_latents_shape():
    cfg = HGNativeConfig(variant="hg", transformer=SMALL_TRUNK, hg_latent=SMALL_LATENT, heads=SMALL_HEADS, seed=0)
    model = build_model(cfg)
    episodes = _episodes(10)
    latents = extract_latents(model, ByteTokenizer(), episodes, max_seq_len=48)
    assert latents.shape == (10, SMALL_LATENT.latent_dim)


def test_extract_latents_batching_does_not_change_result():
    cfg = HGNativeConfig(variant="hg", transformer=SMALL_TRUNK, hg_latent=SMALL_LATENT, heads=SMALL_HEADS, seed=0)
    model = build_model(cfg)
    episodes = _episodes(20)
    small_batches = extract_latents(model, ByteTokenizer(), episodes, max_seq_len=48, batch_size=3)
    one_batch = extract_latents(model, ByteTokenizer(), episodes, max_seq_len=48, batch_size=1000)
    assert torch.allclose(torch.tensor(small_batches), torch.tensor(one_batch), atol=1e-5)
