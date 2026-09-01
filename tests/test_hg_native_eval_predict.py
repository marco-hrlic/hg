"""`eval.predict.predict` -- needs `torch`; skips here, runs on the Spark."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hg_native.data.accessibility import generate_episode as gen_accessibility
from hg_native.data.static_hidden import generate_episode as gen_static_hidden
from hg_native.eval.predict import predict
from hg_native.model.config import DEFAULT_DECISION_LABELS, DEFAULT_ROUTES, HeadsConfig, HGLatentConfig, HGNativeConfig, TransformerConfig
from hg_native.model.hg_model import build_model
from hg_native.training.batch import ByteTokenizer

SMALL_TRUNK = TransformerConfig(vocab_size=259, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=64)
SMALL_LATENT = HGLatentConfig(hidden_dim=16, latent_dim=8, layers=1)
SMALL_HEADS = HeadsConfig(n_decision_labels=2, n_actions=7, action_embed_dim=6, router_hidden_dim=12, n_routes=3, cost_dim=1)


def _episodes():
    return [
        gen_static_hidden(0, "one_bit_deficit", 0),
        gen_static_hidden(0, "no_deficit", 1),
        gen_accessibility(0, "reordered_form", 0),
        gen_accessibility(0, "canonical_compact", 1),
    ]


def test_predict_rejects_baseline_model():
    cfg = HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=0)
    model = build_model(cfg)
    with pytest.raises(ValueError, match="requires an HG-variant model"):
        predict(model, ByteTokenizer(), _episodes(), max_seq_len=48)


def test_predict_shapes_and_ranges():
    cfg = HGNativeConfig(variant="hg", transformer=SMALL_TRUNK, hg_latent=SMALL_LATENT, heads=SMALL_HEADS, seed=0)
    model = build_model(cfg)
    episodes = _episodes()
    preds = predict(model, ByteTokenizer(), episodes, max_seq_len=48)

    assert preds.episode_id == [ep.episode_id for ep in episodes]
    assert preds.info_deficit_prob.shape == (len(episodes),)
    assert preds.accessibility_prob.shape == (len(episodes),)
    assert ((preds.info_deficit_prob >= 0) & (preds.info_deficit_prob <= 1)).all()
    assert ((preds.accessibility_prob >= 0) & (preds.accessibility_prob <= 1)).all()
    assert set(preds.decision_pred.tolist()) <= set(range(len(DEFAULT_DECISION_LABELS)))
    assert set(preds.route_pred.tolist()) <= set(range(len(DEFAULT_ROUTES)))
    assert preds.decision_label(0) in DEFAULT_DECISION_LABELS
    assert preds.route_label(0) in DEFAULT_ROUTES


def test_predict_batches_without_changing_results():
    cfg = HGNativeConfig(variant="hg", transformer=SMALL_TRUNK, hg_latent=SMALL_LATENT, heads=SMALL_HEADS, seed=0)
    model = build_model(cfg)
    episodes = _episodes() * 5  # 20 episodes
    small_batches = predict(model, ByteTokenizer(), episodes, max_seq_len=48, batch_size=3)
    one_batch = predict(model, ByteTokenizer(), episodes, max_seq_len=48, batch_size=1000)
    assert torch.allclose(torch.tensor(small_batches.info_deficit_prob), torch.tensor(one_batch.info_deficit_prob), atol=1e-5)
    assert (small_batches.decision_pred == one_batch.decision_pred).all()
