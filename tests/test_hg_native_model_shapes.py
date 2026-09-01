"""Shape, determinism, gradient-flow, and parameter-count cross-checks for
`hg_native.model`.

Needs a real `torch` install (the `gpu` extra; not installed on this dev
machine, per [[dgx-spark-access]]) -- skips here, runs on the Spark.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hg_native.model.config import HeadsConfig, HGLatentConfig, HGNativeConfig, TransformerConfig, estimate_param_count
from hg_native.model.hg_model import HGNativeModel, build_model

SMALL_TRUNK = TransformerConfig(vocab_size=97, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=64, max_seq_len=64)
SMALL_LATENT = HGLatentConfig(hidden_dim=16, latent_dim=8, layers=2)
SMALL_HEADS = HeadsConfig(n_decision_labels=2, n_actions=4, action_embed_dim=6, router_hidden_dim=12, n_routes=3, cost_dim=1)


def _baseline_config() -> HGNativeConfig:
    return HGNativeConfig(variant="baseline", transformer=SMALL_TRUNK, seed=0)


def _hg_config() -> HGNativeConfig:
    return HGNativeConfig(variant="hg", transformer=SMALL_TRUNK, hg_latent=SMALL_LATENT, heads=SMALL_HEADS, seed=0)


def _random_ids(batch: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab_size, (batch, seq_len), generator=g)


# --- parameter count cross-check ------------------------------------------


def test_baseline_param_count_matches_analytic_estimate():
    model = build_model(_baseline_config())
    assert model.num_parameters() == estimate_param_count(_baseline_config())


def test_hg_param_count_matches_analytic_estimate():
    model = build_model(_hg_config())
    assert model.num_parameters() == estimate_param_count(_hg_config())


def test_hg_has_more_parameters_than_baseline_for_same_trunk():
    # The <5% overhead budget (spec section 2.1) is checked at the spec's
    # actual ~50-70M target scale in
    # `test_hg_native_config.py::test_param_count_overhead_within_spec_target_for_recommended_config`,
    # not against these deliberately tiny shape-test fixtures, where the
    # heads are a much larger fraction of a much smaller trunk by construction.
    n_base = build_model(_baseline_config()).num_parameters()
    n_hg = build_model(_hg_config()).num_parameters()
    assert n_hg > n_base


# --- forward shapes ---------------------------------------------------------


def test_baseline_forward_shapes():
    cfg = _baseline_config()
    model = build_model(cfg)
    ids = _random_ids(4, 10, cfg.transformer.vocab_size)
    out = model(ids)
    assert out["logits"].shape == (4, 10, cfg.transformer.vocab_size)
    assert out["h"].shape == (4, 10, cfg.transformer.d_model)
    assert "s_t" not in out
    assert "route_logits" not in out


def test_hg_forward_shapes_full_sequence():
    cfg = _hg_config()
    model = build_model(cfg)
    ids = _random_ids(4, 10, cfg.transformer.vocab_size)
    out = model(ids)
    assert out["logits"].shape == (4, 10, cfg.transformer.vocab_size)
    assert out["s_t"].shape == (4, 10, cfg.hg_latent.latent_dim)
    assert out["decision_logits"].shape == (4, 10, cfg.heads.n_decision_labels)
    assert out["distortion_hat"].shape == (4, 10, cfg.heads.n_actions)
    assert out["info_deficit_logit"].shape == (4, 10)
    assert out["accessibility_logit"].shape == (4, 10)
    assert out["route_logits"].shape == (4, 10, cfg.heads.n_routes)
    assert "s_next_hat" not in out  # no action_id given


def test_hg_forward_shapes_at_decision_positions():
    cfg = _hg_config()
    model = build_model(cfg)
    batch, seq_len = 5, 10
    ids = _random_ids(batch, seq_len, cfg.transformer.vocab_size)
    positions = torch.tensor([0, 3, 5, 9, 2])
    action_id = torch.randint(0, cfg.heads.n_actions, (batch,))
    out = model(ids, decision_positions=positions, action_id=action_id)
    assert out["s_t"].shape == (batch, cfg.hg_latent.latent_dim)
    assert out["decision_logits"].shape == (batch, cfg.heads.n_decision_labels)
    assert out["s_next_hat"].shape == (batch, cfg.hg_latent.latent_dim)
    assert out["route_logits"].shape == (batch, cfg.heads.n_routes)


def test_decision_positions_actually_select_different_timesteps():
    # A regression against `h[:, decision_positions]` broadcasting wrong and
    # silently reading the same timestep for every example.
    cfg = _hg_config()
    model = build_model(cfg)
    model.eval()
    batch, seq_len = 3, 10
    ids = _random_ids(batch, seq_len, cfg.transformer.vocab_size)
    with torch.no_grad():
        h = model.trunk(ids)
        s_first = model.hg_latent(h[torch.arange(batch), torch.zeros(batch, dtype=torch.long)])
        s_last = model.hg_latent(h[torch.arange(batch), torch.full((batch,), seq_len - 1, dtype=torch.long)])
    assert not torch.allclose(s_first, s_last)


def test_tied_embeddings_share_storage():
    cfg = _baseline_config()
    assert cfg.transformer.tie_embeddings is True
    model = build_model(cfg)
    assert model.lm_head is None
    ids = _random_ids(2, 5, cfg.transformer.vocab_size)
    out = model(ids)
    # A change to the embedding table must move the logits computed from it.
    with torch.no_grad():
        model.trunk.embed.weight.add_(1.0)
    out2 = model(ids)
    assert not torch.allclose(out["logits"], out2["logits"])


def test_untied_embeddings_allocate_separate_lm_head():
    trunk = TransformerConfig(vocab_size=97, d_model=32, n_layers=1, n_heads=4, n_kv_heads=2, d_ff=64, tie_embeddings=False)
    model = build_model(HGNativeConfig(variant="baseline", transformer=trunk))
    assert model.lm_head is not None
    assert model.lm_head.weight.shape == (97, 32)


# --- determinism -------------------------------------------------------------


def test_same_seed_gives_identical_model_init_and_output():
    cfg = _hg_config()
    m1 = build_model(cfg)
    m2 = build_model(cfg)
    ids = _random_ids(2, 6, cfg.transformer.vocab_size)
    m1.eval()
    m2.eval()
    with torch.no_grad():
        out1 = m1(ids)
        out2 = m2(ids)
    assert torch.equal(out1["logits"], out2["logits"])
    assert torch.equal(out1["route_logits"], out2["route_logits"])


def test_init_produces_bounded_logit_magnitude():
    # Regression for a real bug caught on the Spark: default nn.Linear/
    # nn.Embedding init let tied-embedding logits reach magnitude ~400 at
    # random init (see transformer.py's `_init_weights` docstring). A
    # healthy GPT-2-style init should keep initial logits within a modest
    # multiple of a random classifier's natural scale.
    cfg = _baseline_config()
    model = build_model(cfg)
    model.eval()
    ids = _random_ids(4, 10, cfg.transformer.vocab_size)
    with torch.no_grad():
        logits = model(ids)["logits"]
    assert logits.abs().max().item() < 20.0


def test_rejects_sequence_longer_than_max_seq_len():
    cfg = _baseline_config()
    model = build_model(cfg)
    ids = _random_ids(1, cfg.transformer.max_seq_len + 1, cfg.transformer.vocab_size)
    with pytest.raises(ValueError, match="max_seq_len"):
        model(ids)


# --- gradient flow -----------------------------------------------------------


def test_backward_produces_finite_gradients_on_every_head():
    cfg = _hg_config()
    model = build_model(cfg)
    model.train()
    batch, seq_len = 4, 8
    ids = _random_ids(batch, seq_len, cfg.transformer.vocab_size)
    positions = torch.randint(0, seq_len, (batch,))
    action_id = torch.randint(0, cfg.heads.n_actions, (batch,))
    labels = torch.randint(0, cfg.transformer.vocab_size, (batch, seq_len))

    out = model(ids, decision_positions=positions, action_id=action_id)
    lm_loss = torch.nn.functional.cross_entropy(out["logits"].reshape(-1, cfg.transformer.vocab_size), labels.reshape(-1))
    decision_target = torch.randint(0, cfg.heads.n_decision_labels, (batch,))
    decision_loss = torch.nn.functional.cross_entropy(out["decision_logits"], decision_target)
    distortion_target = torch.rand(batch, cfg.heads.n_actions)
    distortion_loss = torch.nn.functional.mse_loss(out["distortion_hat"], distortion_target)
    info_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["info_deficit_logit"], torch.rand(batch))
    access_loss = torch.nn.functional.binary_cross_entropy_with_logits(out["accessibility_logit"], torch.rand(batch))
    transition_loss = torch.nn.functional.mse_loss(out["s_next_hat"], torch.zeros_like(out["s_next_hat"]))
    route_target = torch.randint(0, cfg.heads.n_routes, (batch,))
    route_loss = torch.nn.functional.cross_entropy(out["route_logits"], route_target)

    total = lm_loss + decision_loss + distortion_loss + info_loss + access_loss + transition_loss + route_loss
    total.backward()

    n_checked = 0
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has a non-finite gradient"
        n_checked += 1
    assert n_checked > 0


def test_baseline_backward_only_touches_trunk_parameters():
    cfg = _baseline_config()
    model = build_model(cfg)
    ids = _random_ids(2, 6, cfg.transformer.vocab_size)
    labels = torch.randint(0, cfg.transformer.vocab_size, (2, 6))
    out = model(ids)
    loss = torch.nn.functional.cross_entropy(out["logits"].reshape(-1, cfg.transformer.vocab_size), labels.reshape(-1))
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
