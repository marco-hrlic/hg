"""HGNativeConfig validation and analytic parameter counting.

Deliberately torch-free: `estimate_param_count` is pure arithmetic, so the
spec section 2.1 parameter-match requirement and the F0 falsification
control can be checked from a config file alone, on any machine, before a
model is ever built. `tests/test_hg_native_model_shapes.py` (torch-gated)
cross-checks the estimate against a real instantiated model's parameter
count.
"""

from __future__ import annotations

import pytest

from hg.config import ConfigError
from hg_native.model.config import (
    DEFAULT_DECISION_LABELS,
    DEFAULT_ROUTES,
    HeadsConfig,
    HGLatentConfig,
    HGLossSchedule,
    HGNativeConfig,
    LossWeights,
    TransformerConfig,
    estimate_param_count,
    param_count_overhead,
)


# --- TransformerConfig --------------------------------------------------


def test_transformer_config_defaults_match_spec_recommended_yaml():
    t = TransformerConfig()
    assert (t.vocab_size, t.n_layers, t.d_model, t.n_heads, t.n_kv_heads, t.d_ff, t.max_seq_len) == (
        16000,
        12,
        768,
        12,
        4,
        2048,
        1024,
    )
    assert t.tie_embeddings is True
    assert t.dropout == 0.0


def test_transformer_config_rejects_indivisible_heads():
    with pytest.raises(ConfigError, match="d_model"):
        TransformerConfig(d_model=100, n_heads=12)


def test_transformer_config_rejects_indivisible_kv_heads():
    with pytest.raises(ConfigError, match="n_heads"):
        TransformerConfig(n_heads=12, n_kv_heads=5)


def test_transformer_config_head_dim():
    assert TransformerConfig(d_model=768, n_heads=12).head_dim == 64


def test_transformer_config_from_dict_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown keys"):
        TransformerConfig.from_dict({"d_model": 768, "not_a_field": 1})


def test_transformer_config_only_implemented_variants_accepted():
    with pytest.raises(ConfigError, match="norm"):
        TransformerConfig(norm="layernorm")
    with pytest.raises(ConfigError, match="activation"):
        TransformerConfig(activation="gelu")
    with pytest.raises(ConfigError, match="positional_encoding"):
        TransformerConfig(positional_encoding="learned")


# --- HGLatentConfig / HeadsConfig / LossWeights / HGLossSchedule --------


def test_hg_latent_config_resolved_input_dim_defaults_to_trunk_d_model():
    trunk = TransformerConfig(d_model=768)
    latent = HGLatentConfig()
    assert latent.resolved_input_dim(trunk) == 768


def test_hg_latent_config_explicit_input_dim_overrides_trunk():
    trunk = TransformerConfig(d_model=768)
    latent = HGLatentConfig(input_dim=999)
    assert latent.resolved_input_dim(trunk) == 999


def test_hg_latent_config_rejects_zero_layers():
    with pytest.raises(ConfigError, match="layers"):
        HGLatentConfig(layers=0)


def test_loss_weights_defaults_match_spec():
    w = LossWeights()
    assert (w.lm, w.decision, w.quotient, w.transition, w.info_deficit, w.accessibility, w.route) == (
        1.0,
        0.25,
        0.10,
        0.05,
        0.10,
        0.10,
        0.10,
    )


def test_loss_schedule_multiplier_ramp():
    sched = HGLossSchedule(warmup_start_pct=0.10, warmup_end_pct=0.25)
    assert sched.hg_multiplier(0.0) == 0.0
    assert sched.hg_multiplier(0.10) == 0.0
    assert sched.hg_multiplier(0.25) == 1.0
    assert sched.hg_multiplier(1.0) == 1.0
    mid = sched.hg_multiplier(0.175)  # halfway through the ramp
    assert mid == pytest.approx(0.5)


def test_loss_schedule_rejects_inverted_bounds():
    with pytest.raises(ConfigError):
        HGLossSchedule(warmup_start_pct=0.5, warmup_end_pct=0.1)


def test_default_routes_and_labels():
    assert DEFAULT_ROUTES == ("ACT", "THINK", "ASK")
    assert DEFAULT_DECISION_LABELS == ("A", "B")
    assert HeadsConfig().n_routes == 3
    assert HeadsConfig().n_decision_labels == 2


# --- HGNativeConfig -------------------------------------------------------


def test_hg_native_config_baseline_has_no_hg_latent():
    cfg = HGNativeConfig(variant="baseline")
    assert cfg.hg_latent is None


def test_hg_native_config_hg_gets_default_hg_latent():
    cfg = HGNativeConfig(variant="hg")
    assert cfg.hg_latent is not None
    assert cfg.hg_latent.latent_dim == 64


def test_hg_native_config_rejects_hg_latent_on_baseline():
    with pytest.raises(ConfigError, match="baseline"):
        HGNativeConfig(variant="baseline", hg_latent=HGLatentConfig())


def test_hg_native_config_rejects_unknown_variant():
    with pytest.raises(ConfigError, match="variant"):
        HGNativeConfig(variant="frobnicate")


def test_hg_native_config_from_dict_round_trip():
    d = {
        "variant": "hg",
        "seed": 3,
        "transformer": {"d_model": 256, "n_heads": 8, "n_kv_heads": 4, "n_layers": 2, "d_ff": 512, "vocab_size": 1000},
        "hg_latent": {"hidden_dim": 64, "latent_dim": 16, "layers": 1},
        "loss_weights": {"lm": 1.0, "decision": 0.5},
    }
    cfg = HGNativeConfig.from_dict(d)
    assert cfg.variant == "hg"
    assert cfg.seed == 3
    assert cfg.transformer.d_model == 256
    assert cfg.hg_latent.latent_dim == 16
    assert cfg.loss_weights.decision == 0.5
    assert cfg.loss_weights.quotient == 0.10  # untouched default


def test_hg_native_config_from_dict_rejects_unknown_top_level_key():
    with pytest.raises(ConfigError, match="unknown keys"):
        HGNativeConfig.from_dict({"variant": "baseline", "not_a_field": 1})


def test_hg_native_config_from_yaml(tmp_path):
    path = tmp_path / "m0_hg_50m.yaml"
    path.write_text(
        "variant: hg\n"
        "seed: 7\n"
        "transformer:\n"
        "  d_model: 128\n"
        "  n_heads: 4\n"
        "  n_kv_heads: 2\n"
        "  n_layers: 2\n"
        "  d_ff: 256\n"
        "  vocab_size: 500\n"
    )
    cfg = HGNativeConfig.from_yaml(path)
    assert cfg.seed == 7
    assert cfg.transformer.d_model == 128


# --- analytic parameter counting -----------------------------------------


def test_estimate_param_count_positive_and_dominated_by_embedding_and_layers():
    cfg = HGNativeConfig(variant="baseline", transformer=TransformerConfig(vocab_size=1000, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128))
    n = estimate_param_count(cfg)
    assert n > 0
    # Hand-computed: embedding 1000*64=64000, final_norm=64.
    # attn = 2*64*64 + 2*64*32 = 8192+4096=12288; mlp=3*64*128=24576; norms=2*64=128
    # per_layer=12288+24576+128=37000 (wait: keep as a loose sanity bound instead of
    # re-deriving the exact formula here, since that duplicates config.py verbatim).
    assert n > 1000 * 64  # at least bigger than the embedding table alone


def test_estimate_param_count_hg_ge_baseline_for_same_trunk():
    trunk = TransformerConfig(vocab_size=1000, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128)
    baseline = HGNativeConfig(variant="baseline", transformer=trunk)
    hg = HGNativeConfig(variant="hg", transformer=trunk)
    assert estimate_param_count(hg) > estimate_param_count(baseline)


def test_estimate_param_count_exact_small_case():
    # Fully hand-computed, independent of config.py's own formula: vocab=10,
    # d_model=4, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=8.
    trunk = TransformerConfig(vocab_size=10, d_model=4, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=8)
    cfg = HGNativeConfig(variant="baseline", transformer=trunk)
    head_dim = 4 // 2  # 2
    kv_dim = head_dim * 1  # 2
    attn = 2 * 4 * 4 + 2 * 4 * kv_dim  # q+o : k+v = 32 + 16 = 48
    mlp = 3 * 4 * 8  # 96
    norms = 2 * 4  # 8
    per_layer = attn + mlp + norms  # 152
    embedding = 10 * 4  # 40
    final_norm = 4
    expected = 1 * per_layer + embedding + final_norm  # 196
    assert estimate_param_count(cfg) == expected


def test_param_count_overhead_within_spec_target_for_recommended_config():
    # spec section 2.1: (N_HG - N_baseline) / N_baseline < 0.05.
    baseline = HGNativeConfig(variant="baseline")
    hg = HGNativeConfig(variant="hg")
    overhead = param_count_overhead(hg, baseline)
    assert 0 <= overhead < 0.05


def test_param_count_overhead_rejects_wrong_variant_pair():
    cfg = HGNativeConfig(variant="hg")
    with pytest.raises(ValueError):
        param_count_overhead(cfg, cfg)


def test_recommended_config_trunk_size_is_a_known_spec_discrepancy():
    # The spec's own "recommended first configuration" (section 3.1) targets
    # "~50-70M total parameters" but its literal YAML values (vocab_size
    # 16000, 12 layers, d_model 768, d_ff 2048, GQA 12/4 heads) come out well
    # above that band once counted exactly -- the discrepancy this analytic
    # counter exists to catch per spec section 3.1's own instruction to
    # "compute exact parameter counts programmatically" before any long run.
    # Recorded here as a pinned regression rather than left to be
    # rediscovered by surprise when a real training run turns out ~90M.
    baseline = HGNativeConfig(variant="baseline")
    n = estimate_param_count(baseline)
    assert n > 70_000_000
    assert n == 87_804_672


def _m1_transformer() -> TransformerConfig:
    # M1 kickoff section 3's suggested starting architecture verbatim
    # (experiments/hg_native/configs/m1_{baseline,hg}.yaml).
    return TransformerConfig(vocab_size=16000, d_model=896, n_layers=16, n_heads=14, n_kv_heads=7, d_ff=2560, max_seq_len=512)


def test_m1_config_lands_in_the_130_to_170m_band_with_no_retuning():
    # Unlike M0's own literal recommended config (~87.8M against a stated
    # "~50-70M" target, see the discrepancy test above), the M1 kickoff's
    # suggested architecture lands where it should as-is -- pinned here so a
    # future edit to either the architecture or the counter formula gets
    # caught immediately rather than surfacing as a surprise ~150M run.
    trunk = _m1_transformer()
    baseline = HGNativeConfig(variant="baseline", transformer=trunk)
    hg = HGNativeConfig(
        variant="hg",
        transformer=trunk,
        hg_latent=HGLatentConfig(hidden_dim=384, latent_dim=96, layers=2),
        heads=HeadsConfig(n_actions=7),
    )
    n_baseline = estimate_param_count(baseline)
    n_hg = estimate_param_count(hg)
    assert n_baseline == 163_001_216
    assert n_hg == 163_549_278
    assert 130_000_000 <= n_baseline <= 170_000_000
    assert 130_000_000 <= n_hg <= 170_000_000


def test_m1_config_overhead_within_spec_target():
    trunk = _m1_transformer()
    baseline = HGNativeConfig(variant="baseline", transformer=trunk)
    hg = HGNativeConfig(
        variant="hg",
        transformer=trunk,
        hg_latent=HGLatentConfig(hidden_dim=384, latent_dim=96, layers=2),
        heads=HeadsConfig(n_actions=7),
    )
    overhead = param_count_overhead(hg, baseline)
    assert 0 <= overhead < 0.05
    assert overhead < 0.01  # measured ~0.34% -- comfortably inside the stricter <2% preference too
