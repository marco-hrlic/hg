"""HG-native model configuration and analytic parameter counting (spec section 3).

Mirrors `hg.config`'s dataclass-plus-`from_dict`-plus-`ConfigError` style
(`ModelConfig`/`DatasetConfig` in `src/hg/config.py`) and reuses its YAML
reader (`read_config_file`) rather than re-implementing `${VAR}`-expanding
YAML loading.

`estimate_param_count` is pure Python/arithmetic — no `torch` import, no
GPU needed — so the parameter-matching requirement in spec section 2.1
(`(N_HG - N_baseline) / N_baseline < 0.05`) and the F0 falsification control
("HG improves router metrics only because it has more parameters", checked
via parameter-match within 1-2%) can be checked from a config file alone,
before any model is built or run on the Spark. `tests/test_hg_native_model_shapes.py`
cross-checks this estimate against `sum(p.numel() for p in model.parameters())`
on an actual instantiated model, wherever `torch` is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from hg.config import ConfigError, read_config_file

# spec section 3.3: the M0 route set. RETRIEVE/DELIBERATE are postponed.
DEFAULT_ROUTES = ("ACT", "THINK", "ASK")
# spec section 10.2: M0 binary decision tasks.
DEFAULT_DECISION_LABELS = ("A", "B")


def _reject_unknown(name: str, d: Mapping[str, Any], known: set[str], required: set[str] = frozenset()) -> None:
    unknown = set(d) - known
    if unknown:
        raise ConfigError(f"{name}: unknown keys {sorted(unknown)}")
    missing = required - set(d)
    if missing:
        raise ConfigError(f"{name}: missing required keys {sorted(missing)}")


@dataclass(slots=True)
class TransformerConfig:
    """The shared trunk (spec section 3.1). Baseline and HG use one instance."""

    vocab_size: int = 16000
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    d_ff: int = 2048
    max_seq_len: int = 1024
    norm: str = "rmsnorm"
    activation: str = "swiglu"
    positional_encoding: str = "rope"
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ConfigError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ConfigError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        if self.norm != "rmsnorm":
            raise ConfigError(f"only norm='rmsnorm' is implemented, got {self.norm!r}")
        if self.activation != "swiglu":
            raise ConfigError(f"only activation='swiglu' is implemented, got {self.activation!r}")
        if self.positional_encoding != "rope":
            raise ConfigError(f"only positional_encoding='rope' is implemented, got {self.positional_encoding!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TransformerConfig":
        _reject_unknown("transformer", d, set(cls.__dataclass_fields__))
        return cls(**d)


@dataclass(slots=True)
class HGLatentConfig:
    """The compact decision latent S_t = f_HG(h_t) (spec section 3.2).

    `input_dim=None` (the common case) defers to the trunk's `d_model` at
    model-build time, so a config file need not repeat it.
    """

    input_dim: int | None = None
    hidden_dim: int = 256
    latent_dim: int = 64
    layers: int = 2
    activation: str = "silu"
    normalization: str = "layernorm"

    def __post_init__(self) -> None:
        if self.layers < 1:
            raise ConfigError(f"hg_latent.layers must be >= 1, got {self.layers}")
        if self.activation != "silu":
            raise ConfigError(f"only hg_latent.activation='silu' is implemented, got {self.activation!r}")
        if self.normalization != "layernorm":
            raise ConfigError(f"only hg_latent.normalization='layernorm' is implemented, got {self.normalization!r}")

    def resolved_input_dim(self, trunk: TransformerConfig) -> int:
        return self.input_dim if self.input_dim is not None else trunk.d_model

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HGLatentConfig":
        _reject_unknown("hg_latent", d, set(cls.__dataclass_fields__))
        return cls(**d)


@dataclass(slots=True)
class HeadsConfig:
    """Sizes for the HG heads (spec section 3.3).

    Not fully pinned by the spec (only the router's route set and the
    decision head's binary label space are explicit there) — the rest
    (action vocabulary size for the distortion/transition heads, router
    hidden width, cost-vector width `c_t`) are implementation choices,
    flagged here rather than silently assumed, per
    [[hg-theory-implementation-split]].
    """

    n_decision_labels: int = len(DEFAULT_DECISION_LABELS)
    # Upper bound on distinct discrete actions the distortion/transition
    # heads need to size for (ACT_A, ACT_B, ASK_h1..ASK_hk, ...). Family A's
    # generator (`static_hidden.py`) never needs more than 4 (2 hidden vars +
    # 2 ACT actions); left configurable since later families may need more.
    n_actions: int = 8
    action_embed_dim: int = 16
    router_hidden_dim: int = 64
    n_routes: int = len(DEFAULT_ROUTES)
    # Width of c_t, the router's explicitly-allowed cost variables (spec
    # section 3.3); 1 is a placeholder scalar price until a real cost vector
    # is defined.
    cost_dim: int = 1

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HeadsConfig":
        _reject_unknown("heads", d, set(cls.__dataclass_fields__))
        return cls(**d)


@dataclass(slots=True)
class LossWeights:
    """Initial conservative weights (spec section 7)."""

    lm: float = 1.0
    decision: float = 0.25
    quotient: float = 0.10
    transition: float = 0.05
    info_deficit: float = 0.10
    accessibility: float = 0.10
    route: float = 0.10

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LossWeights":
        _reject_unknown("loss_weights", d, set(cls.__dataclass_fields__))
        return cls(**d)


@dataclass(slots=True)
class HGLossSchedule:
    """Warm-up schedule for the HG loss multiplier (spec section 7)."""

    warmup_start_pct: float = 0.10
    warmup_end_pct: float = 0.25

    def __post_init__(self) -> None:
        if not (0.0 <= self.warmup_start_pct <= self.warmup_end_pct <= 1.0):
            raise ConfigError(
                "hg_loss_schedule: require 0 <= warmup_start_pct <= warmup_end_pct <= 1, "
                f"got {self.warmup_start_pct}, {self.warmup_end_pct}"
            )

    def hg_multiplier(self, progress: float) -> float:
        """0 before warmup_start_pct, linear ramp to 1 at warmup_end_pct, 1 after."""
        if progress <= self.warmup_start_pct:
            return 0.0
        if progress >= self.warmup_end_pct:
            return 1.0
        span = self.warmup_end_pct - self.warmup_start_pct
        return (progress - self.warmup_start_pct) / span

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HGLossSchedule":
        _reject_unknown("hg_loss_schedule", d, set(cls.__dataclass_fields__))
        return cls(**d)


@dataclass(slots=True)
class HGNativeConfig:
    """Top-level config for one model: either `variant="baseline"` or `"hg"`."""

    variant: str = "baseline"
    seed: int = 0
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    hg_latent: HGLatentConfig | None = None
    heads: HeadsConfig = field(default_factory=HeadsConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    loss_schedule: HGLossSchedule = field(default_factory=HGLossSchedule)

    def __post_init__(self) -> None:
        if self.variant not in ("baseline", "hg"):
            raise ConfigError(f"variant must be 'baseline' or 'hg', got {self.variant!r}")
        if self.variant == "hg" and self.hg_latent is None:
            self.hg_latent = HGLatentConfig()
        if self.variant == "baseline" and self.hg_latent is not None:
            raise ConfigError("variant='baseline' must not set hg_latent")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HGNativeConfig":
        d = dict(d)
        known = {"variant", "seed", "transformer", "hg_latent", "heads", "loss_weights", "loss_schedule"}
        _reject_unknown("hg_native config", d, known)
        return cls(
            variant=d.get("variant", "baseline"),
            seed=int(d.get("seed", 0)),
            transformer=TransformerConfig.from_dict(d.get("transformer", {}) or {}),
            hg_latent=HGLatentConfig.from_dict(d["hg_latent"]) if d.get("hg_latent") else None,
            heads=HeadsConfig.from_dict(d.get("heads", {}) or {}),
            loss_weights=LossWeights.from_dict(d.get("loss_weights", {}) or {}),
            loss_schedule=HGLossSchedule.from_dict(d.get("loss_schedule", {}) or {}),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "HGNativeConfig":
        return cls.from_dict(read_config_file(path))


# --- analytic parameter counting (no torch needed) --------------------------


def _trunk_param_count(t: TransformerConfig) -> int:
    """Exact count for the RoPE/RMSNorm/SwiGLU/GQA/tied-embedding trunk.

    RoPE contributes no learned parameters. Attention and MLP projections are
    bias-free, matching the Llama-family convention `norm=rmsnorm` /
    `activation=swiglu` implies and `transformer.py` implements.
    """
    kv_dim = t.head_dim * t.n_kv_heads
    attn = 2 * t.d_model * t.d_model + 2 * t.d_model * kv_dim  # q, o : k, v
    mlp = 3 * t.d_model * t.d_ff  # SwiGLU: gate, up, down
    norms_per_layer = 2 * t.d_model  # pre-attn + pre-mlp RMSNorm
    per_layer = attn + mlp + norms_per_layer
    embedding = t.vocab_size * t.d_model  # counted once: tied with the LM head
    final_norm = t.d_model
    return t.n_layers * per_layer + embedding + final_norm


def _mlp_stack_param_count(input_dim: int, hidden_dim: int, output_dim: int, n_hidden_layers: int) -> int:
    """`n_hidden_layers` Linear(*, hidden_dim) blocks then Linear(hidden_dim, output_dim). Biased."""
    if n_hidden_layers < 1:
        raise ValueError("n_hidden_layers must be >= 1")
    total = input_dim * hidden_dim + hidden_dim  # first hidden layer
    total += (n_hidden_layers - 1) * (hidden_dim * hidden_dim + hidden_dim)  # remaining hidden layers
    total += hidden_dim * output_dim + output_dim  # output projection
    return total


def _hg_latent_param_count(h: HGLatentConfig, trunk: TransformerConfig) -> int:
    mlp = _mlp_stack_param_count(h.resolved_input_dim(trunk), h.hidden_dim, h.latent_dim, h.layers)
    layernorm = 2 * h.latent_dim  # weight + bias
    return mlp + layernorm


def _heads_param_count(heads: HeadsConfig, latent_dim: int) -> int:
    decision = latent_dim * heads.n_decision_labels + heads.n_decision_labels
    distortion = latent_dim * heads.n_actions + heads.n_actions
    info_deficit = latent_dim * 1 + 1
    accessibility = latent_dim * 1 + 1
    transition_embed = heads.n_actions * heads.action_embed_dim
    transition_proj = (latent_dim + heads.action_embed_dim) * latent_dim + latent_dim
    router_in_dim = latent_dim + 2 + heads.cost_dim  # S_t, R_info_hat, R_access_hat, c_t
    router = _mlp_stack_param_count(router_in_dim, heads.router_hidden_dim, heads.n_routes, n_hidden_layers=1)
    return decision + distortion + info_deficit + accessibility + transition_embed + transition_proj + router


def estimate_param_count(config: HGNativeConfig) -> int:
    """Exact parameter count implied by `config`, without building a model."""
    total = _trunk_param_count(config.transformer)
    if config.variant == "hg":
        assert config.hg_latent is not None  # enforced by __post_init__
        latent_count = _hg_latent_param_count(config.hg_latent, config.transformer)
        heads_count = _heads_param_count(config.heads, config.hg_latent.latent_dim)
        total += latent_count + heads_count
    return total


def param_count_overhead(hg_config: HGNativeConfig, baseline_config: HGNativeConfig) -> float:
    """(N_HG - N_baseline) / N_baseline, the spec section 2.1 / F0 quantity."""
    if hg_config.variant != "hg" or baseline_config.variant != "baseline":
        raise ValueError("param_count_overhead expects an ('hg', 'baseline') config pair")
    n_hg = estimate_param_count(hg_config)
    n_base = estimate_param_count(baseline_config)
    return (n_hg - n_base) / n_base
