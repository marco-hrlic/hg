"""The shared baseline trunk: RoPE + RMSNorm + SwiGLU + grouped-query attention.

Both Baseline-50M and HG-50M use one instance of this trunk (spec section
2.1's matched-comparison requirement) — `HGNativeModel` in `hg_model.py`
adds the HG latent and heads on top of the same `h_t = F_theta(x_<=t)`
output this module produces.

`config.py`'s `_trunk_param_count` computes this module's exact parameter
count analytically, without importing torch; `tests/test_hg_native_model_shapes.py`
cross-checks the two agree for an actually-instantiated model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import TransformerConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rope_cache(seq_len: int, head_dim: int, device: torch.device, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (seq_len, head_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (batch, n_heads, seq_len, head_dim); cos/sin: (seq_len, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """Causal grouped-query attention. Bias-free, matching the Llama convention."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        kv_dim = self.head_dim * cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """gate/up/down MLP with a SiLU-gated hidden branch. Bias-free."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    """The trunk: token embedding -> `n_layers` blocks -> final norm -> h_t.

    Does not include the LM head; `hg_model.py` owns that so the tied-weight
    LM head and (for the HG variant) the HG heads can share this one output.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self._rope_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._init_weights()

    def _init_weights(self, std: float = 0.02) -> None:
        """GPT-2-style init. Caught by an actual Spark smoke run: with
        default `nn.Linear`/`nn.Embedding` init, the tied-embedding LM head
        produced logits of magnitude ~400 at random init, which is not a
        correctness bug on its own (the smoke run's loss still decreased)
        but is a well-known transformer-training footgun -- unscaled
        residual growth across layers risks instability at longer/larger
        runs. `std=0.02` and the o_proj/down_proj residual-output scaling
        below match GPT-2's own init.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
        # Residual-output projections get an extra 1/sqrt(2*n_layers): each
        # of the 2*n_layers residual additions (attn + mlp per layer) would
        # otherwise compound the output variance across the whole trunk.
        residual_scale = 1.0 / math.sqrt(2 * self.cfg.n_layers)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.o_proj.weight.mul_(residual_scale)
                block.mlp.down_proj.weight.mul_(residual_scale)

    def _cos_sin(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope_cache is None or self._rope_cache[0].shape[0] < seq_len or self._rope_cache[0].device != device:
            self._rope_cache = _rope_cache(max(seq_len, self.cfg.max_seq_len), self.cfg.head_dim, device)
        cos, sin = self._rope_cache
        return cos[:seq_len], sin[:seq_len]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq_len) -> h_t: (batch, seq_len, d_model)."""
        bsz, seq_len = input_ids.shape
        if seq_len > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}")
        x = self.dropout(self.embed(input_ids))
        cos, sin = self._cos_sin(seq_len, input_ids.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.final_norm(x)
