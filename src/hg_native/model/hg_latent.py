"""The decision latent S_t = f_HG(h_t) (spec section 3.2).

No information-bottleneck regularization by default — the spec is explicit
that compression should be an ablation, not the first run's default, so
there is nothing here beyond the MLP + LayerNorm map itself.

Structure must match `config.py`'s `_mlp_stack_param_count`/`_hg_latent_param_count`
exactly (both bias-full `nn.Linear`s, `layers` total hidden-width blocks
before the output projection, plus an affine `LayerNorm`) — the two are
cross-checked in `tests/test_hg_native_model_shapes.py`.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import HGLatentConfig


class HGLatent(nn.Module):
    def __init__(self, cfg: HGLatentConfig, input_dim: int):
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = [nn.Linear(input_dim, cfg.hidden_dim), nn.SiLU()]
        for _ in range(cfg.layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(cfg.hidden_dim, cfg.latent_dim))
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(cfg.latent_dim)

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        """h_t: (..., input_dim) -> S_t: (..., latent_dim)."""
        return self.norm(self.mlp(h_t))
