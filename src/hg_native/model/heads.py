"""The HG heads (spec section 3.3). All consume `S_t` unless noted.

Each head's shape must match `config.py`'s `_heads_param_count` term for it
exactly — cross-checked in `tests/test_hg_native_model_shapes.py`. Sizes not
pinned by the spec (action vocabulary, router hidden width, cost-vector
width) come from `HeadsConfig`; see its docstring for why they're flagged
as implementation choices rather than spec-given values.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import HeadsConfig


class DecisionHead(nn.Module):
    """p_theta(J_t | S_t): the final binary A/B task decision."""

    def __init__(self, cfg: HeadsConfig, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, cfg.n_decision_labels)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.proj(s_t)  # logits, (..., n_decision_labels)


class DistortionHead(nn.Module):
    """d_hat_t(a) = g_d(S_t, a): predicted loss of every available action."""

    def __init__(self, cfg: HeadsConfig, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, cfg.n_actions)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.proj(s_t)  # raw predicted distortion, (..., n_actions)


class InfoDeficitHead(nn.Module):
    """p_theta(G_t^info = 1 | S_t): is external information genuinely required."""

    def __init__(self, cfg: HeadsConfig, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 1)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.proj(s_t).squeeze(-1)  # logit, (...)


class AccessibilityHead(nn.Module):
    """p(G_t^access = 1 | S_t): determined-but-likely-to-fail-without-scaffold."""

    def __init__(self, cfg: HeadsConfig, latent_dim: int):
        super().__init__()
        self.proj = nn.Linear(latent_dim, 1)

    def forward(self, s_t: torch.Tensor) -> torch.Tensor:
        return self.proj(s_t).squeeze(-1)  # logit, (...)


class TransitionHead(nn.Module):
    """S_hat_{t+1} = g_T(S_t, A_t), trained against a stop-gradient target (spec section 3.3)."""

    def __init__(self, cfg: HeadsConfig, latent_dim: int):
        super().__init__()
        self.action_embed = nn.Embedding(cfg.n_actions, cfg.action_embed_dim)
        self.proj = nn.Linear(latent_dim + cfg.action_embed_dim, latent_dim)

    def forward(self, s_t: torch.Tensor, action_id: torch.Tensor) -> torch.Tensor:
        """s_t: (..., latent_dim); action_id: (...) integer action indices."""
        a = self.action_embed(action_id)
        return self.proj(torch.cat([s_t, a], dim=-1))


class RouterHead(nn.Module):
    """Route over {ACT, THINK, ASK} given [S_t, R_info_hat, R_access_hat, c_t]."""

    def __init__(self, cfg: HeadsConfig, latent_dim: int):
        super().__init__()
        in_dim = latent_dim + 2 + cfg.cost_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.router_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.router_hidden_dim, cfg.n_routes),
        )

    def forward(
        self,
        s_t: torch.Tensor,
        r_info_hat: torch.Tensor,
        r_access_hat: torch.Tensor,
        c_t: torch.Tensor,
    ) -> torch.Tensor:
        """r_info_hat/r_access_hat: (...); c_t: (..., cost_dim) -> route logits (..., n_routes)."""
        x = torch.cat([s_t, r_info_hat.unsqueeze(-1), r_access_hat.unsqueeze(-1), c_t], dim=-1)
        return self.net(x)


class HGHeads(nn.Module):
    """Bundles all six heads, so `HGNativeModel` builds/holds one submodule."""

    def __init__(self, cfg: HeadsConfig, latent_dim: int):
        super().__init__()
        self.decision = DecisionHead(cfg, latent_dim)
        self.distortion = DistortionHead(cfg, latent_dim)
        self.info_deficit = InfoDeficitHead(cfg, latent_dim)
        self.accessibility = AccessibilityHead(cfg, latent_dim)
        self.transition = TransitionHead(cfg, latent_dim)
        self.router = RouterHead(cfg, latent_dim)
