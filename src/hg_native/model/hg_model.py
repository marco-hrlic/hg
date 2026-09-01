"""Assembles the trunk (+ LM head) and, for the HG variant, the latent + heads.

One class serves both `variant="baseline"` and `variant="hg"` (spec section
2.1: same trunk, tokenizer, schedule — the HG variant is the baseline plus
additive heads, not a different trunk), so a baseline/HG pair built from
otherwise-identical `HGNativeConfig`s is guaranteed to share every trunk
hyperparameter by construction rather than by convention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import HGNativeConfig
from .heads import HGHeads
from .hg_latent import HGLatent
from .transformer import Transformer


class HGNativeModel(nn.Module):
    def __init__(self, config: HGNativeConfig):
        super().__init__()
        self.config = config
        self.trunk = Transformer(config.transformer)
        # Untied LM head only if the config asks for it; tied is the spec's
        # recommended default (section 3.1) and the common case is expected
        # to stay tied, so no separate weight is allocated then.
        self.lm_head = (
            None if config.transformer.tie_embeddings else nn.Linear(config.transformer.d_model, config.transformer.vocab_size, bias=False)
        )

        if config.variant == "hg":
            assert config.hg_latent is not None  # enforced by HGNativeConfig.__post_init__
            input_dim = config.hg_latent.resolved_input_dim(config.transformer)
            self.hg_latent: HGLatent | None = HGLatent(config.hg_latent, input_dim)
            self.heads: HGHeads | None = HGHeads(config.heads, config.hg_latent.latent_dim)
        else:
            self.hg_latent = None
            self.heads = None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def lm_logits(self, h: torch.Tensor) -> torch.Tensor:
        weight = self.trunk.embed.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(h, weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        decision_positions: torch.Tensor | None = None,
        action_id: torch.Tensor | None = None,
        r_info_hat: torch.Tensor | None = None,
        r_access_hat: torch.Tensor | None = None,
        c_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """input_ids: (batch, seq_len).

        `decision_positions`, if given, is a (batch,) tensor of per-example
        sequence indices at which to read off `h_t` for the HG heads (spec
        section 3.2's "at selected decision positions"); omitted, the HG
        latent and heads run at every position instead, which is what
        shape/gradient tests exercise since it needs no extra input.
        `action_id`/`r_info_hat`/`r_access_hat`/`c_t` are ignored for
        `variant="baseline"`.
        """
        h = self.trunk(input_ids)  # (batch, seq_len, d_model)
        out: dict[str, torch.Tensor] = {"logits": self.lm_logits(h), "h": h}

        if self.config.variant != "hg":
            return out

        if decision_positions is None:
            s_t = self.hg_latent(h)  # (batch, seq_len, latent_dim)
        else:
            gathered = h[torch.arange(h.shape[0], device=h.device), decision_positions]  # (batch, d_model)
            s_t = self.hg_latent(gathered)  # (batch, latent_dim)
        out["s_t"] = s_t

        out["decision_logits"] = self.heads.decision(s_t)
        out["distortion_hat"] = self.heads.distortion(s_t)
        out["info_deficit_logit"] = self.heads.info_deficit(s_t)
        out["accessibility_logit"] = self.heads.accessibility(s_t)

        if action_id is not None:
            out["s_next_hat"] = self.heads.transition(s_t, action_id)

        if r_info_hat is None:
            r_info_hat = torch.sigmoid(out["info_deficit_logit"]).detach()
        if r_access_hat is None:
            r_access_hat = torch.sigmoid(out["accessibility_logit"]).detach()
        if c_t is None:
            c_t = torch.zeros(*s_t.shape[:-1], self.config.heads.cost_dim, device=s_t.device, dtype=s_t.dtype)
        out["route_logits"] = self.heads.router(s_t, r_info_hat, r_access_hat, c_t)

        return out


def build_model(config: HGNativeConfig) -> HGNativeModel:
    torch.manual_seed(config.seed)
    return HGNativeModel(config)
