"""Extract S_t latent vectors from a trained HG model, for `quotient_probes.py`.

Needs `torch`. Kept separate from `eval.predict` (which returns head
*predictions*, not the latent itself) so `quotient_probes.py` can stay
torch-free and operate on a saved `latents.npy` array.
"""

from __future__ import annotations

import numpy as np
import torch

from hg_native.data.schema import Episode
from hg_native.model.hg_model import HGNativeModel
from hg_native.training.batch import Tokenizer, collate, serialize_episode


def extract_latents(
    model: HGNativeModel,
    tokenizer: Tokenizer,
    episodes: list[Episode],
    max_seq_len: int,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Batched no-grad forward pass -> `(len(episodes), latent_dim)` array of S_t.

    Requires `model.config.variant == "hg"` (a baseline model has no latent).
    """
    if model.config.variant != "hg":
        raise ValueError("extract_latents() requires an HG-variant model (baseline has no HG latent)")

    model.eval()
    chunks = []
    for start in range(0, len(episodes), batch_size):
        chunk = episodes[start : start + batch_size]
        raw = collate([serialize_episode(ep, tokenizer, max_seq_len) for ep in chunk])
        ids = torch.tensor(raw["input_ids"], device=device)
        positions = torch.tensor(raw["decision_position"], device=device)
        with torch.no_grad():
            out = model(ids, decision_positions=positions)
        chunks.append(out["s_t"].cpu().numpy())
    return np.concatenate(chunks, axis=0)


def extract_hidden_states(
    model: HGNativeModel,
    tokenizer: Tokenizer,
    episodes: list[Episode],
    max_seq_len: int,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """Batched no-grad forward pass -> `(len(episodes), d_model)` array of the
    *raw trunk* hidden state `h_t` at each episode's decision position.

    Unlike `extract_latents`, this works for **either** variant: `out["h"]`
    (the trunk output before the HG latent projection) is always present in
    `HGNativeModel.forward`'s return dict (see `model/hg_model.py`). This is
    the M0-R1 kickoff doc's section 4.1 "raw vanilla hidden-state probe" --
    the natural non-HG competitor to `extract_latents`'s 64-dim `S_t`, read at
    the exact same `decision_position` (from `serialize_episode`, itself
    variant-agnostic) so the two representations are compared at the same
    token, just before vs. after the HG-specific projection.
    """
    model.eval()
    chunks = []
    for start in range(0, len(episodes), batch_size):
        chunk = episodes[start : start + batch_size]
        raw = collate([serialize_episode(ep, tokenizer, max_seq_len) for ep in chunk])
        ids = torch.tensor(raw["input_ids"], device=device)
        positions = torch.tensor(raw["decision_position"], device=device)
        with torch.no_grad():
            out = model(ids)
        h = out["h"]  # (batch, seq_len, d_model)
        gathered = h[torch.arange(h.shape[0], device=h.device), positions]  # (batch, d_model)
        chunks.append(gathered.cpu().numpy())
    return np.concatenate(chunks, axis=0)
