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
