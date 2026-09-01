"""Run an HG model over a set of episodes and extract per-episode predictions.

Needs `torch`. The rest of `eval/` is deliberately torch-free and consumes
this module's plain-`numpy` output, not a live model, so grounding/
accessibility/routing/reuse metrics can be recomputed from a saved
predictions array without a model or GPU at hand (spec section 4's
recomputability requirement, extended to predictions the same way it
applies to episodes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hg_native.data.schema import Episode
from hg_native.model.config import DEFAULT_DECISION_LABELS, DEFAULT_ROUTES
from hg_native.model.hg_model import HGNativeModel
from hg_native.training.batch import Tokenizer, collate, serialize_episode


@dataclass(slots=True)
class Predictions:
    episode_id: list[str]
    info_deficit_prob: np.ndarray
    accessibility_prob: np.ndarray
    decision_pred: np.ndarray  # index into DEFAULT_DECISION_LABELS
    route_pred: np.ndarray  # index into DEFAULT_ROUTES

    def decision_label(self, i: int) -> str:
        return DEFAULT_DECISION_LABELS[self.decision_pred[i]]

    def route_label(self, i: int) -> str:
        return DEFAULT_ROUTES[self.route_pred[i]]


def predict(
    model: HGNativeModel,
    tokenizer: Tokenizer,
    episodes: list[Episode],
    max_seq_len: int,
    device: str = "cpu",
    batch_size: int = 64,
) -> Predictions:
    """Batched no-grad forward pass. Requires `model.config.variant == "hg"`
    (a baseline model has none of these heads)."""
    if model.config.variant != "hg":
        raise ValueError("predict() requires an HG-variant model (baseline has no HG heads)")

    model.eval()
    episode_ids: list[str] = []
    info_deficit_chunks, access_chunks, decision_chunks, route_chunks = [], [], [], []

    for start in range(0, len(episodes), batch_size):
        chunk = episodes[start : start + batch_size]
        raw = collate([serialize_episode(ep, tokenizer, max_seq_len) for ep in chunk])
        ids = torch.tensor(raw["input_ids"], device=device)
        positions = torch.tensor(raw["decision_position"], device=device)
        with torch.no_grad():
            out = model(ids, decision_positions=positions)
        episode_ids.extend(raw["episode_id"])
        info_deficit_chunks.append(torch.sigmoid(out["info_deficit_logit"]).cpu().numpy())
        access_chunks.append(torch.sigmoid(out["accessibility_logit"]).cpu().numpy())
        decision_chunks.append(out["decision_logits"].argmax(dim=-1).cpu().numpy())
        route_chunks.append(out["route_logits"].argmax(dim=-1).cpu().numpy())

    return Predictions(
        episode_id=episode_ids,
        info_deficit_prob=np.concatenate(info_deficit_chunks),
        accessibility_prob=np.concatenate(access_chunks),
        decision_pred=np.concatenate(decision_chunks),
        route_pred=np.concatenate(route_chunks),
    )
