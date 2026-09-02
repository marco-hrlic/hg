"""Vanilla confidence features + info-deficit/accessibility proxies --
M0-R1 kickoff section 8.

HG's `InfoDeficitHead`/`AccessibilityHead` read a supervised signal off the
64-dim HG latent. A baseline model has neither head (and never saw action
labels: `serialize_episode`'s decision position is the prompt's own EOS
token, so the baseline's next-token distribution there was never trained
towards an action either) -- so there is no "action probability" to read
off a baseline model in the sense spec section 3.3 defines for HG.

The feature set below substitutes the model-agnostic quantities every
autoregressive LM actually produces at that same position: the entropy,
max probability, and top1-top2 margin of its next-token distribution, plus
the norm of its raw hidden state there. This generalizes the kickoff doc's
"max action probability / margin / entropy" list (written with an
action-having model in mind) to something computable for *either* variant,
which is what a fair comparison needs -- HG's own info/access heads are
evaluated on exactly the same episodes via `eval.predict`, so the actual
head-vs-proxy comparison lives in `grounding.py`/`scaffolding.py`, not here.

Ground-truth labels are never inputs to these features (kickoff section
8.1's explicit ban on using target labels as features) -- everything here
comes from `out["logits"]`/`out["h"]` at the decision position, nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from hg_native.data.schema import Episode
from hg_native.training.batch import Tokenizer, collate, serialize_episode

if TYPE_CHECKING:  # pragma: no cover -- only for static type checkers; avoids a
    # hard torch import at module load time (fit_logistic_proxy/LogisticProxy
    # below are torch-free and must stay importable without torch installed).
    from hg_native.model.hg_model import HGNativeModel

FEATURE_NAMES = ("max_token_prob", "top1_top2_margin", "token_entropy", "hidden_norm")


def confidence_features(
    model: HGNativeModel,
    tokenizer: Tokenizer,
    episodes: list[Episode],
    max_seq_len: int,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """`(len(episodes), 4)` array of `FEATURE_NAMES`, read at each episode's
    decision position. Works for either variant -- only `out["logits"]`/
    `out["h"]` are used, both always present (`model/hg_model.py`)."""
    import torch  # lazy: keeps this module importable torch-free (see the
    # `TYPE_CHECKING` guard above) for `fit_logistic_proxy`/`LogisticProxy`.

    model.eval()
    chunks = []
    for start in range(0, len(episodes), batch_size):
        chunk = episodes[start : start + batch_size]
        raw = collate([serialize_episode(ep, tokenizer, max_seq_len) for ep in chunk])
        ids = torch.tensor(raw["input_ids"], device=device)
        positions = torch.tensor(raw["decision_position"], device=device)
        with torch.no_grad():
            out = model(ids)
        arange = torch.arange(ids.shape[0], device=ids.device)
        logits = out["logits"][arange, positions]  # (batch, vocab)
        hidden = out["h"][arange, positions]  # (batch, d_model)

        probs = torch.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        max_prob = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=-1)
        hidden_norm = hidden.norm(dim=-1)

        features = torch.stack([max_prob, margin, entropy, hidden_norm], dim=-1)
        chunks.append(features.cpu().numpy())
    return np.concatenate(chunks, axis=0)


@dataclass(slots=True)
class LogisticProxy:
    """`P(y=1|x)` from a linear readout of confidence features, fit on a
    train-only split (never the eval split -- caller's responsibility, same
    convention as `eval.projections`)."""

    mean: np.ndarray
    std: np.ndarray
    weights: np.ndarray
    bias: float

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        z = ((X - self.mean) / self.std) @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic_proxy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iters: int = 500,
    lr: float = 0.5,
    l2: float = 1e-3,
    seed: int = 0,
) -> LogisticProxy:
    """Kickoff section 8.2/8.3's `g_info(x)`/`g_access(x)`: a simple logistic
    model over confidence features, reusing the same hand-rolled fitter as
    `quotient_probes.py`'s linear probes (`fit_logistic_regression`)."""
    from .quotient_probes import fit_logistic_regression

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if len(X) != len(y):
        raise ValueError(f"{len(X)} feature rows but {len(y)} labels")
    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-8
    X_norm = (X - mean) / std
    w, b = fit_logistic_regression(X_norm, y, n_iters=n_iters, lr=lr, l2=l2, seed=seed)
    return LogisticProxy(mean=mean, std=std, weights=w, bias=b)
