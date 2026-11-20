"""Small PyTorch helpers for lab 1. The training loop itself stays in the notebook."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from . import LABELS

LABEL_TO_ID = {lab: i for i, lab in enumerate(LABELS)}


def encode_labels(labels: Sequence[str]) -> np.ndarray:
    return np.array([LABEL_TO_ID[lab] for lab in labels], dtype=np.int64)


def batches(X, y: np.ndarray, batch_size: int = 64, shuffle: bool = True,
            seed: int = 0) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (features, labels) mini-batches as tensors.

    X is a scipy sparse matrix (one row per review). Only one batch at a time
    is turned into a dense tensor, so memory stays small.
    """
    order = np.random.default_rng(seed).permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield (torch.from_numpy(X[idx].toarray().astype(np.float32)),
               torch.from_numpy(y[idx]))


@torch.no_grad()
def evaluate(model: torch.nn.Module, X, y: np.ndarray, loss_fn) -> tuple[float, float]:
    """Average loss and accuracy over a whole dataset."""
    model.eval()
    total_loss, correct = 0.0, 0
    for xb, yb in batches(X, y, batch_size=512, shuffle=False):
        logits = model(xb)
        total_loss += loss_fn(logits, yb).item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
    model.train()
    return total_loss / len(y), correct / len(y)


@torch.no_grad()
def predict_proba(model: torch.nn.Module, X) -> np.ndarray:
    model.eval()
    out = [torch.softmax(model(xb), dim=1).numpy()
           for xb, _ in batches(X, np.zeros(X.shape[0], dtype=np.int64), batch_size=512, shuffle=False)]
    model.train()
    return np.concatenate(out)


def plot_history(history: list[dict], title: str = "") -> None:
    """history: one dict per epoch with train_loss, dev_loss, train_acc, dev_acc."""
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2))
    a1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    a1.plot(epochs, [h["dev_loss"] for h in history], "o-", label="dev")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss (lower is better)"); a1.legend()
    a2.plot(epochs, [h["train_acc"] for h in history], "o-", label="train")
    a2.plot(epochs, [h["dev_acc"] for h in history], "o-", label="dev")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy"); a2.legend()
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()
