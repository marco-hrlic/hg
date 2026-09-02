"""Dimension-matched vanilla controls -- M0-R1 kickoff section 4.2.

HG's latent S_t is 64-dimensional; a raw vanilla hidden state (d_model, e.g.
640) is not a fair comparison for `quotient_probes.py`'s linear probes on its
own (a bigger space gives a linear probe more to work with regardless of
whether the extra dimensions encode anything task-relevant). This module
projects a raw hidden-state array down to 64 dimensions two ways:

- `fit_pca` -- the kickoff doc's *primary* dimension-matched control: PCA
  captures whatever directions of variance the vanilla trunk actually
  represents, without introducing any task supervision.
- `fit_random_projection` -- the kickoff doc's *optional sensitivity* control:
  a fixed seeded orthogonal projection, to check whether PCA's variance-max
  choice of directions matters or any 64D subspace behaves similarly.

Torch-free (numpy only, matching `quotient_probes.py`'s no-sklearn
convention) and side-effect-free: fitting and transforming are separate steps
so callers must explicitly fit on a train-only split and transform both train
and eval splits with that one fitted projection -- fitting on eval data (or
refitting per split) would leak information the kickoff doc explicitly bans
("Do not fit PCA or any learned projection on the evaluation split").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Projection:
    """A fitted mean-center + linear-map projection to `n_components` dims."""

    mean: np.ndarray  # (d,)
    components: np.ndarray  # (n_components, d) -- rows are the projection directions
    n_components: int

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.mean.shape[0]:
            raise ValueError(f"expected {self.mean.shape[0]} input dims, got {X.shape[1]}")
        return (X - self.mean) @ self.components.T


def fit_pca(X_train: np.ndarray, n_components: int = 64, seed: int = 0) -> Projection:
    """Fit PCA via SVD on `X_train` only. `seed` is accepted for interface
    symmetry with `fit_random_projection` (SVD itself is deterministic) and
    to make sign/ordering ties reproducible if numpy's SVD implementation
    ever changes.
    """
    X = np.asarray(X_train, dtype=float)
    n, d = X.shape
    if n_components > min(n, d):
        raise ValueError(f"n_components={n_components} exceeds min(n_samples={n}, n_features={d})")
    mean = X.mean(axis=0)
    centered = X - mean
    # Full SVD of the centered data matrix: right singular vectors (rows of
    # Vt) are the principal directions, ordered by explained variance.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components]
    return Projection(mean=mean, components=components, n_components=n_components)


def fit_random_projection(X_train: np.ndarray, n_components: int = 64, seed: int = 0) -> Projection:
    """A fixed seeded orthogonal projection to `n_components` dims -- the
    kickoff doc's optional sensitivity control alongside PCA. Only the
    dimensionality (`X_train.shape[1]`) and the mean are taken from the data;
    the directions themselves carry no information from `X_train` beyond
    that, unlike PCA.
    """
    X = np.asarray(X_train, dtype=float)
    n, d = X.shape
    if n_components > d:
        raise ValueError(f"n_components={n_components} exceeds n_features={d}")
    mean = X.mean(axis=0)
    rng = np.random.default_rng(seed)
    random_matrix = rng.normal(size=(d, d))
    q, _ = np.linalg.qr(random_matrix)  # orthonormal columns
    components = q[:, :n_components].T
    return Projection(mean=mean, components=components, n_components=n_components)
