"""Dimension-matched vanilla controls (PCA-64 / random projection) --
torch-free (M0-R1 kickoff section 4.2)."""

from __future__ import annotations

import numpy as np
import pytest

from hg_native.eval.projections import fit_pca, fit_random_projection


def test_fit_pca_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 20))
    proj = fit_pca(X, n_components=5, seed=0)
    out = proj.transform(X)
    assert out.shape == (200, 5)


def test_fit_pca_recovers_a_low_rank_signal():
    rng = np.random.default_rng(0)
    n = 300
    latent = rng.normal(size=(n, 2))
    directions = rng.normal(size=(2, 10))
    X = latent @ directions + rng.normal(scale=0.01, size=(n, 10))
    proj = fit_pca(X, n_components=2, seed=0)
    out = proj.transform(X)
    # The top-2 PCA components should span essentially the same subspace as
    # the true 2D latent: reconstructing X from the PCA scores should match
    # the (near noise-free) original almost exactly.
    reconstruction = out @ proj.components + proj.mean
    assert np.allclose(reconstruction, X, atol=0.1)


def test_fit_pca_rejects_too_many_components():
    X = np.zeros((10, 5))
    with pytest.raises(ValueError, match="n_components"):
        fit_pca(X, n_components=8)


def test_fit_pca_transform_rejects_wrong_dim():
    X = np.random.default_rng(0).normal(size=(50, 8))
    proj = fit_pca(X, n_components=3, seed=0)
    with pytest.raises(ValueError):
        proj.transform(np.zeros((5, 9)))


def test_fit_pca_deterministic():
    X = np.random.default_rng(0).normal(size=(50, 8))
    a = fit_pca(X, n_components=3, seed=0)
    b = fit_pca(X, n_components=3, seed=0)
    assert np.allclose(a.components, b.components)


def test_fit_pca_does_not_refit_on_eval_split():
    # Split isolation (kickoff section 19): fitting on train and
    # transforming eval data must not change the projection itself.
    rng = np.random.default_rng(0)
    train = rng.normal(size=(100, 10))
    eval_data = rng.normal(size=(30, 10)) + 5.0  # a shifted eval split
    proj = fit_pca(train, n_components=4, seed=0)
    mean_before = proj.mean.copy()
    proj.transform(eval_data)
    assert np.array_equal(proj.mean, mean_before)


def test_fit_random_projection_shape_and_orthonormality():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 16))
    proj = fit_random_projection(X, n_components=6, seed=0)
    out = proj.transform(X)
    assert out.shape == (100, 6)
    gram = proj.components @ proj.components.T
    assert np.allclose(gram, np.eye(6), atol=1e-8)


def test_fit_random_projection_deterministic_given_seed():
    X = np.random.default_rng(0).normal(size=(50, 10))
    a = fit_random_projection(X, n_components=4, seed=7)
    b = fit_random_projection(X, n_components=4, seed=7)
    assert np.array_equal(a.components, b.components)


def test_fit_random_projection_differs_across_seeds():
    X = np.random.default_rng(0).normal(size=(50, 10))
    a = fit_random_projection(X, n_components=4, seed=1)
    b = fit_random_projection(X, n_components=4, seed=2)
    assert not np.allclose(a.components, b.components)
