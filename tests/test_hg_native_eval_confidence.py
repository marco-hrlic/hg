"""Vanilla confidence features + info/access logistic proxies --
M0-R1 kickoff section 8.

`fit_logistic_proxy`/`LogisticProxy` are torch-free (plain numpy, mirroring
`quotient_probes.probe_accuracy`'s fitter); `confidence_features` needs a
real model forward pass and so needs `torch` -- skipped here, runs on Spark.
"""

from __future__ import annotations

import numpy as np
import pytest

from hg_native.eval.confidence import FEATURE_NAMES, fit_logistic_proxy


def test_fit_logistic_proxy_recovers_a_clean_signal():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n).astype(float)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    X[:, 0] = y * 6 - 3 + rng.normal(scale=0.1, size=n)  # one clean signal feature
    proxy = fit_logistic_proxy(X, y, seed=0)
    pred = (proxy.predict_proba(X) > 0.5).astype(float)
    assert (pred == y).mean() > 0.9


def test_fit_logistic_proxy_near_chance_on_pure_noise():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.integers(0, 2, size=n).astype(float)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    proxy = fit_logistic_proxy(X, y, seed=0)
    pred = (proxy.predict_proba(X) > 0.5).astype(float)
    assert 0.3 < (pred == y).mean() < 0.7


def test_fit_logistic_proxy_rejects_length_mismatch():
    X = np.zeros((10, 4))
    y = np.zeros(9)
    with pytest.raises(ValueError):
        fit_logistic_proxy(X, y)


def test_fit_logistic_proxy_deterministic_given_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.integers(0, 2, size=50).astype(float)
    a = fit_logistic_proxy(X, y, seed=0)
    b = fit_logistic_proxy(X, y, seed=0)
    assert np.array_equal(a.weights, b.weights)
    assert a.bias == b.bias


def test_logistic_proxy_predict_proba_in_unit_interval():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = rng.integers(0, 2, size=50).astype(float)
    proxy = fit_logistic_proxy(X, y, seed=0)
    proba = proxy.predict_proba(rng.normal(size=(20, 4)))
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_confidence_features_never_take_labels_as_input():
    # kickoff section 8.1's explicit ban: `confidence_features`'s signature
    # has no episode-label/ground-truth parameter to leak through at all.
    import inspect

    from hg_native.eval.confidence import confidence_features

    params = set(inspect.signature(confidence_features).parameters)
    assert params == {"model", "tokenizer", "episodes", "max_seq_len", "device", "batch_size"}
