"""The localization race."""

from __future__ import annotations

import numpy as np
import pytest

from hg.routing.strategies import (
    disagreement_score,
    failure_memory_score,
    run_race,
    uncertainty_score,
)
from tests.conftest import make_matrix


def three_label_matrix(n=600, n_core=40, seed=0):
    """A 3-label task with a clustered clean core.

    Three labels, not two: with two labels a consensus failure has zero
    disagreement by construction and the race is unfair to disagreement
    sampling for arithmetic reasons.
    """
    rng = np.random.default_rng(seed)
    n_members = 5
    e = rng.random((n, n_members)) < 0.15
    preds = np.empty((n, n_members), dtype=object)
    for i in range(n):
        for j in range(n_members):
            preds[i, j] = rng.choice(["b", "c"]) if e[i, j] else "a"
    # A core every member fails, all on the same wrong label -> no disagreement.
    e[:n_core] = True
    preds[:n_core] = "b"
    conf = np.where(e, 0.55, 0.9) + rng.normal(0, 0.01, (n, n_members))
    emb = rng.normal(size=(n, 16))
    emb[:n_core] += 6.0
    return make_matrix(e, predictions=preds, confidences=conf), emb


def test_disagreement_score_is_zero_when_members_agree():
    preds = np.array([["a", "a", "a"], ["a", "b", "c"]], dtype=object)
    m = make_matrix([[0, 0, 0], [1, 1, 1]], predictions=preds)
    scores = disagreement_score(m)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] > scores[0]


def test_uncertainty_score_falls_back_to_uniform_without_confidences():
    """A uniform fallback degrades the strategy to random, which is honest.

    Substituting error counts would leak the labels the strategy is not
    supposed to see.
    """
    m = make_matrix([[1, 1], [0, 0]])
    assert np.allclose(uncertainty_score(m), 0.0)


def test_failure_memory_is_uniform_without_embeddings():
    m, _ = three_label_matrix()
    assert np.allclose(failure_memory_score(m, [0, 1, 2]), 0.0)


def test_failure_memory_finds_a_clustered_core():
    m, emb = three_label_matrix()
    race = run_race(m, embeddings=emb, seed=0)
    assert race.results["failure_memory"].auc > race.results["random"].auc


def test_oracle_bounds_every_strategy():
    m, emb = three_label_matrix()
    race = run_race(m, embeddings=emb, seed=0)
    oracle = race.results["oracle"].auc
    for name, res in race.results.items():
        assert res.auc <= oracle + 1e-9, name


def test_history_items_are_excluded_from_the_candidate_pool():
    """No strategy may be credited for finding what it was told."""
    m, emb = three_label_matrix(n=400)
    race = run_race(m, embeddings=emb, history_size=50, seed=0)
    assert race.n_items == 350


def test_binary_task_race_is_reported_as_not_applicable():
    """With two labels 'all wrong' and 'all agreed' are the same event, so the
    disagreement strategy ranks its own targets last. P5/P6 are untestable."""
    rng = np.random.default_rng(1)
    e = rng.random((400, 5)) < 0.2
    e[:30] = True
    m = make_matrix(e)  # binary a/b convention
    race = run_race(m, seed=0)
    assert race.n_labels == 2
    assert race.degenerate_for_disagreement
    interp = race.interpret()
    assert interp["verdict"] == "not_applicable"
    assert "binary" in interp["note"]


def test_three_label_race_produces_a_real_verdict():
    m, emb = three_label_matrix()
    race = run_race(m, embeddings=emb, seed=0)
    assert not race.degenerate_for_disagreement
    assert race.interpret()["verdict"] in ("targeting_wins", "disagreement_wins", "tie")


def test_race_is_not_applicable_without_targets():
    m = make_matrix(np.zeros((200, 5), dtype=bool))
    race = run_race(m, seed=0)
    assert race.interpret()["verdict"] == "not_applicable"


def test_localization_surplus_is_targeted_minus_disagreement():
    m, emb = three_label_matrix()
    race = run_race(m, embeddings=emb, seed=0)
    budget = race.results["disagreement"].budgets[2]
    expected = race.results["failure_memory"].value_at(budget) - race.results[
        "disagreement"
    ].value_at(budget)
    assert race.surplus(budget) == pytest.approx(expected)
