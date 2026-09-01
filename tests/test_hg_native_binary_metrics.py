"""AUROC/AUPRC/Brier/ECE against hand-computable cases -- torch-free."""

from __future__ import annotations

import math

import pytest

from hg_native.eval.binary_metrics import auprc, auroc, brier_score, ece, summary


# --- auroc ---------------------------------------------------


def test_auroc_perfect_separation():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_auroc_perfectly_anticorrelated():
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_auroc_all_ties_is_a_coin_flip():
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == pytest.approx(0.5)


def test_auroc_nan_when_one_class_missing():
    assert math.isnan(auroc([0.1, 0.2, 0.3], [1, 1, 1]))
    assert math.isnan(auroc([0.1, 0.2, 0.3], [0, 0, 0]))


def test_auroc_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        auroc([0.1, 0.2], [1])


def test_auroc_known_partial_case():
    # 3 positives, 2 negatives; count concordant pairs by hand.
    # pos = [0.9, 0.4, 0.3], neg = [0.5, 0.2]
    # pairs: (0.9,0.5)win (0.9,0.2)win (0.4,0.5)lose (0.4,0.2)win (0.3,0.5)lose (0.3,0.2)win
    # 4 wins / 6 pairs = 0.6667
    scores = [0.9, 0.4, 0.3, 0.5, 0.2]
    labels = [1, 1, 1, 0, 0]
    assert auroc(scores, labels) == pytest.approx(4 / 6)


# --- auprc ---------------------------------------------------


def test_auprc_perfect_separation():
    assert auprc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_auprc_nan_when_no_positives():
    assert math.isnan(auprc([0.1, 0.2, 0.3], [0, 0, 0]))


def test_auprc_worst_case_ranking():
    # All negatives ranked above all positives -- the worst possible ranking.
    # By hand: ranks (neg, neg, pos, pos) with cumulative precision/recall
    # (0, 0), (0, 0), (1/3, 1/2), (2/4, 1). AP = 0.5*(1/3) + 0.5*(1/2) = 5/12.
    scores = [0.9, 0.8, 0.1, 0.05]
    labels = [0, 0, 1, 1]
    assert auprc(scores, labels) == pytest.approx(5 / 12)


# --- brier_score ---------------------------------------------------


def test_brier_score_exact():
    # (0.9-1)^2 + (0.1-0)^2 + (0.5-1)^2 + (0.5-0)^2 = 0.01+0.01+0.25+0.25 = 0.52 / 4
    assert brier_score([0.9, 0.1, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.52 / 4)


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


# --- ece ---------------------------------------------------


def test_ece_perfectly_calibrated_is_zero():
    # A single bin whose mean confidence (0.5) exactly matches its mean
    # empirical positive rate (0.5, from 2 of 4 labels positive) -- ECE
    # measures calibration (does confidence match empirical frequency), not
    # discriminative accuracy, so a *correct, confident* classifier is not
    # automatically "calibrated" in this sense.
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    assert ece(scores, labels, n_bins=1) == pytest.approx(0.0)


def test_ece_overconfident_wrong_predictions():
    # Predicts 0.95 for everything but half are wrong -> |0.5 - 0.95| weighted.
    scores = [0.95, 0.95, 0.95, 0.95]
    labels = [1, 1, 0, 0]
    assert ece(scores, labels, n_bins=10) == pytest.approx(0.45)


def test_ece_rejects_scores_outside_unit_interval():
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        ece([1.5, 0.2], [1, 0])


# --- summary ---------------------------------------------------


def test_summary_has_every_field():
    result = summary([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert set(result) == {"auroc", "auprc", "brier", "ece", "n", "n_positive"}
    assert result["n"] == 4
    assert result["n_positive"] == 2
    assert result["auroc"] == pytest.approx(1.0)
