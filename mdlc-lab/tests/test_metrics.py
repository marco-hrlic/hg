import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score, f1_score

from mdlc import LABELS, metrics

rng = np.random.default_rng(0)
Y = rng.choice(LABELS, 300).tolist()
P = [y if rng.random() < 0.6 else rng.choice(LABELS) for y in Y]


def test_confusion_matrix_rows_are_truth():
    cm = metrics.confusion_matrix(["negative", "negative"], ["positive", "negative"])
    assert cm[0, 2] == 1 and cm[0, 0] == 1 and cm.sum() == 2


def test_macro_f1_matches_sklearn():
    assert metrics.macro_f1(Y, P) == pytest.approx(f1_score(Y, P, average="macro", labels=list(LABELS)))


def test_macro_f1_handles_missing_class():
    y = ["positive"] * 5 + ["negative"] * 5
    p = ["positive"] * 10
    assert metrics.macro_f1(y, p) == pytest.approx(f1_score(y, p, average="macro", labels=list(LABELS),
                                                            zero_division=0))


def test_cohen_kappa_matches_sklearn():
    assert metrics.cohen_kappa(Y, P) == pytest.approx(cohen_kappa_score(Y, P))


def test_fleiss_kappa_wikipedia_example():
    # 10 items, 14 raters, 5 categories: the worked example on Wikipedia, kappa ~= 0.210
    table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
             [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
    cats = list("abcde")
    rows = [{"item_id": i, "label": cats[c]} for i, counts in enumerate(table)
            for c, k in enumerate(counts) for _ in range(k)]
    assert metrics.fleiss_kappa(pd.DataFrame(rows), labels=cats) == pytest.approx(0.210, abs=1e-3)


def test_slice_report_counts_items_once_per_slice():
    df = pd.DataFrame({"y": ["positive", "negative", "neutral"], "p": ["positive", "positive", "neutral"],
                       "slices": [["short", "negation"], ["negation"], []]})
    rep = metrics.slice_report(df, "y", "p")
    assert rep.loc["ALL", "n"] == 3 and rep.loc["negation", "n"] == 2 and rep.loc["short", "n"] == 1
    assert rep.loc["negation", "accuracy"] == 0.5
