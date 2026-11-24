"""Harness tests. Those needing built data (`make data`) are skipped when it's missing."""

import re
import shutil

import numpy as np
import pandas as pd
import pytest

from mdlc import data, lab

HAVE_DATA = (data.DATA_DIR / "raw" / "answer_key.csv").exists() and (data.DATA_DIR / "heldout" / "heldout.csv").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="run `make data` first")
SECTIONS = ["## Why this matters", "## Verify", "## Go deeper"]


def test_every_card_has_the_standard_sections_and_a_matching_check():
    cards = sorted(lab.STEPS_DIR.glob("lab*/*.md"))
    assert len(cards) == 17
    for card in cards:
        body = card.read_text()
        step = f"{card.parent.name[3:]}.{int(card.name[:2])}"
        if step == "1.0":
            continue
        for s in SECTIONS:
            assert s in body, f"{card.name} lacks {s}"
        if f"lab check {step}" in body:
            assert step in lab.CHECKS, f"{card.name} mentions a check that doesn't exist"


def test_cards_only_reference_files_inside_the_workspace():
    for card in lab.STEPS_DIR.glob("lab*/*.md"):
        assert not re.search(r"\.\./(golden_labels|data|reference|solutions)", card.read_text()), card.name


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, "WORKSPACE", tmp_path)
    (tmp_path / "outputs").mkdir()
    return tmp_path


def test_read_predictions_validates(workspace):
    p = workspace / "p.csv"
    pd.DataFrame({"id": ["a", "b"], "label": ["positive", "happy"]}).to_csv(p, index=False)
    with pytest.raises(lab.CheckFailed, match="labels must be one of"):
        lab.read_predictions(p)
    pd.DataFrame({"review_id": ["a", "a"], "label": ["positive", "neutral"]}).to_csv(p, index=False)
    with pytest.raises(lab.CheckFailed, match="more than once"):
        lab.read_predictions(p)


def test_mathlab_check_accepts_reference_and_rejects_unstable_softmax(workspace):
    shutil.copy(lab.REFERENCE_DIR / "mathlab.py", workspace / "mathlab.py")
    assert lab.check_2_4()
    src = (workspace / "mathlab.py").read_text().replace("Z = Z - Z.max(axis=1, keepdims=True)", "pass")
    (workspace / "mathlab.py").write_text(src)
    with np.errstate(all="ignore"):
        assert not lab.check_2_4()


@needs_data
def test_heldout_is_rationed_and_logged_before_printing(workspace, monkeypatch, capsys):
    held = pd.read_csv(data.DATA_DIR / "heldout" / "heldout.csv")
    f = workspace / "outputs" / "h.csv"
    pd.DataFrame({"id": held.review_id, "label": "positive"}).to_csv(f, index=False)
    for i in range(lab.HELDOUT_ALLOWANCE):
        lab.main(["submit", "heldout", str(f), "--name", f"m{i}"])
    assert len(lab.load_log()) == lab.HELDOUT_ALLOWANCE
    with pytest.raises(SystemExit):
        lab.main(["submit", "heldout", str(f), "--name", "one-more"])
    assert len(lab.load_log()) == lab.HELDOUT_ALLOWANCE
    lab.main(["submit", "heldout", str(f), "--name", "forced", "--force"])
    assert lab.load_log()[-1]["forced"] is True


@needs_data
def test_split_check_catches_leaky_column_and_unstratified_split(workspace):
    raw = data.load_raw()
    clean, _ = data.reference_clean(raw)
    tr, dv = data.stratified_split(clean, (0.9, 0.1), seed=0)
    cols = ["review_id", "text", "label"]
    tr[cols].to_csv(workspace / "outputs" / "train.csv", index=False)
    dv[cols].to_csv(workspace / "outputs" / "dev.csv", index=False)
    assert lab.check_2_2()
    srt = clean.sort_values("stars")  # "last 10% of the file" on a sorted file
    srt.iloc[: int(len(srt) * 0.9)].assign(stars_display="x")[cols + ["stars_display"]].to_csv(
        workspace / "outputs" / "train.csv", index=False)
    srt.iloc[int(len(srt) * 0.9):][cols].to_csv(workspace / "outputs" / "dev.csv", index=False)
    assert not lab.check_2_2()


@needs_data
@pytest.mark.parametrize("fit_on_dev_too", [False, True])
def test_vectorizer_leakage_check(workspace, fit_on_dev_too):
    import pickle

    from sklearn.feature_extraction.text import TfidfVectorizer
    (workspace / "data" / "lab1").mkdir(parents=True)
    train, dev = data.load_prepared()
    train.to_csv(workspace / "data" / "lab1" / "train.csv", index=False)
    texts = pd.concat([train.text, dev.text]) if fit_on_dev_too else train.text
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(texts)
    (workspace / "outputs" / "vectorizer.pkl").write_bytes(pickle.dumps(vec))
    if fit_on_dev_too:  # capped vocabulary looks identical; the IDF test must still catch it
        with pytest.raises(lab.CheckFailed, match="leakage"):
            lab._check_vectorizer_fitted_on_train()
    else:
        lab._check_vectorizer_fitted_on_train()
