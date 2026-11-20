import numpy as np
import pandas as pd
import pytest

from mdlc import data


@pytest.fixture(scope="module")
def fake_source():
    rng = np.random.default_rng(1)
    words = [f"w{i}" for i in range(3000)]
    rows = [{"stars": s, "text": " ".join(rng.choice(words, 30))} for s in range(1, 6) for _ in range(8000)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def built(fake_source, tmp_path_factory):
    raw, key = data.build_raw(fake_source, seed=7)
    d = tmp_path_factory.mktemp("d")
    (d / "raw").mkdir()
    raw.to_csv(d / "raw" / "reviews_raw.csv", index=False)
    key.to_csv(d / "raw" / "answer_key.csv", index=False)
    return raw, key, d


def test_build_is_deterministic(fake_source, built):
    raw2, _ = data.build_raw(fake_source, seed=7)
    pd.testing.assert_frame_equal(built[0], raw2)


def test_planted_counts(built):
    raw, key, _ = built
    assert len(raw) == data.N_BASE + 700
    assert key["issue"].value_counts().to_dict() == {
        "html_markup": 1000, "label_noise": 600, "exact_duplicate": 300, "near_duplicate": 300,
        "invalid_label": 60, "conflicting_duplicate": 60, "empty_or_placeholder": 40}
    assert raw["review_id"].is_unique


def test_label_noise_flips_polarity(built):
    _, key, _ = built
    noise = key[key["issue"] == "label_noise"]
    for orig, new in zip(noise["original_stars"], noise["stars"]):
        assert data.stars_to_sentiment(int(orig)) != data.stars_to_sentiment(int(new))


def test_file_sorted_and_leaky_column(built):
    raw, _, _ = built
    s = pd.to_numeric(raw["stars"], errors="coerce").dropna()
    assert s.is_monotonic_increasing
    valid = raw[raw["stars"].astype(str).isin(list("12345"))]
    assert (valid["stars_display"].str.count("★") == valid["stars"].astype(int)).all()


def test_score_findings_duplicates_either_copy(built):
    _, key, d = built
    dups = key[key["issue"] == "exact_duplicate"]
    # flag the originals rather than the planted copies: still counts as caught
    res = data.score_findings(dups["related"], "exact_duplicate", data_dir=d)
    assert res["caught"] == 300 and res["flagged_but_not_planted"] == 0
    combo = data.score_findings([], ["exact_duplicate", "near_duplicate"], data_dir=d)
    assert combo["planted"] == 600 and combo["missed"] == 600


def test_heldout_is_locked():
    with pytest.raises(PermissionError):
        data.load_heldout()


def test_versioning_roundtrip(tmp_path):
    df = pd.DataFrame({"text": ["a", "b"], "label": ["positive", "negative"]})
    m = data.save_versioned({"train": df}, tmp_path)
    assert m["splits"]["train"]["rows"] == 2
    pd.testing.assert_frame_equal(data.load_splits(tmp_path)["train"], df)
    (tmp_path / "train.csv").write_text("text,label\nc,neutral\n")
    with pytest.warns(UserWarning):
        data.load_splits(tmp_path)


def test_stratified_split_keeps_label_mix_and_partitions():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"review_id": range(1000), "label": rng.choice(["a", "b", "c"], 1000, p=[.6, .3, .1])})
    tr, dv = data.stratified_split(df, (0.9, 0.1), seed=1)
    assert len(tr) + len(dv) == 1000 and not set(tr.review_id) & set(dv.review_id)
    mix = lambda d: d.label.value_counts(normalize=True).sort_index()  # noqa: E731
    assert (mix(tr) - mix(dv)).abs().max() < 0.01


def test_reference_clean_removes_planted_row_problems(built):
    raw, key, _ = built
    clean, log = data.reference_clean(raw)
    assert "stars_display" not in clean
    gone = set(raw.review_id) - set(clean.review_id)
    for issue in ["invalid_label", "empty_or_placeholder"]:
        assert key.loc[key.issue == issue, "review_id"].isin(gone).all(), issue
    conflicting = key[key.issue == "conflicting_duplicate"]
    assert conflicting.review_id.isin(gone).all() and conflicting.related.isin(gone).all()
    exact = key[key.issue == "exact_duplicate"]  # one copy of each pair goes, the other stays
    assert (exact.review_id.isin(gone) ^ exact.related.isin(gone)).all()
    assert log["final"] == len(clean)
