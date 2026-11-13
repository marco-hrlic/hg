from mdlc.baselines import LexiconClassifier, MajorityClassifier
from mdlc.slices import tag_slices
from mdlc.text import clean_text, cluster_pairs, find_near_duplicates, normalize_for_dedup


def test_clean_text():
    assert clean_text("<p>Fish &amp; chips<br />were great</p>") == "Fish & chips were great"
    assert clean_text("line one\\nline two") == "line one line two"
    assert clean_text(None) == ""


def test_normalize_for_dedup_ignores_case_and_punctuation():
    assert normalize_for_dedup("Great food!!  Loved it.") == normalize_for_dedup("great food loved it")


def test_near_duplicates_found_and_unrelated_not():
    base = ("we came here on a friday night and the wait was long but the staff were lovely and "
            "the pasta was cooked perfectly with a rich tomato sauce and fresh basil")
    texts = [base, base + " Update: still true.", "completely different review about a car wash that "
             "took forever and left streaks all over the windows of my truck"]
    pairs = find_near_duplicates(texts, threshold=0.7)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert list(cluster_pairs(3, pairs)) == [0, 0, 2]


def test_lexicon_negation():
    plain, neg = LexiconClassifier(), LexiconClassifier(handle_negation=True)
    assert plain.predict(["the food was not good"]) == ["positive"]
    assert neg.predict(["the food was not good"]) == ["negative"]


def test_majority():
    assert MajorityClassifier().fit([1, 2, 3], ["a", "b", "b"]).predict([0, 0]) == ["b", "b"]


def test_slices():
    tags = tag_slices("The food was great but the waiter was rude.")
    assert {"contrast", "mixed_signals", "short"} <= set(tags)
