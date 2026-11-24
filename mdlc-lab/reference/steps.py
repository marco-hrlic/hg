"""Reference outputs for every step. Used by `lab catchup STEP` and by the test suite.

This is *one* good solution, kept short. Participants' agent-written code will look different;
the harness checks outputs, not code.

    python reference/steps.py 1.4
"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mdlc import LABELS, data, text  # noqa: E402
from mdlc.baselines import LexiconClassifier  # noqa: E402
from mdlc.nn import encode_labels, predict_proba, train_classifier  # noqa: E402

WS = Path(os.environ.get("MDLC_WORKSPACE", ROOT / "workspace"))
OUT = WS / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def lab1_data():
    return (pd.read_csv(WS / "data/lab1/train.csv", keep_default_na=False),
            pd.read_csv(WS / "data/lab1/dev.csv", keep_default_na=False))


def eval_texts():
    return (pd.read_csv(WS / "data/eval/golden_texts.csv", keep_default_na=False),
            pd.read_csv(WS / "data/eval/heldout_texts.csv", keep_default_na=False))


def vectorizer():
    return TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def labels_of(model, X):
    return [LABELS[i] for i in predict_proba(model, X).argmax(axis=1)]


def write_preds(ids, labels, name):
    pd.DataFrame({"id": list(ids), "label": list(labels)}).to_csv(OUT / name, index=False)


def fit(train, dev, seed=0, **kw):
    vec = vectorizer().fit(train.text)
    model, hist = train_classifier(vec.transform(train.text), encode_labels(train.label),
                                   vec.transform(dev.text), encode_labels(dev.label), seed=seed, **kw)
    return model, vec, hist


def save_model(model, vec):
    torch.save(model.state_dict(), OUT / "model.pt")
    (OUT / "vectorizer.pkl").write_bytes(pickle.dumps(vec))


def load_model():
    vec = pickle.loads((OUT / "vectorizer.pkl").read_bytes())
    m = torch.nn.Sequential(torch.nn.Linear(len(vec.vocabulary_), 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
    m.load_state_dict(torch.load(OUT / "model.pt"))
    return m, vec


# ---- lab 1 ---------------------------------------------------------------

def step_1_2():
    train, dev = lab1_data()
    majority = train.label.value_counts().idxmax()
    write_preds(dev.review_id, [majority] * len(dev), "dev_majority.csv")
    write_preds(dev.review_id, LexiconClassifier(handle_negation=True).predict(dev.text), "dev_keywords.csv")


def step_1_4():
    train, dev = lab1_data()
    model, vec, hist = fit(train, dev, epochs=6)
    pd.DataFrame(hist).to_csv(OUT / "training_log.csv", index=False)
    write_preds(dev.review_id, labels_of(model, vec.transform(dev.text)), "dev_network.csv")
    save_model(model, vec)


def step_1_5():
    train, dev = lab1_data()
    rows = []
    for name, change, predicted, kw in [
        ("baseline", "hidden=32, lr=0.001", "-", {}),
        ("bigger", "hidden=128", "overfits sooner, similar best score", {"hidden": 128}),
        ("fast lr", "lr=0.05", "loss jumps around, worse", {"lr": 0.05}),
    ]:
        model, vec, hist = fit(train, dev, **kw)
        rows.append({"experiment": name, "change": change, "predicted": predicted,
                     "dev_macro_f1": round(max(h["dev_macro_f1"] for h in hist), 3), "notes": ""})
    pd.DataFrame(rows).to_csv(OUT / "experiments.csv", index=False)


def step_1_6():
    if not (OUT / "model.pt").exists():
        step_1_4()
    model, vec = load_model()
    golden, heldout = eval_texts()
    write_preds(golden.item_id, labels_of(model, vec.transform(golden.text)), "golden_lab1.csv")
    write_preds(heldout.review_id, labels_of(model, vec.transform(heldout.text)), "heldout_lab1.csv")


def step_1_7():
    if not (OUT / "model.pt").exists():
        step_1_4()
    model, vec = load_model()
    tries = [("Not bad at all, actually.", "positive"),
             ("Oh great, another 45 minute wait for a cold burger. Fantastic.", "negative"),
             ("Food was amazing but the service was so rude I'm never coming back.", "negative"),
             ("Better than the place next door, which isn't saying much.", "neutral"),
             ("meh", "neutral"),
             ("I wanted to hate it. I couldn't.", "positive")]
    pred = labels_of(model, vec.transform([t for t, _ in tries]))
    pd.DataFrame({"text": [t for t, _ in tries], "expected": [e for _, e in tries], "predicted": pred}).to_csv(
        OUT / "break_it.csv", index=False)


# ---- lab 2 ---------------------------------------------------------------

def step_2_1():
    raw = pd.read_csv(WS / "data/lab2/reviews_raw.csv", dtype=str, keep_default_na=False)
    flags = []
    flags += [(r, "invalid_label") for r in raw.review_id[raw.stars.map(data.parse_stars).isna()]]
    flags += [(r, "placeholder") for r in raw.review_id[raw.text.map(data.is_placeholder)]]
    html = raw.text.str.contains(r"<[a-z/][^>]*>|&[a-z#0-9]+;", regex=True)
    flags += [(r, "html") for r in raw.review_id[html]]
    norm = raw.text.map(text.normalize_for_dedup)
    flags += [(r, "duplicate") for r in raw.review_id[norm.duplicated(keep="first")]]
    conflicting = raw.assign(n=norm).groupby("n").stars.transform("nunique") > 1
    flags += [(r, "conflicting_duplicate") for r in raw.review_id[conflicting & (norm.str.len() > 2)]]
    flags += [(raw.review_id[j], "near_duplicate") for _, j, _ in text.find_near_duplicates(raw.text.tolist())]
    pd.DataFrame(flags, columns=["review_id", "issue"]).drop_duplicates().to_csv(OUT / "issues.csv", index=False)


def step_2_2():
    raw = pd.read_csv(WS / "data/lab2/reviews_raw.csv", dtype=str, keep_default_na=False)
    clean, _ = data.reference_clean(raw)
    train, dev = data.stratified_split(clean, (0.9, 0.1), seed=42)
    cols = ["review_id", "text", "label"]
    data.save_versioned({"train": train[cols], "dev": dev[cols]}, OUT,
                        note="reference_clean + stratified 90/10, seed 42")  # writes the CSVs + manifest.json


def lab2_data():
    if not (OUT / "train.csv").exists():
        step_2_2()
    return (pd.read_csv(OUT / "train.csv", keep_default_na=False), pd.read_csv(OUT / "dev.csv", keep_default_na=False))


def step_2_4():
    shutil.copy(ROOT / "reference" / "mathlab.py", WS / "mathlab.py")


def class_weights(train, kind):
    counts = train.label.value_counts()
    inv = np.array([len(train) / (len(LABELS) * counts[lab]) for lab in LABELS])
    return {"none": None, "sqrt_inverse": np.sqrt(inv), "inverse": inv}[kind]


def step_2_5():
    train, dev = lab2_data()
    rows = []
    for kind in ("none", "sqrt_inverse", "inverse"):
        model, vec, _ = fit(train, dev, class_weights=class_weights(train, kind))
        pred = labels_of(model, vec.transform(dev.text))
        row = {"weighting": kind, "dev_macro_f1": None}
        from mdlc import metrics
        rep = metrics.classification_report(dev.label, pred)
        row["dev_macro_f1"] = round(rep.loc["macro avg", "f1"], 3)
        row.update({f"recall_{lab}": round(rep.loc[lab, "recall"], 3) for lab in LABELS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "class_weights.csv", index=False)


def step_2_6():
    train, dev = lab2_data()
    halves = np.array_split(np.random.default_rng(0).permutation(len(train)), 2)
    losses = np.zeros(len(train))
    for a, b in [(0, 1), (1, 0)]:
        part = train.iloc[halves[a]]
        model, vec, _ = fit(part, dev)
        P = predict_proba(model, vec.transform(train.text.iloc[halves[b]]))
        y = encode_labels(train.label.iloc[halves[b]])
        losses[halves[b]] = -np.log(P[np.arange(len(y)), y] + 1e-12)
    train.assign(loss=losses).nlargest(600, "loss")[["review_id", "loss"]].to_csv(OUT / "suspects.csv", index=False)


def step_2_8():
    pd.read_csv(WS / "data/lab2/synthetic_pregenerated.csv")[["text", "label"]].to_csv(OUT / "synthetic.csv", index=False)


def more_reviews(n=None):
    more = pd.read_csv(WS / "data/lab2/more_reviews.csv", keep_default_na=False)
    return more if n is None else more.iloc[:n]


def step_2_7():
    import time
    train, dev = lab2_data()
    rows = []
    for n in (0, 32_000, 82_000):
        t = time.time()
        model, vec, _ = fit(pd.concat([train, more_reviews(n)], ignore_index=True), dev)
        from mdlc import metrics
        f1 = metrics.macro_f1(dev.label, labels_of(model, vec.transform(dev.text)))
        rows.append({"n_train": len(train) + n, "dev_macro_f1": round(f1, 4), "seconds": round(time.time() - t)})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "learning_curve.csv", index=False)


def step_2_9():
    """Candidates 0 (lab 1 recipe), A (class weights), B (A + suspects removed), C (B + synthetic),
    D (B + more data); seeds 0-1."""
    train, dev = lab2_data()
    if not (OUT / "suspects.csv").exists():
        step_2_6()
    suspects = set(pd.read_csv(OUT / "suspects.csv").review_id)
    cleaned = train[~train.review_id.isin(suspects)]
    if not (OUT / "synthetic.csv").exists():
        step_2_8()
    with_synthetic = pd.concat([cleaned, pd.read_csv(OUT / "synthetic.csv")], ignore_index=True)
    more = pd.concat([cleaned, more_reviews()], ignore_index=True)
    candidates = {
        "0": (train, None),
        "A": (train, class_weights(train, "sqrt_inverse")),
        "B": (cleaned, class_weights(cleaned, "sqrt_inverse")),
        "C": (with_synthetic, class_weights(with_synthetic, "sqrt_inverse")),
        "D": (more, class_weights(more, "sqrt_inverse")),
    }
    golden, heldout = eval_texts()
    for name, (df, w) in candidates.items():
        for seed in range(2):
            model, vec, _ = fit(df, dev, seed=seed, class_weights=w)
            write_preds(dev.review_id, labels_of(model, vec.transform(dev.text)), f"dev_{name}_s{seed}.csv")
            if seed == 0:
                write_preds(golden.item_id, labels_of(model, vec.transform(golden.text)), f"golden_{name}_s0.csv")
                write_preds(heldout.review_id, labels_of(model, vec.transform(heldout.text)), f"heldout_{name}_s0.csv")
            print(f"candidate {name} seed {seed} done", flush=True)


STEPS = {k.removeprefix("step_").replace("_", "."): v for k, v in dict(globals()).items() if k.startswith("step_")}

if __name__ == "__main__":
    step = sys.argv[1]
    if step not in STEPS:
        print(f"No reference outputs for step {step}. Available: {', '.join(sorted(STEPS))}")
        sys.exit(1)
    STEPS[step]()
    print(f"done: reference outputs for step {step} are in {OUT}")
