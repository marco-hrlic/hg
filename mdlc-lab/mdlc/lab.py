"""The `lab` command: step cards, independent checks, and a scorer with hidden evaluation labels.

Participants build the project with Claude Code in `workspace/`. This harness is the part the
agent doesn't write: it holds the evaluation labels and the answer key, scores prediction files,
rations held-out submissions, and checks each step's outputs for the classic mistakes.

    lab setup                      build data and export the participant's view of it into workspace/data
    lab steps                      list the step cards
    lab step 1.4                   print a step card
    lab check 1.4                  check that step's outputs
    lab score FILE [--truth FILE]  score dev predictions (lab 1 or lab 2 dev, picked by ids)
    lab submit golden FILE --name NAME     score on the golden set (hidden labels), per slice
    lab submit heldout FILE --name NAME    score on the held-out set (hidden labels), rationed
    lab log                        every submission so far
    lab compare A B [--on dev|golden|heldout]   paired bootstrap: is candidate A really better than B?
    lab catchup 1.4                write the reference outputs for a step, to catch up
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABELS, data, metrics, text

LAB_ROOT = data.LAB_ROOT
STEPS_DIR = LAB_ROOT / "steps"
REFERENCE_DIR = LAB_ROOT / "reference"
WORKSPACE = Path(os.environ.get("MDLC_WORKSPACE", LAB_ROOT / "workspace"))
HELDOUT_ALLOWANCE = 2  # one look at the end of each lab
MIN_SLICE_FOR_F1 = 30  # golden slices smaller than this show accuracy only

OK, FAIL, WARN = "✓", "✗", "!"


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def ws(*parts: str) -> Path:
    return WORKSPACE.joinpath(*parts)


def say(mark: str, msg: str) -> None:
    print(f"  {mark} {msg}")


def heading(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


class CheckFailed(Exception):
    pass


def require_file(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise CheckFailed(f"missing {path.relative_to(WORKSPACE)}. The step card says what to create.")
    df = pd.read_csv(path, keep_default_na=False)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise CheckFailed(f"{path.name} needs columns {columns}; missing {missing} (has {list(df.columns)})")
    return df


def read_predictions(path: Path) -> pd.Series:
    """Prediction files have an id column (id, review_id or item_id) and a label column."""
    if not Path(path).exists():
        raise CheckFailed(f"no such file: {path}")
    df = pd.read_csv(path, keep_default_na=False)
    id_col = next((c for c in ("id", "review_id", "item_id") if c in df.columns), None)
    if id_col is None or "label" not in df.columns:
        raise CheckFailed(f"{Path(path).name}: expected columns `id,label` (got {list(df.columns)})")
    bad = sorted(set(df["label"]) - set(LABELS))
    if bad:
        raise CheckFailed(f"{Path(path).name}: labels must be one of {LABELS}; found {bad[:5]}")
    if df[id_col].duplicated().any():
        raise CheckFailed(f"{Path(path).name}: some ids appear more than once")
    return df.set_index(id_col)["label"]


def align(pred: pd.Series, truth: pd.Series, what: str) -> pd.Series:
    missing = truth.index.difference(pred.index)
    if len(missing):
        raise CheckFailed(f"predictions are missing {len(missing)} of the {len(truth)} {what} items "
                          f"(e.g. {list(missing[:3])})")
    return pred.loc[truth.index]


def summary(truth: pd.Series, pred: pd.Series) -> dict:
    return {"accuracy": round(metrics.accuracy(truth, pred), 4), "macro_f1": round(metrics.macro_f1(truth, pred), 4)}


def print_scores(truth: pd.Series, pred: pd.Series, title: str) -> dict:
    acc, f1 = metrics.accuracy(truth, pred), metrics.macro_f1(truth, pred)
    heading(f"{title}: accuracy {acc:.3f}   macro-F1 {f1:.3f}")
    cm = metrics.confusion_matrix(truth, pred)
    print(pd.DataFrame(cm, index=[f"true {lab}" for lab in LABELS],
                       columns=[f"pred {lab}" for lab in LABELS]).to_string())
    print()
    print(metrics.classification_report(truth, pred).to_string())
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4)}


def state_path() -> Path:
    return ws(".lab", "submissions.jsonl")


def load_log() -> list[dict]:
    p = state_path()
    return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []


def append_log(entry: dict) -> None:
    state_path().parent.mkdir(parents=True, exist_ok=True)
    with open(state_path(), "a") as f:
        f.write(json.dumps(entry) + "\n")


def golden_truth() -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return data.load_golden()


# ---------------------------------------------------------------------------
# setup: export the participant's view of the data (no evaluation labels)
# ---------------------------------------------------------------------------

def cmd_setup(args) -> None:
    paths = data.build_all()
    d = ws("data")
    for sub in ("lab1", "lab2", "eval"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    train, dev = data.load_prepared()
    cols = ["review_id", "text", "label"]
    train[cols].to_csv(d / "lab1" / "train.csv", index=False)
    dev[cols].to_csv(d / "lab1" / "dev.csv", index=False)
    shutil.copy(paths["raw"], d / "lab2" / "reviews_raw.csv")

    golden = golden_truth()
    golden.assign(slices=golden["slices"].map(";".join))[["item_id", "text", "slices"]].to_csv(
        d / "eval" / "golden_texts.csv", index=False)
    heldout = pd.read_csv(paths["heldout"])
    heldout[["review_id", "text"]].to_csv(d / "eval" / "heldout_texts.csv", index=False)
    exercise = pd.read_csv(paths["exercise"])
    exercise = exercise[exercise["item_id"].isin(golden["item_id"])].head(10)
    exercise[["item_id", "text"]].to_csv(d / "lab2" / "label_these_10.csv", index=False)
    shutil.copy(LAB_ROOT / "golden_labels" / "GUIDELINE.md", d / "lab2" / "GUIDELINE.md")
    shutil.copy(paths["more"], d / "lab2" / "more_reviews.csv")
    synth = LAB_ROOT / "synthetic" / "synthetic_reviews.jsonl"
    pd.read_json(synth, lines=True).to_csv(d / "lab2" / "synthetic_pregenerated.csv", index=False)

    ws("outputs").mkdir(exist_ok=True)
    print(f"Workspace data ready in {d}:")
    for p in sorted(d.rglob("*.csv")):
        print(f"  {p.relative_to(WORKSPACE)}  ({sum(1 for _ in open(p)) - 1:,} rows)")
    print("\nEvaluation sets contain texts only. Their labels stay with the harness: use `lab submit`.")


# ---------------------------------------------------------------------------
# step cards
# ---------------------------------------------------------------------------

def _card(step: str) -> Path:
    lab, num = step.split(".")
    matches = sorted((STEPS_DIR / f"lab{lab}").glob(f"{int(num):02d}-*.md"))
    if not matches:
        raise CheckFailed(f"no step {step}. Run `lab steps` for the list.")
    return matches[0]


def cmd_steps(args) -> None:
    for lab_dir in sorted(STEPS_DIR.glob("lab*")):
        print(f"\n{lab_dir.name}")
        for p in sorted(lab_dir.glob("*.md")):
            first = re.sub(r"^#\s*(\d+\.\d+\s+)?", "", p.read_text().splitlines()[0])
            print(f"  {lab_dir.name[3:]}.{int(p.name[:2])}  {first}")


def cmd_step(args) -> None:
    print(_card(args.step).read_text())


# ---------------------------------------------------------------------------
# scoring and submissions
# ---------------------------------------------------------------------------

def _dev_truth_for(pred: pd.Series) -> tuple[pd.Series, Path]:
    """The dev split these predictions belong to: lab 2's outputs/dev.csv or lab 1's data/lab1/dev.csv,
    whichever shares the most ids with the prediction file."""
    candidates = [p for p in (ws("outputs", "dev.csv"), ws("data", "lab1", "dev.csv")) if p.exists()]
    if not candidates:
        raise CheckFailed("no dev set found: run `lab setup` first")
    truths = {p: pd.read_csv(p, keep_default_na=False).set_index("review_id")["label"] for p in candidates}
    best = max(truths, key=lambda p: truths[p].index.isin(pred.index).mean())
    return truths[best], best


def cmd_score(args) -> None:
    pred = read_predictions(Path(args.file))
    if args.truth:
        path = Path(args.truth)
        truth = pd.read_csv(path, keep_default_na=False).set_index("review_id")["label"]
    else:
        truth, path = _dev_truth_for(pred)
    pred = align(pred, truth, "dev")
    print_scores(truth, pred, f"dev score for {Path(args.file).name} (truth: {path.name} in {path.parent.name}/)")


def cmd_submit(args) -> None:
    pred_all = read_predictions(Path(args.file))
    log = load_log()
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "set": args.set,
             "name": args.name, "file": str(args.file),
             "sha256": hashlib.sha256(Path(args.file).read_bytes()).hexdigest()[:12]}
    if args.set == "heldout":
        used = [e for e in log if e["set"] == "heldout"]
        if len(used) >= HELDOUT_ALLOWANCE and not args.force:
            print(f"Refused: the held-out set has already been used {len(used)} times "
                  f"({', '.join(e['name'] for e in used)}).")
            print("Every look at it, followed by a change, turns it into another dev set. If you really "
                  "must, rerun with --force; the log will show it.")
            sys.exit(1)
        entry["forced"] = bool(args.force and len(used) >= HELDOUT_ALLOWANCE)
        truth = pd.read_csv(data.DATA_DIR / "heldout" / "heldout.csv").set_index("review_id")["label"]
        pred = align(pred_all, truth, "held-out")
        append_log({**entry, **summary(truth, pred)})  # log before showing anything
        print_scores(truth, pred, f"HELD-OUT: {args.name}")
        print(f"\nHeld-out looks used so far: {len(used) + 1} of {HELDOUT_ALLOWANCE} "
              f"(one per lab; this includes earlier labs: {', '.join(e['name'] for e in used) or 'none'}).")
    else:
        g = golden_truth()
        truth = g.set_index("item_id")["label"]
        pred = align(pred_all, truth, "golden")
        lo, hi = metrics.bootstrap_ci(truth.values, pred.values)
        append_log({**entry, **summary(truth, pred), "macro_f1_ci": [round(lo, 3), round(hi, 3)]})
        print_scores(truth, pred, f"GOLDEN: {args.name}")
        print(f"\n95% bootstrap interval for macro-F1: {lo:.3f} to {hi:.3f}. With {len(truth)} items (only "
              f"{(truth == 'neutral').sum()} neutral), differences smaller than this band are mostly noise.")
        heading("per slice (where the hard cases are)")
        rep = metrics.slice_report(g.assign(pred=pred.values), "label", "pred")
        small = rep["n"] < MIN_SLICE_FOR_F1
        shown = rep.astype({"macro_f1": object})
        shown.loc[small, "macro_f1"] = "-"  # macro-F1 on a handful of items swings wildly
        print(shown.loc[rep.sort_values("accuracy").index].to_string())
        if small.any():
            print(f"(macro-F1 hidden for slices under {MIN_SLICE_FOR_F1} items: one review can swing it "
                  "by 0.2. Read their accuracy, and treat it as a hint.)")
        n = sum(1 for e in log if e["set"] == "golden") + 1  # log was read before this submission
        print(f"\n{g.attrs.get('label_source', '')} labels, {len(g)} items. Golden submissions so far: {n}. "
              "Use it to compare finished candidates, not to tune.")


def _dev_truth() -> pd.Series:
    """Lab 2's own dev split if it exists, otherwise lab 1's."""
    p = ws("outputs", "dev.csv") if ws("outputs", "dev.csv").exists() else ws("data", "lab1", "dev.csv")
    return pd.read_csv(p, keep_default_na=False).set_index("review_id")["label"]


def _truth(on: str) -> pd.Series:
    if on == "golden":
        return golden_truth().set_index("item_id")["label"]
    if on == "heldout":
        return pd.read_csv(data.DATA_DIR / "heldout" / "heldout.csv").set_index("review_id")["label"]
    return _dev_truth()


def _runs(on: str) -> dict[str, list[pd.Series]]:
    """Prediction runs grouped by candidate. Dev: files outputs/dev_<candidate>_s<seed>.csv.
    Golden and held-out: logged submissions, grouped by name without its -sN suffix."""
    runs: dict[str, list[pd.Series]] = {}
    if on == "dev":
        for path in sorted(ws("outputs").glob("dev_*_s*.csv")):
            m = re.fullmatch(r"dev_(.+)_s\d+\.csv", path.name)
            if m:
                runs.setdefault(m.group(1), []).append(read_predictions(path))
        return runs
    for e in load_log():
        if e["set"] != on:
            continue
        path = Path(e["file"])
        path = path if path.is_absolute() else WORKSPACE / path
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()[:12] != e["sha256"]:
            continue  # file changed or gone since it was submitted: can't use it for a fair comparison
        cand = re.sub(r"-s\d+$", "", e["name"])
        runs.setdefault(cand, []).append(read_predictions(path))
    return runs


def compare_candidates(a: str, b: str, on: str = "dev") -> dict:
    g = _truth(on)
    runs = _runs(on)
    for name in (a, b):
        if name not in runs:
            where = "files outputs/dev_<name>_s<seed>.csv" if on == "dev" else f"{on} submissions (unchanged files)"
            raise CheckFailed(f"no {where} for candidate {name}. Have: {sorted(runs) or 'none'}")
    res = metrics.paired_bootstrap_diff(g.values, [align(p, g, on).values for p in runs[a]],
                                        [align(p, g, on).values for p in runs[b]])
    res.update(n_a=len(runs[a]), n_b=len(runs[b]))
    return res


def describe_comparison(a: str, b: str, r: dict) -> str:
    verdict = ("significantly better" if r["lo"] > 0 else "significantly worse" if r["hi"] < 0
               else "not distinguishable (the interval includes 0)")
    return (f"{a} vs {b}: macro-F1 {r['diff']:+.3f}, 95% interval [{r['lo']:+.3f}, {r['hi']:+.3f}] "
            f"({r['n_a']} vs {r['n_b']} runs): {verdict}")


def cmd_compare(args) -> None:
    r = compare_candidates(args.a, args.b, args.on)
    print(f"[{args.on}, {len(_truth(args.on)):,} items] " + describe_comparison(args.a, args.b, r))
    print("Paired bootstrap: both are re-scored on the same resampled items, so the luck of the item "
          "sample mostly cancels out; several seeds per side average out run-to-run noise. The bigger the "
          "evaluation set, the smaller the difference it can detect.")


def cmd_log(args) -> None:
    log = load_log()
    if not log:
        print("No submissions yet.")
        return
    df = pd.DataFrame(log)
    print(df[[c for c in ["time", "set", "name", "accuracy", "macro_f1", "forced"] if c in df]].to_string(index=False))


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def _answer_key() -> pd.DataFrame:
    return pd.read_csv(data.DATA_DIR / "raw" / "answer_key.csv", dtype=str, keep_default_na=False)


def _dev_truth_lab1() -> pd.Series:
    return pd.read_csv(ws("data", "lab1", "dev.csv"), keep_default_na=False).set_index("review_id")["label"]


def _dev_f1(name: str, truth: pd.Series) -> float:
    pred = align(read_predictions(ws("outputs", name)), truth, "dev")
    return metrics.macro_f1(truth, pred)


def check_1_2() -> bool:
    truth = _dev_truth_lab1()
    scores = {}
    for name in ("dev_majority.csv", "dev_keywords.csv"):
        pred = align(read_predictions(ws("outputs", name)), truth, "dev")
        scores[name] = (metrics.accuracy(truth, pred), metrics.macro_f1(truth, pred))
        say(OK, f"{name}: accuracy {scores[name][0]:.3f}, macro-F1 {scores[name][1]:.3f}")
    maj_acc, maj_f1 = scores["dev_majority.csv"]
    if maj_acc > 0.55 and maj_f1 < 0.35:
        say(OK, f"majority baseline: {maj_acc:.0%} accurate and useless. That gap is why we track macro-F1.")
    better = scores["dev_keywords.csv"][1] > maj_f1
    say(OK if better else FAIL, "keyword baseline beats majority on macro-F1" if better
        else "keyword baseline should beat majority on macro-F1: look at what it predicts")
    return better


def check_1_4() -> bool:
    log = require_file(ws("outputs", "training_log.csv"), ["epoch", "train_loss", "dev_loss"])
    if len(log) < 3:
        raise CheckFailed("training_log.csv should have at least 3 epochs")
    say(OK, f"training log: {len(log)} epochs")
    best_epoch = int(log.loc[log.dev_loss.idxmin(), "epoch"])
    if log.train_loss.iloc[-1] >= log.train_loss.iloc[0]:
        say(WARN, "training loss didn't go down. Is the optimizer actually stepping?")
    if best_epoch < int(log.epoch.max()):
        say(OK, f"dev loss bottoms out at epoch {best_epoch}, then rises while training loss keeps falling: "
                "overfitting. Did you keep the weights from the best epoch?")
    else:
        say(WARN, "dev loss still falling at the last epoch: try more epochs to see where it turns")
    _check_vectorizer_fitted_on_train()
    truth = _dev_truth_lab1()
    f1 = _dev_f1("dev_network.csv", truth)
    acc = metrics.accuracy(truth, align(read_predictions(ws("outputs", "dev_network.csv")), truth, "dev"))
    say(OK, f"network on dev: accuracy {acc:.3f}, macro-F1 {f1:.3f}")
    if acc > 0.92:
        say(WARN, "suspiciously good. Check for leakage: was the dev set used for training, or the "
                  "vectorizer fitted on dev too?")
    best_base = max((_dev_f1(n, truth) for n in ("dev_keywords.csv", "dev_majority.csv")
                     if ws("outputs", n).exists()), default=0.0)
    ok = f1 > best_base
    say(OK if ok else FAIL, f"beats the best baseline ({best_base:.3f})" if ok
        else f"doesn't beat the best baseline ({best_base:.3f}) yet")
    return ok


def _check_vectorizer_fitted_on_train() -> None:
    """Verify the vectorizer learned only from train.

    Two tests: every vocabulary term must occur in train, and, for TF-IDF, every stored IDF must equal
    the IDF recomputed from train alone (IDF depends on how many documents were seen and which
    contained the term, so fitting on train + dev changes it even when the vocabulary looks the same).
    """
    import pickle
    from collections import Counter
    path = ws("outputs", "vectorizer.pkl")
    if not path.exists():
        raise CheckFailed("missing outputs/vectorizer.pkl (the fitted vectorizer, pickled)")
    try:
        vec = pickle.loads(path.read_bytes())
        analyze = vec.build_analyzer()
        vocab = vec.vocabulary_
    except Exception:  # not a scikit-learn vectorizer: can't verify, don't block
        say(WARN, "couldn't inspect vectorizer.pkl (not a scikit-learn vectorizer?): fit-on-train not verified")
        return
    train = pd.read_csv(ws("data", "lab1", "train.csv"), keep_default_na=False).text
    df = Counter()
    for t in train:
        df.update(set(analyze(t)))
    outside = [term for term in vocab if term not in df]
    if outside:
        raise CheckFailed(f"{len(outside)} vocabulary terms never occur in train (e.g. {sorted(outside)[:3]}): "
                          "the vectorizer was fitted on more than the training data. That's leakage.")
    idf = getattr(vec, "idf_", None)
    if idf is not None:
        terms = sorted(vocab, key=vocab.get)
        counts = np.array([df[t] for t in terms], dtype=float)
        n = float(len(train))
        smooth = float(getattr(vec, "smooth_idf", True))
        expected = np.log((n + smooth) / (counts + smooth)) + 1
        if not np.allclose(idf, expected, atol=1e-6):
            raise CheckFailed("the vectorizer's IDF weights don't match the training data alone: it was fitted "
                              "on more than train (e.g. train + dev). That's leakage. Fit on train, then transform.")
    say(OK, f"vectorizer ({len(vocab):,} terms) was fitted on train only")


def check_1_5() -> bool:
    exp = require_file(ws("outputs", "experiments.csv"), ["experiment", "change", "predicted", "dev_macro_f1"])
    print(exp.to_string(index=False))
    ok = len(exp) >= 3
    say(OK if ok else FAIL, f"{len(exp)} experiments logged" + ("" if ok else " (need at least 3)"))
    if (exp.predicted.astype(str).str.strip() == "").any():
        say(FAIL, "every experiment needs a prediction written *before* running it")
        ok = False
    spread = pd.to_numeric(exp.dev_macro_f1, errors="coerce").agg(["min", "max"])
    say(OK, f"dev macro-F1 ranged {spread['min']:.3f}–{spread['max']:.3f}. Which differences are bigger "
            "than what a different random seed alone would cause?")
    return ok


def check_1_6() -> bool:
    log = load_log()
    sets = {e["set"] for e in log}
    for s in ("golden", "heldout"):
        say(OK if s in sets else FAIL, f"{s} submission " + ("recorded" if s in sets else "missing: see the step card"))
    return {"golden", "heldout"} <= sets


def check_1_7() -> bool:
    df = require_file(ws("outputs", "break_it.csv"), ["text", "expected", "predicted"])
    broke = (df.expected.str.strip().str.lower() != df.predicted.str.strip().str.lower()).sum()
    say(OK, f"you fooled the model {broke} times out of {len(df)}")
    return len(df) >= 5


def check_2_1() -> bool:
    issues = require_file(ws("outputs", "issues.csv"), ["review_id", "issue"])
    key = _answer_key()
    flagged = set(issues.review_id)
    heading("row-level problems (matched by review_id; your issue names don't need to match)")
    ok = True
    for issue, rows in key.groupby("issue"):
        dup = issue.endswith("duplicate")
        hit = rows.review_id.isin(flagged) | (dup & rows.related.isin(flagged))
        frac = hit.mean()
        mark = OK if frac >= 0.8 else (WARN if issue == "label_noise" or frac >= 0.4 else FAIL)
        say(mark, f"{issue:22s} caught {hit.sum():4d} / {len(rows)}")
        if issue in {"invalid_label", "empty_or_placeholder", "exact_duplicate", "conflicting_duplicate"} and frac < 0.8:
            ok = False
    planted = set(key.review_id) | set(key.loc[key.issue.str.endswith("duplicate"), "related"])
    extra = len(flagged - planted)
    say(OK if extra < 100 else WARN, f"{extra} flagged rows aren't in the answer key (some may still be real problems)")
    print("\n  label_noise: almost nobody finds flipped labels by inspecting rows. Step 6 does it with a model.")
    heading("dataset-level problems: compare with what you wrote in notes.md")
    for name, desc in data.GLOBAL_ISSUES.items():
        print(f"  • {name}: {desc}")
    return ok


def check_2_2() -> bool:
    cols = ["review_id", "text", "label"]
    tr = require_file(ws("outputs", "train.csv"), cols)
    dv = require_file(ws("outputs", "dev.csv"), cols)
    key = _answer_key()
    ok = True

    def test(cond: bool, good: str, bad: str, fatal: bool = True) -> None:
        nonlocal ok
        say(OK if cond else (FAIL if fatal else WARN), good if cond else bad)
        if fatal and not cond:
            ok = False

    both = pd.concat([tr.assign(_split="train"), dv.assign(_split="dev")], ignore_index=True)
    test(both.label.isin(LABELS).all(), "labels are negative/neutral/positive", "unexpected label values")
    leaky = [c for c in both.columns if "display" in c or c == "stars"]
    test(not leaky, "no leaky columns", f"columns {leaky} encode the label: drop them")
    gone = lambda issue: not both.review_id.isin(key.loc[key.issue == issue, "review_id"]).any()  # noqa: E731
    test(gone("invalid_label"), "invalid ratings removed", "rows with invalid ratings are still there")
    test(gone("empty_or_placeholder"), "placeholders removed", "placeholder/empty texts are still there")
    conf = key[key.issue == "conflicting_duplicate"]
    test(not both.review_id.isin(set(conf.review_id) | set(conf.related)).any(),
         "conflicting duplicates dropped (all copies)", "texts with conflicting labels are still there")
    test(not both.text.str.contains(r"<[a-z/][^>]*>|&[a-z#0-9]+;", regex=True).any(),
         "no HTML left in the text", "HTML tags or entities still in the text")
    escaped = both.text.str.contains("\\n", regex=False) | both.text.str.contains('\\"', regex=False)
    test(escaped.mean() < 0.01, "escaped characters (\\n, \\\") decoded",
         f"{escaped.sum():,} texts still contain literal \\n or \\\" escapes from the source")
    test(ws("outputs", "manifest.json").exists(), "manifest.json written", "no outputs/manifest.json (row counts, label mix, hashes)",
         fatal=False)
    norm = both.text.map(text.normalize_for_dedup)
    test(not norm.duplicated().any(), "no exact duplicates (after normalising)",
         f"{norm.duplicated().sum()} exact duplicates remain")
    test(not set(tr.review_id) & set(dv.review_id), "no review in both train and dev", "reviews in both splits")
    pairs = text.find_near_duplicates(both.text.tolist(), threshold=0.7)
    cross = sum(1 for i, j, _ in pairs if both._split[i] != both._split[j])
    test(cross == 0, "no near-duplicates across train/dev", f"{cross} near-duplicate pairs across train/dev", fatal=False)
    mix = pd.DataFrame({"train": tr.label.value_counts(normalize=True), "dev": dv.label.value_counts(normalize=True)})
    gap = (mix.train - mix.dev).abs().max()
    test(gap < 0.02, f"label mix matches within {gap:.1%} (stratified)", f"label mix differs by {gap:.1%}: stratify")
    frac = len(dv) / len(both)
    test(0.05 <= frac <= 0.25, f"dev is {frac:.0%} of the data", f"dev is {frac:.0%} of the data: aim for ~10%", fatal=False)
    noise = both.review_id.isin(key.loc[key.issue == "label_noise", "review_id"]).sum()
    print(f"\n  {len(tr):,} train / {len(dv):,} dev. Still hiding in there: {noise} flipped labels (step 6).")
    return ok


def check_2_3() -> bool:
    mine = require_file(ws("outputs", "my_labels.csv"), ["item_id", "label"])
    g = golden_truth().set_index("item_id")
    mine = mine[mine.item_id.isin(g.index)]
    if len(mine):
        k = metrics.cohen_kappa(mine.label.str.strip().str.lower().tolist(), g.loc[mine.item_id, "label"].tolist())
        say(OK, f"your labels vs the golden labels: Cohen's kappa {k:.2f} on {len(mine)} items")
        stars_k = metrics.cohen_kappa(g.loc[mine.item_id, "star_label"].tolist(), g.loc[mine.item_id, "label"].tolist())
        say(OK, f"star-derived labels vs the golden labels on the same items: kappa {stars_k:.2f}")
    heading("all golden items: label from reading the text vs label from the stars")
    print(pd.crosstab(g.label, g.stars, rownames=["golden label"], colnames=["stars"]).to_string())
    print("\n  Look at the 3-star column.")
    rel = LAB_ROOT / "relabels" / "three_star_labels.csv"
    if rel.exists():
        mix = pd.read_csv(rel).label.value_counts(normalize=True)
        heading("at scale: an LLM rater relabelled all 2,316 three-star reviews in the development data")
        print("  " + ", ".join(f"{lab} {mix.get(lab, 0):.0%}" for lab in (*LABELS, "ambiguous")))
        print("  So 'three stars = neutral' gives roughly 1,450 wrong training labels: more than twice the 600 planted flips.")
    return len(mine) >= 8


def check_2_4() -> bool:
    path = ws("mathlab.py")
    if not path.exists():
        raise CheckFailed("missing mathlab.py with softmax, cross_entropy and gradients")
    spec = importlib.util.spec_from_file_location("mathlab", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ok = True
    try:
        P = np.asarray(mod.softmax(np.array([[2.0, 1.0, 0.0], [1000.0, 1000.0, 1000.0]])))
        good = np.allclose(P[0], [0.665, 0.245, 0.090], atol=1e-3) and np.allclose(P[1], 1 / 3)
        say(OK if good else FAIL, "softmax correct, and stable for large scores" if good
            else "softmax wrong, or overflows for large scores (subtract each row's max first)")
        ok &= good
        ce = float(mod.cross_entropy(np.full((4, 3), 1 / 3), np.array([0, 1, 2, 0])))
        good = abs(ce - np.log(3)) < 1e-6
        say(OK if good else FAIL, f"cross-entropy of a uniform guess = ln 3 = {np.log(3):.3f}" if good
            else f"cross-entropy of a uniform guess should be ln 3 ≈ 1.099, got {ce:.3f}")
        ok &= good
        import scipy.sparse as sp
        import torch
        rng = np.random.default_rng(0)
        X = sp.random(32, 50, density=0.2, format="csr", random_state=0)
        y = rng.integers(0, 3, 32)
        W0, b0 = rng.normal(0, 0.1, (50, 3)), rng.normal(0, 0.1, 3)
        dW, db = mod.gradients(X, np.asarray(mod.softmax(np.asarray(X @ W0) + b0)), np.eye(3)[y])
        Wt, bt = torch.tensor(W0, requires_grad=True), torch.tensor(b0, requires_grad=True)
        torch.nn.functional.cross_entropy(torch.tensor(X.toarray()) @ Wt + bt, torch.tensor(y)).backward()
        good = np.allclose(np.asarray(dW), Wt.grad.numpy(), atol=1e-8) and np.allclose(np.asarray(db), bt.grad.numpy(), atol=1e-8)
        say(OK if good else FAIL, "gradients match PyTorch autograd" if good else "gradients don't match PyTorch autograd")
        ok &= good
    except AttributeError as e:
        raise CheckFailed(f"mathlab.py: {e}") from e
    return ok


def check_2_5() -> bool:
    df = require_file(ws("outputs", "class_weights.csv"),
                      ["weighting", "dev_macro_f1", "recall_negative", "recall_neutral", "recall_positive"])
    print(df.to_string(index=False))
    say(OK if len(df) >= 2 else FAIL, f"{len(df)} weightings compared")
    print("\n  Weights don't make the model smarter; they move where its mistakes land. Which row would "
          "you ship for 'route unhappy customers to support'? Write the decision and the reason in notes.md.")
    return len(df) >= 2


def check_2_6() -> bool:
    sus = require_file(ws("outputs", "suspects.csv"), ["review_id"])
    key = _answer_key()
    noise = set(key.loc[key.issue == "label_noise", "review_id"])
    hits = sus.review_id.isin(noise)
    say(OK if hits.sum() >= 150 else WARN, f"{hits.sum()} of your {len(sus)} suspects are planted flipped labels "
                                            f"({hits.mean():.0%} precision)")
    raw = data.load_raw().set_index("review_id")
    others = sus.loc[~hits, "review_id"]
    three = (raw.loc[raw.index.intersection(others), "stars"] == "3").sum()
    say(OK, f"of the {len(others)} other suspects, {three} are 3-star reviews: often real label errors nobody planted")
    return hits.sum() >= 150


def check_2_7() -> bool:
    lc = require_file(ws("outputs", "learning_curve.csv"), ["n_train", "dev_macro_f1"])
    lc = lc.sort_values("n_train")
    print(lc.to_string(index=False))
    ok = len(lc) >= 3 and lc.n_train.nunique() >= 3
    say(OK if ok else FAIL, f"{lc.n_train.nunique()} training-set sizes" + ("" if ok else " (need at least 3)"))
    if ok:
        gain = lc.dev_macro_f1.iloc[-1] - lc.dev_macro_f1.iloc[0]
        say(OK, f"{lc.n_train.iloc[0]:,} → {lc.n_train.iloc[-1]:,} reviews: dev macro-F1 {gain:+.3f}. Is the curve still "
                "rising at the right-hand end? Then more data would still help.")
    return ok


def check_2_8() -> bool:
    syn = require_file(ws("outputs", "synthetic.csv"), ["text", "label"])
    ok = syn.label.isin(LABELS).all()
    say(OK if ok else FAIL, f"{len(syn)} synthetic reviews, labels valid" if ok else "invalid labels")
    say(OK, "label mix: " + ", ".join(f"{k} {v}" for k, v in syn.label.value_counts().items()))
    eval_texts = (pd.read_csv(ws("data", "eval", "golden_texts.csv")).text.tolist()
                  + pd.read_csv(ws("data", "eval", "heldout_texts.csv")).text.tolist())
    combined = syn.text.tolist() + eval_texts
    leaks = sum(1 for i, j, _ in text.find_near_duplicates(combined, threshold=0.5) if i < len(syn) <= j)
    say(OK if leaks == 0 else FAIL, "no synthetic review near-duplicates an evaluation item" if leaks == 0
        else f"{leaks} synthetic reviews near-duplicate evaluation items: remove them")
    if ws("outputs", "train.csv").exists():
        share = len(syn) / (len(syn) + len(pd.read_csv(ws("outputs", "train.csv"))))
        say(OK if share < 0.2 else WARN, f"synthetic share of training data: {share:.1%}")
    return ok and leaks == 0


def check_2_9() -> bool:
    runs = _runs("dev")
    seeded = {c: len(r) for c, r in runs.items() if len(r) >= 2}
    heading("dev: paired comparison with candidate 0 (bootstrap over dev items, averaged over seeds)")
    if "0" not in seeded:
        raise CheckFailed("need outputs/dev_0_s0.csv, dev_0_s1.csv, ...: candidate 0 (lab 1's recipe, retrained in "
                          "the same script) with at least 2 seeds")
    for cand in sorted(c for c in seeded if c != "0"):
        say(OK, describe_comparison(cand, "0", compare_candidates(cand, "0", "dev")))
    held = [e for e in load_log() if e["set"] == "heldout"]
    names = [e["name"] for e in held]
    heading("held-out: your lab 2 model vs the lab 1 model (one look each)")
    if "lab1" not in names or len(held) < 2:
        say(FAIL, f"need a held-out submission named lab1 (from lab 1) and one for your lab 2 model; have {names or 'none'}")
        return False
    final = held[-1]["name"]
    say(OK, describe_comparison(final, "lab1", compare_candidates(final, "lab1", "heldout")))
    golden = [e["name"] for e in load_log() if e["set"] == "golden"]
    if golden:
        print(f"\n  Golden submissions: {', '.join(golden)}. Use `lab compare X Y --on golden` to see why the "
              "small golden set can't settle the same question.")
    return True


CHECKS = {"1.2": check_1_2, "1.4": check_1_4, "1.5": check_1_5, "1.6": check_1_6, "1.7": check_1_7,
          "2.1": check_2_1, "2.2": check_2_2, "2.3": check_2_3, "2.4": check_2_4, "2.5": check_2_5,
          "2.6": check_2_6, "2.7": check_2_7, "2.8": check_2_8, "2.9": check_2_9}


def cmd_check(args) -> None:
    fn = CHECKS.get(args.step)
    if fn is None:
        print(f"Step {args.step} has no automatic check: its card lists what to verify yourself.")
        return
    heading(f"check {args.step}")
    try:
        passed = fn()
    except CheckFailed as e:
        say(FAIL, str(e))
        passed = False
    print(f"\n{'PASSED' if passed else 'NOT YET'}: step {args.step}")
    sys.exit(0 if passed else 1)


def cmd_catchup(args) -> None:
    print(f"Writing reference outputs for step {args.step} into {WORKSPACE / 'outputs'} ...")
    env = {**os.environ, "MDLC_WORKSPACE": str(WORKSPACE)}
    result = subprocess.run([sys.executable, str(REFERENCE_DIR / "steps.py"), args.step], env=env)
    sys.exit(result.returncode)


def main(argv: list[str] | None = None) -> None:
    import signal
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # piping into `head` shouldn't raise
    p = argparse.ArgumentParser(prog="lab", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup").set_defaults(fn=cmd_setup)
    sub.add_parser("steps").set_defaults(fn=cmd_steps)
    s = sub.add_parser("step"); s.add_argument("step"); s.set_defaults(fn=cmd_step)
    s = sub.add_parser("check"); s.add_argument("step"); s.set_defaults(fn=cmd_check)
    s = sub.add_parser("score"); s.add_argument("file")
    s.add_argument("--truth", default=None); s.set_defaults(fn=cmd_score)
    s = sub.add_parser("submit"); s.add_argument("set", choices=["golden", "heldout"]); s.add_argument("file")
    s.add_argument("--name", required=True); s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_submit)
    sub.add_parser("log").set_defaults(fn=cmd_log)
    s = sub.add_parser("compare", help="paired comparison of two golden candidates")
    s.add_argument("a"); s.add_argument("b")
    s.add_argument("--on", choices=["dev", "golden", "heldout"], default="dev"); s.set_defaults(fn=cmd_compare)
    s = sub.add_parser("catchup"); s.add_argument("step"); s.set_defaults(fn=cmd_catchup)
    args = p.parse_args(argv)
    try:
        args.fn(args)
    except CheckFailed as e:
        print(f"{FAIL} {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
