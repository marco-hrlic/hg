"""Download Yelp reviews and build every dataset the lab uses.

Source: Yelp Review Full (Zhang, Zhao & LeCun, 2015), 1-5 stars, from the
fast.ai public S3 mirror. The source data is not redistributed in this repo;
`build_all()` downloads it and rebuilds everything from a fixed seed, so every
participant gets byte-identical files.

Files produced under the data directory:

    raw/reviews_raw.csv         training pool with PLANTED data problems (session 1 hunts them)
    raw/answer_key.csv          instructor-only: one row per planted problem
    golden/candidates.csv       ~400 reviews from the official *test* split, slice-tagged,
                                for golden-set curation
    prepared/                   lab 1's ready-made development data: train/dev (reference_clean + stratified split)
    golden/exercise_30.csv      30 of the candidates for the labelling-agreement exercise
    golden/labelling_sheet.csv  blank sheet for collecting human labels
    heldout/heldout.csv         5,000 clean reviews from the test split, production label mix. Locked:
                                looked at once at the end of each lab.

Development data (train, dev) and evaluation data (golden, held-out) come from different
source files and are deduplicated against each other.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABELS
from .slices import NEGATIVE_HINTS, POSITIVE_HINTS, tag_slices
from .text import clean_text, find_near_duplicates, normalize_for_dedup

YELP_URL = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_full_csv.tgz"
SEED = 20261113

LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MDLC_DATA", LAB_ROOT / "data"))

# Real review streams are skewed towards positive. Proportions per star rating.
STAR_MIX = {1: 0.15, 2: 0.10, 3: 0.12, 4: 0.25, 5: 0.38}
N_BASE = 20_000
MAX_WORDS = 350  # keep CPU training fast and reviews readable
N_MORE = 82_000  # lab 2's "more data" pool: brings training to ~100k reviews


def stars_to_sentiment(stars: int) -> str:
    """The label mapping used throughout the lab: 1-2 negative, 3 neutral, 4-5 positive."""
    return "negative" if stars <= 2 else "neutral" if stars == 3 else "positive"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_yelp(data_dir: Path = DATA_DIR) -> Path:
    """Fetch and unpack the Yelp archive (~200 MB) once. Returns the folder with train/test CSVs."""
    src = Path(data_dir) / "source"
    out = src / "yelp_review_full_csv"
    if (out / "train.csv").exists() and (out / "test.csv").exists():
        return out
    src.mkdir(parents=True, exist_ok=True)
    tgz = src / "yelp_review_full_csv.tgz"
    if not tgz.exists():
        print(f"Downloading {YELP_URL} (~200 MB) ...")
        urllib.request.urlretrieve(YELP_URL, tgz)
    with tarfile.open(tgz) as tf:
        members = [m for m in tf.getmembers() if m.name.endswith((".csv", "readme.txt"))]
        tf.extractall(src, members=members, filter="data")
    return out


def _read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=["stars", "text"], dtype={"stars": int, "text": str})
    df["n_words"] = df["text"].str.split().str.len()
    return df[(df["n_words"] >= 3) & (df["n_words"] <= MAX_WORDS)].drop(columns="n_words")


def _sample_mix(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for star, frac in STAR_MIX.items():
        pool = df[df["stars"] == star]
        k = int(round(n * frac))
        parts.append(pool.iloc[rng.choice(len(pool), size=k, replace=False)])
    return pd.concat(parts).sample(frac=1, random_state=int(rng.integers(1 << 31)))


# ---------------------------------------------------------------------------
# Planting data problems
# ---------------------------------------------------------------------------

def _stars_display(stars) -> str:
    try:
        s = int(stars)
    except (TypeError, ValueError):
        return ""
    return "★" * s + "☆" * (5 - s) if 1 <= s <= 5 else ""


def _add_html(text: str, rng: np.random.Generator) -> str:
    t = text.replace("\\n", "<br />")
    t = t.replace("&", "&amp;").replace("'", "&#39;")
    if rng.random() < 0.5 or t == text:
        t = f"<p>{t}</p>"
    if "<br />" not in t:
        sentences = t.split(". ")
        if len(sentences) > 1:
            cut = int(rng.integers(1, len(sentences)))
            t = ". ".join(sentences[:cut]) + ".<br /><br />" + ". ".join(sentences[cut:])
    return t


def _near_variant(text: str, rng: np.random.Generator) -> str:
    kind = rng.integers(5)
    words = text.split()
    if kind == 0:
        return text.lower()
    if kind == 1:
        return "  ".join(words) + "  "
    if kind == 2:
        return text.replace(",", "").replace("!", ".").replace("'", "")
    if kind == 3:
        return " ".join(words[: max(8, int(len(words) * 0.85))])
    return text + rng.choice([" Update: still true.", " Edit: went back, same experience.", " 10/10 accurate."])


def _flip_stars(stars: int, rng: np.random.Generator) -> int:
    if stars == 3:
        return int(rng.choice([1, 5]))
    return 6 - stars  # 1<->5, 2<->4


PLACEHOLDERS = ["", "   ", "N/A", ".", "test", "asdf", "-", "null", "no comment", "(review removed)"]
INVALID_STARS = ["0", "6", "-1", "", "5 stars", "four", "10", "3.5"]


def build_raw(source_train: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample the training pool and plant problems. Returns (raw_df, answer_key)."""
    rng = np.random.default_rng(seed)
    base = _sample_mix(source_train, N_BASE, rng).reset_index(drop=True)
    n_extra = 300 + 60 + 300 + 40
    # Random-looking ids, so row id carries no information about the label.
    ids = [f"R{100_000 + p:06d}" for p in rng.permutation(N_BASE + n_extra)]
    base["review_id"] = ids[:N_BASE]
    base["stars"] = base["stars"].astype(object)
    base["original_stars"] = base["stars"]
    base["issue"] = ""
    base["related"] = ""

    idx = rng.permutation(len(base))
    cuts = np.cumsum([1000, 60, 600])  # html, invalid label, label noise (disjoint rows)
    html_rows, invalid_rows, noise_rows = idx[: cuts[0]], idx[cuts[0] : cuts[1]], idx[cuts[1] : cuts[2]]
    clean_pool = idx[cuts[2] :]  # untouched rows, safe to copy from

    for i in html_rows:
        base.at[i, "text"] = _add_html(base.at[i, "text"], rng)
        base.at[i, "issue"] = "html_markup"
    for i in invalid_rows:
        base.at[i, "stars"] = str(rng.choice(INVALID_STARS))
        base.at[i, "issue"] = "invalid_label"
    for i in noise_rows:
        base.at[i, "stars"] = _flip_stars(int(base.at[i, "stars"]), rng)
        base.at[i, "issue"] = "label_noise"

    picks = rng.choice(clean_pool, size=660, replace=False)
    extra = []
    for i in picks[:300]:
        r = base.loc[i]
        extra.append({"text": r.text, "stars": r.stars, "original_stars": r.stars,
                      "issue": "exact_duplicate", "related": r.review_id})
    for i in picks[300:360]:
        r = base.loc[i]
        other = [s for s in range(1, 6) if stars_to_sentiment(s) != stars_to_sentiment(int(r.stars))]
        extra.append({"text": r.text, "stars": int(rng.choice(other)), "original_stars": r.stars,
                      "issue": "conflicting_duplicate", "related": r.review_id})
    for i in picks[360:660]:
        r = base.loc[i]
        extra.append({"text": _near_variant(r.text, rng), "stars": r.stars, "original_stars": r.stars,
                      "issue": "near_duplicate", "related": r.review_id})
    for _ in range(40):
        s = int(rng.integers(1, 6))
        extra.append({"text": str(rng.choice(PLACEHOLDERS)), "stars": s, "original_stars": s,
                      "issue": "empty_or_placeholder", "related": ""})
    extra = pd.DataFrame(extra)
    extra["review_id"] = ids[N_BASE:]

    df = pd.concat([base, extra], ignore_index=True)

    # Export quirk: the file comes out sorted by rating (a naive head/tail split will be badly skewed).
    df["_sort"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    df = df.sort_values("_sort", kind="stable", na_position="last").drop(columns="_sort")
    # Export quirk: a display column that is just the label in another form (target leakage).
    df["stars_display"] = df["stars"].map(_stars_display)
    df = df.reset_index(drop=True)

    key = df.loc[df["issue"] != "", ["review_id", "issue", "related", "original_stars", "stars"]]
    raw = df[["review_id", "stars", "stars_display", "text"]].copy()
    return raw, key.reset_index(drop=True)


GLOBAL_ISSUES = {
    "sorted_by_label": "The file is sorted by star rating, so taking the first 80% as train skews every split.",
    "leaky_column": "stars_display is the label written as stars. It must never be used as a feature.",
    "class_imbalance": "About 63% positive, 25% negative, 12% neutral. Accuracy alone is misleading.",
    "escaped_characters": "The source escapes newlines as a literal backslash-n and quotes as backslash-quote.",
}


def score_findings(found_ids, issue: str | list[str], data_dir: Path = DATA_DIR) -> dict:
    """Compare the review_ids you flagged against the answer key.

    `issue` is one planted issue name, or a list of them scored together (useful
    for the duplicate family, where one detector catches several kinds).
    For duplicates you only need to flag one copy of each pair: flagging either
    the planted copy or the original it was copied from counts as caught.
    """
    key = pd.read_csv(Path(data_dir) / "raw" / "answer_key.csv", dtype=str, keep_default_na=False)
    issues = [issue] if isinstance(issue, str) else list(issue)
    unknown = set(issues) - set(key["issue"])
    if unknown:
        raise ValueError(f"unknown issue(s) {sorted(unknown)}; known: {sorted(key['issue'].unique())}")
    found = set(found_ids)
    rows = key.loc[key["issue"].isin(issues)]
    is_dup = rows["issue"].str.endswith("duplicate")
    hit = rows["review_id"].isin(found) | (is_dup & rows["related"].isin(found))
    allowed = set(rows["review_id"]) | set(rows.loc[is_dup, "related"])
    out = {
        "issue": issue if isinstance(issue, str) else " + ".join(issues),
        "planted": len(rows),
        "you_flagged": len(found),
        "caught": int(hit.sum()),
        "missed": int((~hit).sum()),
        "flagged_but_not_planted": len(found - allowed),
    }
    if len(issues) > 1:
        out["caught_by_issue"] = {k: f"{int(hit[rows['issue'] == k].sum())}/{int((rows['issue'] == k).sum())}"
                                  for k in issues}
    return out


# ---------------------------------------------------------------------------
# Golden candidates, labelling exercise, held-out test
# ---------------------------------------------------------------------------

GOLDEN_QUOTAS = {  # minimum items per slice in the candidate pool
    "negation": 60, "contrast": 70, "short": 40, "long": 25, "comparison": 20,
    "shouting": 20, "sarcasm_cue": 15, "mixed_signals": 50, "three_star": 50,
}
N_GOLDEN = 400
GOLDEN_MAX_WORDS = 150  # people have to read these quickly
GOLDEN_LONG_MAX_WORDS = 260


def _select_golden(test: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_words = test["text"].str.split().str.len()
    pool = test[n_words <= GOLDEN_LONG_MAX_WORDS].copy()
    pool = pool.iloc[rng.permutation(len(pool))[:20_000]]
    pool["slices"] = pool["text"].map(tag_slices)
    chosen: list[int] = []
    taken: set[int] = set()
    for name, quota in GOLDEN_QUOTAS.items():
        if name == "three_star":
            mask = pool["stars"] == 3
        else:
            mask = pool["slices"].map(lambda s, n=name: n in s)
        have = sum(1 for i in chosen if (pool.at[i, "stars"] == 3 if name == "three_star"
                                         else name in pool.at[i, "slices"]))
        if name != "long":  # only the "long" slice may exceed the reading-speed cap
            mask &= pool["text"].str.split().str.len() <= GOLDEN_MAX_WORDS
        cands = [i for i in pool.index[mask] if i not in taken]
        for i in cands[: max(0, quota - have)]:
            chosen.append(i)
            taken.add(i)
    short_enough = pool["text"].str.split().str.len() <= GOLDEN_MAX_WORDS
    rest = [i for i in pool.index[short_enough] if i not in taken]
    chosen += rest[: N_GOLDEN - len(chosen)]
    out = pool.loc[chosen].sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    out.insert(0, "item_id", [f"G{n:04d}" for n in range(1, len(out) + 1)])
    out["star_label"] = out["stars"].map(stars_to_sentiment)
    out["text"] = out["text"].map(clean_text)
    out["slices"] = out["slices"].map(";".join)
    return out[["item_id", "text", "stars", "star_label", "slices"]]


def _select_exercise(golden: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """8 clear-cut items and 22 genuinely debatable ones."""
    sl = golden["slices"]
    clear = golden[golden["stars"].isin([1, 5]) & ~sl.str.contains("contrast|mixed_signals|sarcasm")]
    hard = golden[(golden["stars"].isin([2, 3, 4])) & sl.str.contains("contrast|mixed_signals|negation")]
    pick = pd.concat([
        clear.iloc[rng.permutation(len(clear))[:8]],
        hard.iloc[rng.permutation(len(hard))[:22]],
    ])
    return pick.sample(frac=1, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reference cleaning pipeline (produces lab 1's prepared data; lab 2A rebuilds it by hand)
# ---------------------------------------------------------------------------

_PLACEHOLDER_NORMS = {"n a", "test", "asdf", "null", "no comment", "review removed"}


def parse_stars(value: object) -> int | None:
    """A valid rating is exactly one of "1".."5". Anything else is unusable, not guessed at."""
    v = str(value).strip()
    return int(v) if v in {"1", "2", "3", "4", "5"} else None


def is_placeholder(t: object) -> bool:
    c = normalize_for_dedup(t)
    return len(c) < 3 or c in _PLACEHOLDER_NORMS


def reference_clean(raw: pd.DataFrame, near_threshold: float = 0.7) -> tuple[pd.DataFrame, pd.Series]:
    """Every cleaning decision in one place. Returns (clean_df, log of rows dropped per reason)."""
    df = raw.drop(columns=["stars_display"]).copy()  # leaky column
    df["stars"] = df["stars"].map(parse_stars)
    df["text"] = df["text"].map(clean_text)
    log = {"start": len(df)}

    df = df[df["stars"].notna()]
    log["invalid_label"] = log["start"] - len(df)
    n = len(df)
    df = df[~df["text"].map(is_placeholder)]
    log["placeholder"] = n - len(df)

    df["stars"] = df["stars"].astype(int)
    df["label"] = df["stars"].map(stars_to_sentiment)
    df["norm"] = df["text"].map(normalize_for_dedup)
    conflicting = df.groupby("norm")["label"].transform("nunique") > 1
    n = len(df)
    df = df[~conflicting]
    log["conflicting_duplicate"] = n - len(df)
    n = len(df)
    df = df.drop_duplicates("norm")
    log["exact_duplicate"] = n - len(df)

    df = df.reset_index(drop=True)
    near = find_near_duplicates(df["text"].tolist(), threshold=near_threshold)
    n = len(df)
    df = df.drop(index=sorted({j for _, j, _ in near}))
    log["near_duplicate"] = n - len(df)
    log["final"] = len(df)
    return df.drop(columns=["norm"]).reset_index(drop=True), pd.Series(log)


def stratified_split(df: pd.DataFrame, fractions=(0.9, 0.1), seed: int = 42,
                     label_col: str = "label") -> tuple[pd.DataFrame, ...]:
    """Shuffle each label group separately and cut it by `fractions`, so every part has the same label mix.

    The last part takes whatever is left after rounding. Default: train 90% / dev 10%.
    """
    parts: list[list[pd.DataFrame]] = [[] for _ in fractions]
    for _, group in df.groupby(label_col):
        group = group.sample(frac=1, random_state=seed)
        start = 0
        for k, frac in enumerate(fractions):
            end = len(group) if k == len(fractions) - 1 else start + int(round(len(group) * frac))
            parts[k].append(group.iloc[start:end])
            start = end
    return tuple(pd.concat(p).sample(frac=1, random_state=seed).reset_index(drop=True) for p in parts)


def build_all(data_dir: Path = DATA_DIR, seed: int = SEED, force: bool = False) -> dict[str, Path]:
    """Build every lab dataset. Safe to call repeatedly: skips work if the files already exist."""
    data_dir = Path(data_dir)
    paths = {
        "raw": data_dir / "raw" / "reviews_raw.csv",
        "answer_key": data_dir / "raw" / "answer_key.csv",
        "candidates": data_dir / "golden" / "candidates.csv",
        "exercise": data_dir / "golden" / "exercise_30.csv",
        "sheet": data_dir / "golden" / "labelling_sheet.csv",
        "heldout": data_dir / "heldout" / "heldout.csv",
        "prepared": data_dir / "prepared" / "manifest.json",
        "more": data_dir / "extra" / "more_reviews.csv",
    }
    if not force and all(p.exists() for p in paths.values()):
        return paths
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)

    src = download_yelp(data_dir)
    rng = np.random.default_rng(seed + 1)
    train = _read_source(src / "train.csv")
    test = _read_source(src / "test.csv")

    raw, key = build_raw(train, seed)
    raw.to_csv(paths["raw"], index=False)
    key.to_csv(paths["answer_key"], index=False)

    # Nothing from the training pool may appear (even approximately) in evaluation data.
    train_norm = set(raw["text"].map(normalize_for_dedup))
    test = test[~test["text"].map(normalize_for_dedup).isin(train_norm)]

    golden = _select_golden(test, rng)
    golden.to_csv(paths["candidates"], index=False)
    exercise = _select_exercise(golden, rng)
    exercise.to_csv(paths["exercise"], index=False)
    sheet = golden[["item_id", "text"]].assign(label="", rater="", notes="")
    sheet.to_csv(paths["sheet"], index=False)

    golden_norm = set(golden["text"].map(normalize_for_dedup))
    rest = test[~test["text"].map(normalize_for_dedup).isin(golden_norm)]
    heldout = _sample_mix(rest, 5000, rng).reset_index(drop=True)
    heldout.insert(0, "review_id", [f"H{n:05d}" for n in range(1, len(heldout) + 1)])
    heldout["text"] = heldout["text"].map(clean_text)
    heldout["label"] = heldout["stars"].map(stars_to_sentiment)
    heldout.to_csv(paths["heldout"], index=False)

    # Lab 2's "more data" pool: a second, larger export, already cleaned and star-labelled,
    # disjoint (after normalisation) from the raw pool and from all evaluation data.
    taken = train_norm | golden_norm | set(heldout["text"].map(normalize_for_dedup))
    pool = train.copy()
    pool["text"] = pool["text"].map(clean_text)
    pool["norm"] = pool["text"].map(normalize_for_dedup)
    pool = pool[~pool["norm"].isin(taken)].drop_duplicates("norm")
    more = _sample_mix(pool, N_MORE, np.random.default_rng(seed + 2)).reset_index(drop=True)
    more.insert(0, "review_id", [f"M{n:06d}" for n in range(1, len(more) + 1)])
    more["label"] = more["stars"].map(stars_to_sentiment)
    more[["review_id", "text", "label"]].to_csv(paths["more"], index=False)

    # Lab 1's ready-made development data: the reference pipeline applied to the raw pool.
    # It is clean in every way you can check row by row, but the ~600 flipped labels are still in it.
    # Evaluation data (golden, held-out) comes from the separate Yelp test file, never from this pool.
    clean, _ = reference_clean(raw)
    train_p, dev_p = stratified_split(clean, (0.9, 0.1), seed=42)
    save_versioned({"train": train_p, "dev": dev_p}, paths["prepared"].parent,
                   note="reference_clean() on raw/reviews_raw.csv, stratified train/dev 90/10, seed 42")

    manifest = {name: {"path": str(p.relative_to(data_dir)), "sha256": file_sha256(p),
                       "rows": int(len(pd.read_csv(p)))} for name, p in paths.items() if p.suffix == ".csv"}
    (data_dir / "manifest.json").write_text(json.dumps({"seed": seed, "files": manifest}, indent=2))
    return paths


# ---------------------------------------------------------------------------
# Loading, splitting, versioning
# ---------------------------------------------------------------------------

def load_prepared(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lab 1's ready-made development data: (train, dev)."""
    s = load_splits(Path(data_dir) / "prepared")
    return s["train"], s["dev"]


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The dirty training pool, read the way a careless first attempt would: everything as text."""
    return pd.read_csv(Path(data_dir) / "raw" / "reviews_raw.csv", dtype=str, keep_default_na=False)


def load_golden_candidates(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "golden" / "candidates.csv", keep_default_na=False)
    df["slices"] = df["slices"].map(lambda s: [x for x in s.split(";") if x])
    return df


GOLDEN_LABELS_DIR = LAB_ROOT / "golden_labels"


def load_golden(data_dir: Path = DATA_DIR, allow_draft: bool = True) -> pd.DataFrame:
    """The golden set: candidates joined with their checked labels.

    Uses golden_labels/golden_labels.csv (human-reviewed, columns item_id,label) when it exists.
    Otherwise, with allow_draft, falls back to the LLM draft labels the two raters agreed on,
    and warns that they have not been reviewed yet. Items labelled `ambiguous` are excluded.
    """
    cands = load_golden_candidates(data_dir)
    final = GOLDEN_LABELS_DIR / "golden_labels.csv"
    if final.exists():
        labels = pd.read_csv(final, keep_default_na=False)[["item_id", "label"]]
        source = "human-reviewed"
    elif allow_draft and (GOLDEN_LABELS_DIR / "draft_labels.csv").exists():
        d = pd.read_csv(GOLDEN_LABELS_DIR / "draft_labels.csv", keep_default_na=False)
        labels = d.loc[d["status"].str.startswith("agreed"), ["item_id", "draft_label"]].rename(columns={"draft_label": "label"})
        source = "LLM DRAFT (two raters agreed, not yet human-reviewed)"
        warnings.warn("Golden labels are unreviewed LLM drafts. Review golden_labels/ before relying on them.")
    else:
        raise FileNotFoundError(f"no golden labels in {GOLDEN_LABELS_DIR}")
    df = cands.merge(labels, on="item_id")
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)
    df.attrs["label_source"] = source
    return df


def load_heldout(data_dir: Path = DATA_DIR, *, i_am_done_iterating: bool = False) -> pd.DataFrame:
    """The locked test set. Look at it once, at the end, after every decision is made."""
    if not i_am_done_iterating:
        raise PermissionError(
            "The held-out set is locked. Every time you look at it and change something, it stops "
            "being an honest estimate. Pass i_am_done_iterating=True when you mean it."
        )
    df = pd.read_csv(Path(data_dir) / "heldout" / "heldout.csv")
    df["slices"] = df["text"].map(tag_slices)
    return df


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_versioned(splits: dict[str, pd.DataFrame], out_dir: Path, note: str = "") -> dict:
    """Write each split as CSV plus a manifest (hash, row count, label mix) so results are reproducible."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"note": note, "splits": {}}
    for name, df in splits.items():
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        entry = {"rows": len(df), "sha256": file_sha256(p)}
        if "label" in df:
            entry["label_counts"] = {k: int(v) for k, v in df["label"].value_counts().items()}
        manifest["splits"][name] = entry
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_splits(split_dir: Path) -> dict[str, pd.DataFrame]:
    """Load splits written by save_versioned and verify they have not changed since."""
    split_dir = Path(split_dir)
    manifest = json.loads((split_dir / "manifest.json").read_text())
    out = {}
    for name, entry in manifest["splits"].items():
        p = split_dir / f"{name}.csv"
        if file_sha256(p) != entry["sha256"]:
            warnings.warn(f"{p} changed since it was versioned; results may not be comparable.")
        out[name] = pd.read_csv(p, keep_default_na=False)
    return out


__all__ = [
    "LABELS", "build_all", "download_yelp", "find_near_duplicates", "load_golden_candidates",
    "load_golden", "load_heldout", "load_prepared", "load_raw", "reference_clean", "stratified_split", "load_splits", "save_versioned", "score_findings", "stars_to_sentiment",
    "NEGATIVE_HINTS", "POSITIVE_HINTS",
]
