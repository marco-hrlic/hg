# %% [markdown]
# # Lab 2: Open the box (DRAFT)
#
# > **Draft:** this is the former session 1 material (raw data → clean splits → golden set →
# > metrics). It is being reworked into lab 2: compressed to fit with the theory and retraining
# > blocks in a single 2h session.
#
# **Goal for today:** before writing a single line of model code, we'll build the thing that tells us
# whether a model is any good. By the end you will have:
#
# 1. a written problem spec,
# 2. a feel for how much *humans* disagree about sentiment,
# 3. a clean, deduplicated, versioned dataset with honest train/val/test splits,
# 4. a golden evaluation set with known coverage of hard cases,
# 5. two baselines and a scoreboard that every future model has to beat.
#
# Cells marked **TODO** raise `NotImplementedError` until you fill them in. Each has a check cell
# right after it. If you get stuck, the finished version is in `solutions/`.

# %%
import sys
from pathlib import Path

try:
    import mdlc
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path.cwd().parent))
    import mdlc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mdlc import LABELS
from mdlc import data, metrics, text
from mdlc.baselines import LexiconClassifier, MajorityClassifier

pd.set_option("display.max_colwidth", 120)
paths = data.build_all()  # downloads ~200 MB the first time, then takes a few seconds
paths

# %% [markdown]
# ---
# ## Part 1. Framing the problem (15 min)
#
# Someone asks you to "detect unhappy customers in reviews". Before any data or code, write down
# what that means. A **classifier** is a function `f(text) -> label` from a fixed set of labels.
# Everything else (the model, the training) is just a way to find a good `f`.
#
# Fill in this spec with your team. Keep it short, but be specific.
#
# | Question | Your answer |
# |---|---|
# | What decision does the prediction feed? | *e.g. route negative reviews to a support agent within 24h* |
# | Labels | negative / neutral / positive |
# | What exactly counts as each label? | *(we'll refine this in Part 2)* |
# | Which mistake is expensive? | *e.g. missing an angry customer (negative predicted as positive)* |
# | Which mistake is cheap? | |
# | How will we measure success? | *(we'll pick a metric in Part 5)* |
# | What does "good enough to ship" mean? | |
#
# **Our label source.** We have 1–5 star ratings, not sentiment labels. We'll use the mapping
# 1–2 ★ → negative, 3 ★ → neutral, 4–5 ★ → positive. That is a **proxy label**: cheap and plentiful,
# but not the same thing as "what the text says". Keep that in mind for the rest of the session.

# %% [markdown]
# ---
# ## Part 2. How much do humans agree? (25 min)
#
# Before trusting any label (ours or a model's), find out how consistent humans are on the task.
#
# **Exercise.** Label the 30 reviews below as negative / neutral / positive. Work on your own, don't
# discuss, and don't look at the stars. Put your labels in the shared sheet your instructor
# gives you (columns `item_id, label, rater`), or in the `MY_LABELS` dict further down.
#
# Draft guideline (deliberately incomplete):
# - **negative**: the reviewer is unhappy overall
# - **positive**: the reviewer is happy overall
# - **neutral**: neither

# %%
exercise = pd.read_csv(paths["exercise"])
for i, row in exercise.iterrows():
    print(f"[{row.item_id}] {row.text}\n")

# %% [markdown]
# ### Measuring agreement: Cohen's kappa
#
# "We agreed on 70% of items" sounds good, but if 70% of items are positive, two people who always
# say "positive" also agree 70% of the time. **Cohen's kappa** corrects for agreement by chance:
#
# $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
#
# - $p_o$ = observed agreement: the fraction of items where both raters chose the same label.
# - $p_e$ = chance agreement: for each label, (fraction rater A used it) × (fraction rater B used it),
#   summed over labels. That is how often two raters would agree if each picked labels at random
#   with their own habits.
#
# Rough reading: < 0.4 poor, 0.4–0.6 moderate, 0.6–0.8 substantial, > 0.8 excellent.
#
# **TODO 1:** implement it. Plain Python is fine.

# %%
def cohen_kappa(a: list[str], b: list[str]) -> float:
    """a[i] and b[i] are two raters' labels for item i."""
    ### BEGIN SOLUTION
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in set(a) | set(b))
    return (p_o - p_e) / (1 - p_e)
    ### END SOLUTION

# %%
# Check: two raters who agree on 4 of 6 items
a = ["positive", "positive", "negative", "neutral", "positive", "negative"]
b = ["positive", "neutral", "negative", "neutral", "positive", "positive"]
assert abs(cohen_kappa(a, b) - metrics.cohen_kappa(a, b)) < 1e-9, "doesn't match the reference"
print(f"kappa = {cohen_kappa(a, b):.3f}  (raw agreement {4/6:.0%})")

# %% [markdown]
# Now compare **your** labels with the star-derived labels. Fill in `MY_LABELS` (item_id → label).
# If it's empty, the cell compares the star labels with a simple word-counting rule instead,
# just so you can see the output format.

# %%
MY_LABELS: dict[str, str] = {
    # "G0123": "negative",
}

if MY_LABELS:
    mine = exercise[exercise.item_id.isin(MY_LABELS)]
    you = [MY_LABELS[i] for i in mine.item_id]
    print(f"you vs stars: kappa = {cohen_kappa(you, list(mine.star_label)):.2f} on {len(mine)} items")
    display(mine.assign(you=you).query("you != star_label")[["item_id", "stars", "star_label", "you", "text"]])
else:
    rule = LexiconClassifier().predict(exercise.text)
    print(f"(demo) word-count rule vs stars: kappa = {cohen_kappa(rule, list(exercise.star_label)):.2f}")

# %% [markdown]
# ### Whole-room agreement (instructor)
#
# Export the shared sheet to `data/golden/exercise_ratings.csv` with columns
# `item_id, label, rater`. **Fleiss' kappa** generalises Cohen's to many raters.

# %%
ratings_path = paths["exercise"].parent / "exercise_ratings.csv"
if ratings_path.exists():
    ratings = pd.read_csv(ratings_path)
    print(f"{ratings.rater.nunique()} raters, Fleiss' kappa = {metrics.fleiss_kappa(ratings):.2f}")
    spread = (ratings.groupby("item_id").label.value_counts().unstack(fill_value=0)
              .reindex(columns=list(LABELS), fill_value=0))
    spread["agreement"] = spread.max(axis=1) / spread.sum(axis=1)
    display(spread.sort_values("agreement").head(10).join(exercise.set_index("item_id")[["stars", "text"]]))
else:
    print("No room ratings yet:", ratings_path)

# %% [markdown]
# **Discuss (10 min):** look at the items with the lowest agreement.
# - Mixed reviews ("food great, service awful"): which label? Is "overall" enough of a rule?
# - A 3-star review that is clearly negative in tone: is the *star* wrong, or the *text*?
# - Sarcasm, comparisons, complaints about something other than the business?
#
# Then **rewrite the guideline** so that a new rater would label those items the same way you did.
# A guideline with examples of edge cases is the most valuable artifact of this whole session.
#
# > **Takeaway:** if humans agree at κ = 0.6, no model will be meaningfully "95% accurate" against
# > human judgement. The quality of the labels sets the ceiling.

# %% [markdown]
# ---
# ## Part 3. Data handling (35 min)
#
# You've been handed an export of 20,700 reviews. It has problems. Some are listed in the answer key
# your instructor holds; your job is to find them. **Look before you clean.**

# %%
raw = data.load_raw()  # every column loaded as text: we don't trust types yet
print(raw.shape)
raw.head()

# %%
raw.sample(5, random_state=1)

# %% [markdown]
# ### 3a. Labels you can't trust
# The `stars` column should hold integers from 1 to 5.

# %%
raw.stars.value_counts()

# %% [markdown]
# **TODO 2:** write `parse_stars`, which returns the integer for a valid rating and `None` for
# anything else. Don't try to rescue `"5 stars"` or `"four"`. Guessing creates labels nobody gave.

# %%
def parse_stars(value: str) -> int | None:
    ### BEGIN SOLUTION
    value = value.strip()
    if value in {"1", "2", "3", "4", "5"}:
        return int(value)
    return None
    ### END SOLUTION

# %%
raw["stars_int"] = raw.stars.map(parse_stars)
bad_label_ids = raw.loc[raw.stars_int.isna(), "review_id"]
data.score_findings(bad_label_ids, "invalid_label")

# %% [markdown]
# ### 3b. Text that isn't a review
# Look at the shortest texts.

# %%
raw.assign(n_chars=raw.text.str.len()).sort_values("n_chars").head(20)[["review_id", "stars", "text"]]

# %% [markdown]
# **TODO 3:** flag empty and placeholder texts. Decide on a rule (very short? a known list of
# placeholders?) and check how much it catches. Be careful not to throw away real short reviews
# like "Best tacos in town!".

# %%
def is_placeholder(t: str) -> bool:
    ### BEGIN SOLUTION
    c = text.normalize_for_dedup(t)
    return len(c) < 3 or c in {"n a", "test", "asdf", "null", "no comment", "review removed"}
    ### END SOLUTION

# %%
placeholder_ids = raw.loc[raw.text.map(is_placeholder), "review_id"]
data.score_findings(placeholder_ids, "empty_or_placeholder")

# %% [markdown]
# ### 3c. Markup and encoding junk
# Some rows went through a web form and came back with HTML. `text.clean_text` unescapes entities,
# removes tags and fixes the source's literal `\n` newline markers. It changes how text *looks*,
# never what it *says*.

# %%
has_html = raw.text.str.contains(r"<[a-z/][^>]*>|&[a-z#0-9]+;", regex=True)
example = raw.loc[has_html, "text"].iloc[0]
print("BEFORE:", example[:300], "\n\nAFTER: ", text.clean_text(example)[:300])
data.score_findings(raw.loc[has_html, "review_id"], "html_markup")

# %% [markdown]
# ### 3d. Duplicates
# Duplicates are the most common way to fool yourself in ML. If the same review lands in both
# train and test, the model is tested on something it memorised, and your score is inflated.
#
# **TODO 4:** find exact duplicates *after normalisation* (`text.normalize_for_dedup` lowercases
# and strips punctuation). Return the `review_id`s of every row except the first in each group,
# i.e. the rows you would drop.

# %%
def exact_duplicate_ids(df: pd.DataFrame) -> pd.Series:
    ### BEGIN SOLUTION
    norm = df.text.map(text.normalize_for_dedup)
    return df.loc[norm.duplicated(keep="first"), "review_id"]
    ### END SOLUTION

# %%
DUPLICATE_ISSUES = ["exact_duplicate", "conflicting_duplicate", "near_duplicate"]
dup_ids = exact_duplicate_ids(raw)
print(len(dup_ids), "rows to drop")
data.score_findings(dup_ids, DUPLICATE_ISSUES)

# %% [markdown]
# Some duplicates have **different labels**: the same text rated 5 ★ and 1 ★. Which one is right?
# You can't tell, so a sensible policy is to drop *all* copies and log them for review.
#
# Normalising catches copies that differ in case or punctuation, but not truncated or edited
# copies. For those we use **near-duplicate detection**: break each text into overlapping
# 3-word chunks and measure overlap (Jaccard similarity = shared chunks / all chunks).
# `text.find_near_duplicates` does this efficiently with MinHash, so it doesn't compare all
# 200 million pairs.

# %%
pairs = text.find_near_duplicates(raw.text.tolist(), threshold=0.7)
print(len(pairs), "near-duplicate pairs")
i, j, sim = next(p for p in pairs if p[2] < 0.95)
print(f"similarity {sim:.2f}\nA: {raw.text[i][:200]}\nB: {raw.text[j][:200]}")

# %%
near_ids = {raw.review_id[j] for i, j, _ in pairs}
data.score_findings(near_ids, DUPLICATE_ISSUES)

# %% [markdown]
# ### 3e. Problems you can't see row by row
#
# Three problems only show up when you look at the dataset as a whole.

# %%
# 1. Column check: could this column be known at prediction time?
raw[["stars", "stars_display"]].drop_duplicates().head(8)

# %%
# 2. Is the file ordered by anything?
order = pd.to_numeric(raw.stars, errors="coerce")
plt.figure(figsize=(8, 2))
plt.plot(order.to_numpy(), ".", markersize=1)
plt.xlabel("row number"); plt.ylabel("stars"); plt.title("stars by position in the file")
plt.show()

# %%
# 3. Class balance
raw.stars_int.map(lambda s: data.stars_to_sentiment(s) if s == s else None).value_counts(normalize=True)

# %% [markdown]
# - `stars_display` is the label in disguise. A model given this column would score ~100% in testing
#   and be useless in production, where the column doesn't exist. This is called **target leakage**.
# - The file is **sorted by rating**. "First 80% train, last 20% test" would give a test set with
#   almost only 5-star reviews.
# - **Imbalance:** about 63% positive. A model that always says "positive" is 63% accurate.
#   Remember that number.

# %% [markdown]
# ### 3f. One cleaning function, top to bottom
# Put every decision in one place, and make it report what it dropped and why. In real projects this
# function lives in version control and runs exactly the same way every time.

# %%
def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = raw.drop(columns=["stars_display"]).copy()  # leaky column
    df["stars"] = df.stars.map(parse_stars)
    df["text"] = df.text.map(text.clean_text)
    log = {"start": len(df)}

    df = df[df.stars.notna()]; log["invalid_label"] = log["start"] - len(df)
    n = len(df); df = df[~df.text.map(is_placeholder)]; log["placeholder"] = n - len(df)

    df["stars"] = df.stars.astype(int)
    df["label"] = df.stars.map(data.stars_to_sentiment)
    df["norm"] = df.text.map(text.normalize_for_dedup)

    conflicting = df.groupby("norm").label.transform("nunique") > 1
    n = len(df); df = df[~conflicting]; log["conflicting_duplicate"] = n - len(df)
    n = len(df); df = df.drop_duplicates("norm"); log["exact_duplicate"] = n - len(df)

    df = df.reset_index(drop=True)
    near = text.find_near_duplicates(df.text.tolist(), threshold=0.7)
    drop = {j for _, j, _ in near}
    n = len(df); df = df.drop(index=sorted(drop)); log["near_duplicate"] = n - len(df)

    log["final"] = len(df)
    return df.drop(columns=["norm"]).reset_index(drop=True), pd.Series(log)


clean, log = clean_dataset(raw)
log

# %% [markdown]
# Notice what we did **not** catch: the answer key has ~600 rows whose stars were flipped
# (a 5 ★ review stored as 1 ★). They look perfectly valid row by row. We'll find them in
# session 3, using a trained model to point at suspicious labels.

# %% [markdown]
# ### 3g. Splitting
# - **train**: the model learns from it.
# - **validation**: we compare models and tune settings on it, as often as we like.
# - **test**: we look at it *once*, at the end. Every peek turns it into a second validation set.
#
# A **stratified** split keeps the label mix the same in every split.
#
# **TODO 5:** split `df` into train/val/test (80/10/10) so that each label is split in the same
# proportions. Shuffle with the given `seed`. Hint: `groupby("label")` then split each group.

# %%
def stratified_split(df: pd.DataFrame, fractions=(0.8, 0.1, 0.1), seed: int = 0):
    ### BEGIN SOLUTION
    parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("label"):
        group = group.sample(frac=1, random_state=seed)
        n_train = int(round(len(group) * fractions[0]))
        n_val = int(round(len(group) * fractions[1]))
        parts["train"].append(group.iloc[:n_train])
        parts["val"].append(group.iloc[n_train : n_train + n_val])
        parts["test"].append(group.iloc[n_train + n_val :])
    return tuple(pd.concat(p).sample(frac=1, random_state=seed).reset_index(drop=True)
                 for p in parts.values())
    ### END SOLUTION

# %%
train, val, test = stratified_split(clean, seed=42)
naive_test = clean.iloc[int(len(clean) * 0.9):]  # what "just take the end of the file" would give
mix = pd.DataFrame({
    "train": train.label.value_counts(normalize=True),
    "val": val.label.value_counts(normalize=True),
    "test": test.label.value_counts(normalize=True),
    "naive tail split": naive_test.label.value_counts(normalize=True),
}).round(3)
display(mix)
assert len(train) + len(val) + len(test) == len(clean)
assert not set(train.review_id) & set(test.review_id), "a review is in both train and test"
assert (mix[["train", "val", "test"]].max(axis=1) - mix[["train", "val", "test"]].min(axis=1)).max() < 0.01

# %% [markdown]
# Leakage check: is any test review a near-copy of a training review?

# %%
both = pd.concat([train.assign(split="train"), test.assign(split="test")], ignore_index=True)
cross = [(i, j) for i, j, _ in text.find_near_duplicates(both.text.tolist(), 0.7)
         if both.split[i] != both.split[j]]
print(len(cross), "train/test near-duplicate pairs")

# %% [markdown]
# ### 3h. Version it
# Save the splits with a manifest: row counts, label mix, and a SHA-256 hash of every file.
# Later sessions load from here, and `load_splits` warns if a file changed. A result you can't tie
# to an exact dataset version is a result you can't reproduce.

# %%
manifest = data.save_versioned({"train": train, "val": val, "test": test},
                               data.DATA_DIR / "splits" / "v1",
                               note="session 1: cleaned, deduplicated, stratified 80/10/10, seed 42")
manifest

# %% [markdown]
# ---
# ## Part 4. The golden dataset (25 min)
#
# Our splits use **proxy labels** (stars). Part 2 showed they don't always match what the text says,
# and somewhere in there are ~600 flipped labels. Measuring a model against noisy labels gives a
# noisy score.
#
# A **golden set** is a small evaluation set where every label was checked by people, following a
# written guideline. Properties of a good one:
#
# | Property | Why |
# |---|---|
# | Labelled by ≥ 2 people independently, disagreements adjudicated | one person's habits aren't ground truth |
# | Follows the written guideline from Part 2 | so labels mean the same thing next quarter |
# | **Covers hard cases on purpose** (slices) | random samples are mostly easy; you need enough hard cases to measure them |
# | Disjoint from training data, checked for near-duplicates | otherwise it measures memory |
# | Frozen and versioned; changes are deliberate and logged | so scores stay comparable over time |
# | Small enough to label well (hundreds, not thousands) | quality beats size here |
#
# We prepared 400 candidates from Yelp's *official test split*, which our training pool never
# touches. Each one is tagged with **slices**: cheap rules that flag likely hard cases.

# %%
cands = data.load_golden_candidates()
coverage = cands.explode("slices").slices.value_counts()
display(coverage.to_frame("items"))
cands.sample(5, random_state=0)[["item_id", "stars", "slices", "text"]]

# %% [markdown]
# **Labelling process (teams of 2–3):**
# 1. Each team takes a block of item_ids from `labelling_sheet.csv`.
# 2. Each person labels **independently**, following the revised guideline. Stars are hidden on
#    purpose.
# 3. Compare. Where you disagree, discuss and record an adjudicated label plus a short note.
#    If you still can't agree, mark it `ambiguous`: it leaves the golden set but goes into the
#    guideline as an example.
# 4. The instructor merges the results into `data/golden/golden_labels.csv` (`item_id, label`).
#
# Until those labels exist, the cell below falls back to star labels and says so loudly.

# %%
golden_path = paths["candidates"].parent / "golden_labels.csv"
if golden_path.exists():
    gl = pd.read_csv(golden_path)
    golden = cands.merge(gl[["item_id", "label"]], on="item_id")
    golden = golden[golden.label.isin(LABELS)].reset_index(drop=True)
    print(f"golden set: {len(golden)} human-labelled items")
    print(f"human label vs star label agreement: {(golden.label == golden.star_label).mean():.0%}")
else:
    golden = cands.assign(label=cands.star_label)
    print("⚠ No human labels yet: using STAR labels as a stand-in. Scores below are provisional.")

# %% [markdown]
# ---
# ## Part 5. Metrics and baselines (20 min)
#
# A **baseline** is the simplest thing that could work. If a fancy model can't beat it, the fancy
# model isn't worth its cost.
#
# - **Majority:** always predict the most common training label.
# - **Lexicon:** count positive words minus negative words from a hand-written list. No learning.

# %%
models = {
    "majority": MajorityClassifier().fit(train.text, train.label),
    "lexicon": LexiconClassifier(),
    "lexicon + negation": LexiconClassifier(handle_negation=True),
}
for name, m in models.items():
    pred = m.predict(val.text)
    print(f"{name:20s} val accuracy = {metrics.accuracy(val.label, pred):.3f}")

# %% [markdown]
# The majority baseline scores well on accuracy while being useless: it never finds a single
# unhappy customer. Accuracy hides that. We need a metric per class:
#
# - **precision** for class *c*: of the items we *called* c, what fraction really were c?
# - **recall** for class *c*: of the items that really *were* c, what fraction did we find?
# - **F1**: the harmonic mean of the two, $2PR/(P+R)$. It's high only if both are high.
# - **macro-F1**: the plain average of F1 over classes, so the rare "neutral" counts as much as
#   "positive".
#
# These all come from the **confusion matrix**: rows are true labels, columns are predictions.

# %%
cm = metrics.confusion_matrix(val.label, models["lexicon"].predict(val.text))
pd.DataFrame(cm, index=[f"true {l}" for l in LABELS], columns=[f"pred {l}" for l in LABELS])

# %% [markdown]
# **TODO 6:** compute macro-F1 from a confusion matrix. Precision for class j is
# `cm[j, j] / column j sum`, and recall is `cm[j, j] / row j sum`. Treat 0/0 as 0.

# %%
def macro_f1_from_cm(cm: np.ndarray) -> float:
    ### BEGIN SOLUTION
    f1s = []
    for j in range(len(cm)):
        tp = cm[j, j]
        precision = tp / cm[:, j].sum() if cm[:, j].sum() else 0.0
        recall = tp / cm[j, :].sum() if cm[j, :].sum() else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s))
    ### END SOLUTION

# %%
for name, m in models.items():
    pred = m.predict(val.text)
    mine = macro_f1_from_cm(metrics.confusion_matrix(val.label, pred))
    assert abs(mine - metrics.macro_f1(val.label, pred)) < 1e-9, name
print("macro-F1 matches the reference ✓")

# %% [markdown]
# ### The scoreboard
# Every model from now on is reported on the same evaluation sets, with the same metrics, overall
# **and per slice**.

# %%
rows = []
for name, m in models.items():
    for split_name, df in [("val (stars)", val), ("golden", golden)]:
        pred = m.predict(df.text)
        rows.append({"model": name, "eval set": split_name,
                     "accuracy": metrics.accuracy(df.label, pred),
                     "macro_f1": metrics.macro_f1(df.label, pred)})
scoreboard = pd.DataFrame(rows).pivot(index="model", columns="eval set").round(3)
scoreboard

# %%
g = golden.assign(pred=models["lexicon"].predict(golden.text))
by_slice = metrics.slice_report(g, "label", "pred")
g2 = golden.assign(pred=models["lexicon + negation"].predict(golden.text))
by_slice.join(metrics.slice_report(g2, "label", "pred"), rsuffix=" (+negation)")

# %%
results_dir = data.DATA_DIR / "results"
results_dir.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(results_dir / "session1_baselines.csv", index=False)
print("saved", results_dir / "session1_baselines.csv")

# %% [markdown]
# ---
# ## Wrap-up
#
# What we have now:
# - a spec and a guideline with edge cases,
# - `splits/v1`: clean, deduplicated, stratified, versioned,
# - a golden set with known slice coverage,
# - a scoreboard: majority and lexicon baselines on val and golden, overall and per slice.
#
# What we know is still wrong: ~600 flipped labels hiding in the training data.
#
# **Next session:** we replace the hand-written word list with weights the computer *learns*
# from the training data, and work out the maths of how it learns them.
