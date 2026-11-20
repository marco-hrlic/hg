# %% [markdown]
# # Lab 2: Open the box
#
# In lab 1 you trained a classifier on data someone had prepared for you, and it worked. Today we
# open the box:
#
# 1. **Data engineering**: start from the raw export that "prepared" dataset came from, and find
#    out what was wrong with it.
# 2. **Labels and the golden set**: how far can we trust a label, and how do we build an
#    evaluation set we *can* trust?
# 3. **The maths**: what `fit` actually did, written from scratch in NumPy.
# 4. **Loss design and label noise**: make the loss care about what we care about, and use a
#    model to find wrong labels.
# 5. **Synthetic data**: generate examples for the cases the model gets wrong, without fooling
#    ourselves.
# 6. **Retrain and compare** against the lab 1 model.
#
# Cells marked **TODO** raise `NotImplementedError` until you fill them in. The finished version
# is in `solutions/`.

# %%
import sys
from pathlib import Path

try:
    import mdlc
except ImportError:  # running from a checkout without `pip install -e .`
    sys.path.insert(0, str(Path.cwd().parent))
    import mdlc

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from mdlc import LABELS, data, metrics, text
from mdlc.nn import encode_labels, predict_proba, train_classifier

pd.set_option("display.max_colwidth", 140)
warnings.filterwarnings("ignore", message="Golden labels are unreviewed")
paths = data.build_all()

# %% [markdown]
# ---
# ## 1. Data engineering (35 min)
#
# This is the export lab 1's "prepared" data was built from: 20,700 reviews, read with every column
# as text, because we don't trust the types yet. **Look before you clean.**

# %%
raw = data.load_raw()
print(raw.shape)
raw.sample(5, random_state=1)

# %% [markdown]
# ### 1a. Labels you can't trust
# `stars` should hold whole numbers from 1 to 5.

# %%
raw.stars.value_counts()

# %% [markdown]
# **TODO 1:** return the integer for a valid rating and `None` for anything else. Don't rescue
# `"5 stars"` or `"four"`: guessing creates labels nobody gave.
#
# Every finding below is scored against the answer key of problems that were planted in this file.

# %%
def parse_stars(value: str) -> int | None:
    ### BEGIN SOLUTION
    value = value.strip()
    return int(value) if value in {"1", "2", "3", "4", "5"} else None
    ### END SOLUTION

# %%
raw["stars_int"] = raw.stars.map(parse_stars)
data.score_findings(raw.loc[raw.stars_int.isna(), "review_id"], "invalid_label")

# %% [markdown]
# ### 1b. Text that isn't a review, and markup
# Placeholders ("N/A", "test", empty strings) and HTML left by a web form. The library has a
# reference rule for each. Read them in `mdlc/data.py` and `mdlc/text.py`.

# %%
display(raw.assign(n=raw.text.str.len()).sort_values("n").head(8)[["review_id", "stars", "text"]])
display(data.score_findings(raw.loc[raw.text.map(data.is_placeholder), "review_id"], "empty_or_placeholder"))
has_html = raw.text.str.contains(r"<[a-z/][^>]*>|&[a-z#0-9]+;", regex=True)
example = raw.loc[has_html, "text"].iloc[0]
print("BEFORE:", example[:200], "\nAFTER: ", text.clean_text(example)[:200])
data.score_findings(raw.loc[has_html, "review_id"], "html_markup")

# %% [markdown]
# ### 1c. Duplicates
# Duplicates are the most common way to fool yourself. If the same review is in both the training
# data and the evaluation data, the model is scored on something it memorised.
#
# **TODO 2:** find duplicates *after normalisation* (`text.normalize_for_dedup` lowercases and
# strips punctuation). Return the `review_id` of every row except the first in each group: the rows
# you would drop.

# %%
DUPLICATES = ["exact_duplicate", "conflicting_duplicate", "near_duplicate"]

def exact_duplicate_ids(df: pd.DataFrame) -> pd.Series:
    ### BEGIN SOLUTION
    norm = df.text.map(text.normalize_for_dedup)
    return df.loc[norm.duplicated(keep="first"), "review_id"]
    ### END SOLUTION

# %%
data.score_findings(exact_duplicate_ids(raw), DUPLICATES)

# %% [markdown]
# Exact matching misses about half of the near-copies: truncated reviews, reviews with an "Update:"
# appended. Those need **near-duplicate detection**:
#
# 1. Cut each text into overlapping 3-word chunks ("the food was", "food was cold", ...).
# 2. Two texts are near-duplicates if they share most of their chunks. The **Jaccard similarity**
#    is shared chunks ÷ all distinct chunks; we use a threshold of 0.7.
# 3. Comparing all 200 million pairs is too slow, so **MinHash** gives each text a short
#    fingerprint, and only texts whose fingerprints partly collide get compared.
#
# Some duplicates also carry **conflicting labels**: the same text rated both 5 ★ and 1 ★. You can't
# tell which is right, so drop every copy and log them for review.

# %%
pairs = text.find_near_duplicates(raw.text.tolist(), threshold=0.7)
i, j, sim = next(p for p in pairs if p[2] < 0.95)
print(f"{len(pairs)} near-duplicate pairs. Example, similarity {sim:.2f}:\nA: {raw.text[i][:150]}\nB: {raw.text[j][:150]}")
data.score_findings({raw.review_id[j] for _, j, _ in pairs}, DUPLICATES)

# %% [markdown]
# ### 1d. Problems you only see in the whole dataset

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 2.8))
a1.plot(pd.to_numeric(raw.stars, errors="coerce").to_numpy(), ".", markersize=1)
a1.set_title("stars by row position in the file"); a1.set_xlabel("row")
raw.stars_int.dropna().astype(int).map(data.stars_to_sentiment).value_counts().plot.bar(ax=a2, rot=0, title="label mix")
plt.tight_layout(); plt.show()
raw[["stars", "stars_display"]].drop_duplicates().head(6)

# %% [markdown]
# - **Sorted by label.** "Last 10% of the file is the dev set" would give a dev set of 5-star reviews only.
# - **Target leakage.** `stars_display` *is* the label, drawn as stars. A model given it would score
#   ~100% in testing and be useless in production. For every column, ask: *will this exist, with
#   this meaning, at prediction time?*
# - **Imbalance.** 62% positive, 12% neutral.

# %% [markdown]
# ### 1e. Theory: development data, evaluation data, and imbalance
#
# **Development data** is what you build the model with:
# - **train**: the model learns from it.
# - **dev**: you compare models and settings on it, as often as you like. Because you look at it so
#   often, your decisions slowly fit it too. That's fine: it's what it's for.
#
# **Evaluation data** is what you judge the model with. It only stays honest if it is:
#
# | Principle | Why | How we did it |
# |---|---|---|
# | **Independent** of development data | a review in both is measured on memory | separate source file (Yelp's official test split); near-duplicates against the dev pool removed |
# | **Looked at rarely** | every look-then-change turns it into another dev set | the held-out set is locked behind `i_am_done_iterating=True` |
# | **Fit for its question** | "how good in production?" and "how good on hard cases?" are different questions | **held-out**: 5,000 reviews, production label mix. **golden**: ~400 hard-case-heavy reviews with checked labels |
# | **Frozen and versioned** | so scores are comparable over months | hashed manifests; changes are deliberate |
#
# **Stratified splitting** cuts each label separately, so every split has the same mix. With 12%
# neutral and a random split, a small dev set can easily end up with too few neutrals to measure.
#
# **Leakage**, the ways evaluation data sneaks into development:
# 1. the same or a near-identical item in both (1c);
# 2. a column that encodes the label (1d);
# 3. preprocessing fitted on all the data: lab 1 fitted the TF-IDF vocabulary on train only;
# 4. related items split apart: the same customer, product or time period on both sides. Fix it by
#    splitting by *group* or by *time*. Our data has no such fields, but real data usually does.
#
# **Dealing with imbalance:**
#
# | Option | Effect | Caveat |
# |---|---|---|
# | Measure per class (macro-F1, confusion matrix) | you *see* the problem | doesn't fix it |
# | **Class weights in the loss** | rare-class mistakes cost more during training | section 4 |
# | Resampling (duplicate rare, drop common) | rebalances what the model sees | **training set only**: never resample dev or evaluation data |
# | Decision thresholds | trade precision for recall after training | tune on dev |

# %% [markdown]
# ### 1f. Clean, split, version
# The reference pipeline applies every decision above in one function. It's what produced lab 1's
# data. In a real project, this function lives in version control and runs the same way every time.

# %%
clean, log = data.reference_clean(raw)
log

# %% [markdown]
# **TODO 3:** stratified split. Divide `df` into train and dev (90/10) so each label is split in
# the same proportion. Shuffle with `seed`. Hint: `df.groupby("label")`, then cut each group.

# %%
def train_dev_split(df: pd.DataFrame, dev_fraction: float = 0.1, seed: int = 42):
    ### BEGIN SOLUTION
    train_parts, dev_parts = [], []
    for _, group in df.groupby("label"):
        group = group.sample(frac=1, random_state=seed)
        n_dev = int(round(len(group) * dev_fraction))
        dev_parts.append(group.iloc[:n_dev])
        train_parts.append(group.iloc[n_dev:])
    return (pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True),
            pd.concat(dev_parts).sample(frac=1, random_state=seed).reset_index(drop=True))
    ### END SOLUTION

# %%
train, dev = train_dev_split(clean)
naive_dev = clean.iloc[-len(dev):]  # "just take the end of the file"
display(pd.DataFrame({"train": train.label.value_counts(normalize=True), "dev": dev.label.value_counts(normalize=True),
                      "naive tail": naive_dev.label.value_counts(normalize=True)}).round(3))
assert not set(train.review_id) & set(dev.review_id)
prepared_train, prepared_dev = data.load_prepared()
print("same reviews as lab 1's prepared data:",
      set(train.review_id) | set(dev.review_id) == set(prepared_train.review_id) | set(prepared_dev.review_id))
data.save_versioned({"train": train, "dev": dev}, data.DATA_DIR / "splits" / "lab2",
                    note="lab 2: reference_clean + stratified 90/10, seed 42")["splits"]["dev"]

# %% [markdown]
# Notice what nothing in this section caught: the answer key has **~600 rows whose stars were
# flipped**. Row by row they look perfectly valid. Section 4 finds them.

# %% [markdown]
# ---
# ## 2. Labels and the golden set (15 min)
#
# ### 2a. How much do people agree?
# Label these 10 reviews yourself: negative, neutral or positive. Work alone, and don't look at the
# stars.

# %%
exercise = pd.read_csv(paths["exercise"]).head(10)
for row in exercise.itertuples():
    print(f"[{row.item_id}] {row.text}\n")

# %%
MY_LABELS: dict[str, str] = {
    # "G0123": "negative",
}

golden = data.load_golden()
reference = golden.set_index("item_id").label
if MY_LABELS:
    ids = [i for i in MY_LABELS if i in reference.index]
    print(f"you vs golden: Cohen's kappa {metrics.cohen_kappa([MY_LABELS[i] for i in ids], reference[ids].tolist()):.2f}")
else:
    ex = exercise[exercise.item_id.isin(reference.index)]
    print(f"(fill MY_LABELS) stars vs golden on these items: "
          f"kappa {metrics.cohen_kappa(ex.star_label.tolist(), reference[ex.item_id].tolist()):.2f}")

# %% [markdown]
# **Cohen's kappa** measures agreement *beyond chance*: $\kappa = (p_o - p_e) / (1 - p_e)$, where
# $p_o$ is how often two raters agree and $p_e$ is how often they'd agree by luck, given how often
# each uses each label. Two raters who both say "positive" 62% of the time agree a lot by luck alone.
# Rough reading: < 0.4 poor, 0.4–0.6 moderate, 0.6–0.8 substantial, > 0.8 excellent.
#
# ### 2b. How good were the star labels?
# Our labels came from stars: 1–2 ★ negative, 3 ★ neutral, 4–5 ★ positive. The golden labels
# were assigned by reading the text against a written guideline. Compare:

# %%
pd.crosstab(golden.label, golden.stars, rownames=["golden label"], colnames=["stars"])

# %% [markdown]
# Look at the 3 ★ column. Most 3-star reviews are **not neutral** when you read them. "3 stars =
# neutral" was a convenient proxy, and it's the largest source of wrong labels in the whole dataset.
# That's a big part of why lab 1's model found so few neutral reviews: it was taught that neutral
# means "whatever 3-star reviewers write".
#
# ### 2c. What makes a golden set
#
# | Property | Why | Ours |
# |---|---|---|
# | Written guideline with edge cases | so labels mean the same thing next quarter | `golden_labels/GUIDELINE.md` |
# | ≥ 2 independent raters, disagreements adjudicated | one rater's habits aren't ground truth | two LLM raters (κ = 0.95), disagreements go to a person |
# | Hard cases included **on purpose** (slices) | random samples are mostly easy | negation, contrast, sarcasm, comparisons, short, long |
# | Independent of training data | otherwise it measures memory | different source file, deduplicated |
# | Small and frozen | quality beats size; scores stay comparable | ~400 items, versioned |
#
# Using LLMs as draft raters is a legitimate way to scale labelling, *if* you check them: measure
# their agreement, send disagreements to people, spot-check the rest. Never let the same model
# that you're evaluating label its own golden set.

# %%
print(golden.attrs["label_source"])
display(golden.label.value_counts().to_frame("items"))
golden.explode("slices").slices.value_counts().to_frame("items")

# %% [markdown]
# Note how few neutral items there are. Per-class scores for neutral on the golden set rest on
# ~30 examples, so treat differences of a few points as noise.

# %% [markdown]
# ---
# ## 3. The maths (30 min)
#
# ### 3a. The lab 1 lexicon was already a model
# Lab 1's baseline gave every word a weight by hand: +1 for "great", −1 for "rude", 0 for everything
# else. A review's score was the sum of the weights of its words. In vector form, with `x` the
# review's word counts and `w` the weights:
#
# $$\text{score} = w \cdot x = \sum_i w_i x_i$$
#
# With three classes, keep one weight vector per class (a 3 × vocabulary matrix $W$) plus a bias
# per class, and get three scores, the **logits**:
#
# $$z = W x + b$$
#
# **Learning** means choosing $W$ and $b$ from data instead of by hand. That's all a linear
# classifier is. Lab 1's network did the same thing with one more layer in between.

# %%
vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vectorizer.fit_transform(train.text)
X_dev = vectorizer.transform(dev.text)
y_train, y_dev = encode_labels(train.label), encode_labels(dev.label)
Y_train = np.eye(len(LABELS))[y_train]  # one-hot: row i has a 1 in the column of review i's label
print(X_train.shape, Y_train[:3])

# %% [markdown]
# ### 3b. From scores to probabilities: softmax
# $$p_c = \frac{e^{z_c}}{\sum_k e^{z_k}}$$
# Exponentiating makes every score positive and exaggerates differences; dividing makes them sum
# to 1. Scores (2, 1, 0) become probabilities (0.67, 0.24, 0.09).
#
# **TODO 4:** implement softmax for a matrix of logits (one row per review). Subtract each row's
# maximum before exponentiating: it doesn't change the result, and it stops `exp` overflowing for
# large scores.

# %%
def softmax(Z: np.ndarray) -> np.ndarray:
    ### BEGIN SOLUTION
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)
    ### END SOLUTION

# %%
P = softmax(np.array([[2.0, 1.0, 0.0], [1000.0, 1000.0, 1000.0]]))
assert np.allclose(P.sum(axis=1), 1) and np.allclose(P[0], [0.665, 0.245, 0.090], atol=1e-3)
assert np.allclose(P[1], 1 / 3), "large scores overflowed: subtract the row max first"
print(P.round(3))

# %% [markdown]
# ### 3c. Measuring wrongness: cross-entropy
# For one review whose true class is $y$, the loss is
# $$L = -\log p_y$$
# the negative log of the probability the model gave **the right answer**. The total loss is the
# average over reviews.
#
# | probability on the right answer | loss |
# |---|---|
# | 0.99 | 0.01 |
# | 0.5 | 0.69 |
# | 0.1 | 2.3 |
# | 0.001 | 6.9 |
#
# A confident wrong answer is punished far more than a hesitant one. Why the log? Maximising the
# probability of all the training labels at once means maximising a *product* of probabilities.
# Taking −log turns that product into a sum, and maximising into minimising. Cross-entropy is
# "make the observed labels as likely as possible".
#
# **TODO 5:** mean cross-entropy, given probabilities `P` (n × 3) and integer labels `y` (n).

# %%
def cross_entropy(P: np.ndarray, y: np.ndarray) -> float:
    ### BEGIN SOLUTION
    return float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
    ### END SOLUTION

# %%
assert abs(cross_entropy(np.array([[0.5, 0.25, 0.25]]), np.array([0])) - np.log(2)) < 1e-9
assert abs(cross_entropy(np.full((4, 3), 1 / 3), np.array([0, 1, 2, 0])) - np.log(3)) < 1e-9
print("an untrained model that says 1/3 for everything has loss", round(np.log(3), 3))

# %% [markdown]
# ### 3d. Which way to move each weight: the gradient
# The **gradient** says, for every weight, how much the loss changes if that weight increases a
# little. For softmax + cross-entropy it comes out remarkably simple (derivation in the appendix):
#
# $$\frac{\partial L}{\partial W} = \frac{1}{n}\, X^\top (P - Y) \qquad \frac{\partial L}{\partial b} = \frac{1}{n} \sum_i (p_i - y_i)$$
#
# $P - Y$ is "predicted probability minus truth" for each review and class. If the model gave
# "positive" 0.9 but the review was negative, that entry is +0.9, so every word in that review gets
# its "positive" weight pushed down. It's the lexicon idea again, but the weights adjust themselves.
#
# **TODO 6:** return `(dW, db)`. `X` is a sparse matrix; `X.T @ M` works as usual.

# %%
def gradients(X, P: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ### BEGIN SOLUTION
    n = X.shape[0]
    D = (P - Y) / n
    return np.asarray(X.T @ D), D.sum(axis=0)
    ### END SOLUTION

# %% [markdown]
# Check against PyTorch. In lab 1, `loss.backward()` computed exactly this automatically
# (**autograd**). Here we compare it with your formula on a small batch.

# %%
rng = np.random.default_rng(0)
W0, b0 = rng.normal(0, 0.1, (X_train.shape[1], 3)), rng.normal(0, 0.1, 3)
Xs, ys = X_train[:256], y_train[:256]
dW, db = gradients(Xs, softmax(Xs @ W0 + b0), np.eye(3)[ys])

Wt = torch.tensor(W0, requires_grad=True); bt = torch.tensor(b0, requires_grad=True)
loss = torch.nn.functional.cross_entropy(torch.tensor(Xs.toarray()) @ Wt + bt, torch.tensor(ys))
loss.backward()
assert np.allclose(dW, Wt.grad.numpy(), atol=1e-10) and np.allclose(db, bt.grad.numpy(), atol=1e-10)
print("your gradient matches PyTorch's autograd ✓")

# %% [markdown]
# ### 3e. Gradient descent
# Start with all weights at zero, then repeat: compute probabilities, compute the gradient, take a
# small step *against* it. The step size is the **learning rate**.

# %%
def train_linear(X, y, lr=10.0, steps=1500, log_every=100):
    Y = np.eye(len(LABELS))[y]
    W, b = np.zeros((X.shape[1], len(LABELS))), np.zeros(len(LABELS))
    history = []
    for step in range(steps + 1):
        P = softmax(X @ W + b)
        if step % log_every == 0:
            history.append((step, cross_entropy(P, y)))
        dW, db = gradients(X, P, Y)
        W -= lr * dW
        b -= lr * db
    return W, b, history

curves = {}
for lr in [1.0, 10.0, 25.0]:
    W, b, curves[lr] = train_linear(X_train, y_train, lr=lr, steps=600)
for lr, h in curves.items():
    plt.plot(*zip(*h), "o-", label=f"lr = {lr}")
plt.xlabel("step"); plt.ylabel("training loss"); plt.ylim(0, 2.5); plt.legend(); plt.show()

# %% [markdown]
# - **lr = 1**: every step is safe but tiny. It would get there, eventually.
# - **lr = 25**: steps so big they overshoot the minimum, so the loss jumps around.
# - **lr = 10**: about right.
#
# Train properly, and compare with lab 1's network:

# %%
W, b, _ = train_linear(X_train, y_train, lr=10.0, steps=1500)
linear_pred = [LABELS[i] for i in (X_dev @ W + b).argmax(axis=1)]
print(f"NumPy linear model, dev: accuracy {metrics.accuracy(dev.label, linear_pred):.3f}  "
      f"macro-F1 {metrics.macro_f1(dev.label, linear_pred):.3f}")
vocab = vectorizer.get_feature_names_out()
pd.DataFrame({lab: vocab[np.argsort(-W[:, k])[:10]] for k, lab in enumerate(LABELS)})

# %% [markdown]
# About as good as lab 1's network, from 20 lines of NumPy. The table shows the largest learned
# weights per class: the lexicon, rebuilt from data. On TF-IDF features a linear model is a strong
# baseline, and a network has to earn its extra complexity.
#
# Everything else in deep learning (more layers, other architectures, other optimisers) is this same
# loop: **forward → loss → gradient → step**.

# %% [markdown]
# ---
# ## 4. Loss design and label noise (15 min)
#
# ### 4a. Why the model ignores neutral
# The loss is an **average over reviews**. 62% of training reviews are positive and 12% neutral,
# so getting neutral wrong barely moves the average. The model learns that "rarely say neutral" is
# cheap. Fix it by changing what a mistake costs:
#
# $$L = -\frac{1}{\sum_i w_{y_i}} \sum_i w_{y_i} \log p_{i, y_i}$$
#
# Each class gets a **weight** $w_c$. Inverse frequency ($w_c \propto 1/\text{count}_c$) makes every
# class count equally in total. Its square root is a gentler middle ground.
#
# The same tool expresses **business costs**: if missing an angry customer is the expensive
# mistake, give the negative class a higher weight. The loss function is where you tell the model
# what you care about.

# %%
counts = train.label.value_counts()
inverse = np.array([len(train) / (len(LABELS) * counts[lab]) for lab in LABELS])
weightings = {"none": None, "sqrt inverse": np.sqrt(inverse), "inverse": inverse}
pd.DataFrame({k: v if v is not None else np.ones(3) for k, v in weightings.items()}, index=LABELS).round(2)

# %%
rows = []
for name, w in weightings.items():
    model, _ = train_classifier(X_train, y_train, X_dev, y_dev, class_weights=w, seed=0)
    pred = [LABELS[i] for i in predict_proba(model, X_dev).argmax(axis=1)]
    rep = metrics.classification_report(dev.label, pred)
    rows.append({"class weights": name, "dev macro-F1": rep.loc["macro avg", "f1"],
                 **{f"recall {lab}": rep.loc[lab, "recall"] for lab in LABELS}})
pd.DataFrame(rows).set_index("class weights").round(3)

# %% [markdown]
# Neutral recall goes up and positive recall goes down. Weights don't make the model smarter; they
# move where its mistakes land. Which trade-off is right is a **product decision**.

# %% [markdown]
# ### 4b. Using the model to find wrong labels
# A model that has learned the general pattern will be very *surprised* by a review whose label is
# wrong: high loss on that example. So: train on one half, measure per-review loss on the other half,
# swap, and look at the most surprising reviews. (Each review must be scored by a model that never
# trained on it, otherwise the model has simply memorised the wrong label.)

# %%
def out_of_fold_loss(df: pd.DataFrame, X, y, seed: int = 0) -> np.ndarray:
    halves = np.array_split(np.random.default_rng(seed).permutation(len(df)), 2)
    losses = np.zeros(len(df))
    for fit_idx, score_idx in [(halves[0], halves[1]), (halves[1], halves[0])]:
        m, _ = train_classifier(X[fit_idx], y[fit_idx], X_dev, y_dev, seed=seed)
        P = predict_proba(m, X[score_idx])
        losses[score_idx] = -np.log(P[np.arange(len(score_idx)), y[score_idx]] + 1e-12)
    return losses

train["oof_loss"] = out_of_fold_loss(train, X_train, y_train)
suspects = train.nlargest(600, "oof_loss")
display(data.score_findings(suspects.review_id, "label_noise"))
suspects.head(8)[["stars", "label", "oof_loss", "text"]]

# %% [markdown]
# About half of the 600 most surprising reviews are planted flips. Read the rest of the list: many
# are **real** label errors too, mostly 3-star reviews that clearly aren't neutral (section 2b).
#
# What to do with suspects, from cheapest to best: **drop** them, **down-weight** them, or
# **send them to people** to relabel. We'll drop them. In production, relabelling is usually worth it.

# %%
train_clean = train.drop(index=suspects.index).reset_index(drop=True)
print(f"train: {len(train):,} → {len(train_clean):,}")

# %% [markdown]
# ---
# ## 5. Synthetic data (10 min)
#
# The lab 1 model was weakest on sarcasm, mixed verdicts, negation and lukewarm reviews. Real
# examples of those are rare, so we asked an LLM to write 300 reviews aimed at exactly those cases.
# The prompt is in `synthetic/PROMPT.md`.

# %%
synthetic = pd.read_json(data.LAB_ROOT / "synthetic" / "synthetic_reviews.jsonl", lines=True)
display(pd.crosstab(synthetic.category, synthetic.label))
synthetic.groupby("category").head(1)[["category", "label", "text"]]

# %% [markdown]
# **Rules for synthetic data:**
# 1. **Train only. Never evaluate on it.** A model scored on LLM-written text tells you how well it
#    imitates that LLM, not how it does on your customers.
# 2. **Check for contamination**: nothing generated may be a near-copy of an evaluation item. If
#    the generator saw your evaluation data, it can leak it back.
# 3. **Measure the effect on real evaluation data**, with and without, and keep it only if it helps.
# 4. **Keep it a minority** of the training data, and record the prompt and model used.

# %%
eval_texts = pd.concat([golden.text, data.load_heldout(i_am_done_iterating=True).text]).tolist()
combined = synthetic.text.tolist() + eval_texts
leaks = [(i, j) for i, j, _ in text.find_near_duplicates(combined, threshold=0.5)
         if i < len(synthetic) <= j]
print(f"synthetic reviews that near-duplicate an evaluation item: {len(leaks)}")

# %% [markdown]
# (Opening the held-out set to search it for leaks is fine: we only compare texts, we don't score
# or tune anything.)

# %% [markdown]
# ---
# ## 6. Retrain and compare (10 min)
#
# Four candidates, each trained on our own lab 2 data and compared with the lab 1 model on the
# **golden set**: human-standard labels, hard cases included. We use 3 seeds each, because with
# ~380 golden items, differences of a point or two can be luck.

# %%
def lab1_model():
    """The model saved at the end of lab 1, or the same recipe retrained if it's missing."""
    d = data.DATA_DIR / "models"
    if (d / "lab1_model.pt").exists():
        vec = pickle.loads((d / "lab1_vectorizer.pkl").read_bytes())
        m = torch.nn.Sequential(torch.nn.Linear(len(vec.vocabulary_), 32), torch.nn.ReLU(), torch.nn.Linear(32, 3))
        m.load_state_dict(torch.load(d / "lab1_model.pt"))
        return m, vec
    tr, dv = data.load_prepared()
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(tr.text)
    m, _ = train_classifier(vec.transform(tr.text), encode_labels(tr.label), vec.transform(dv.text), encode_labels(dv.label))
    return m, vec


def golden_scores(model, vec, name):
    pred = [LABELS[i] for i in predict_proba(model, vec.transform(golden.text)).argmax(axis=1)]
    rep = metrics.classification_report(golden.label, pred)
    return {"model": name, "golden macro-F1": rep.loc["macro avg", "f1"],
            "golden accuracy": metrics.accuracy(golden.label, pred), "neutral recall": rep.loc["neutral", "recall"]}, pred


def train_variant(df, class_weights=None, seed=0):
    vec = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(df.text)
    m, _ = train_classifier(vec.transform(df.text), encode_labels(df.label), vec.transform(dev.text), y_dev,
                            class_weights=class_weights, seed=seed)
    return m, vec


m1, v1 = lab1_model()
results = [golden_scores(m1, v1, "lab 1 model")[0]]
syn = synthetic[["text", "label"]]
variants = {
    "+ class weights": (train, np.sqrt(inverse)),
    "+ class weights, noise removed": (train_clean, np.sqrt(inverse)),
    "+ class weights, noise removed, synthetic": (pd.concat([train_clean, syn], ignore_index=True), np.sqrt(inverse)),
}
for name, (df, w) in variants.items():
    for seed in range(3):
        results.append({**golden_scores(*train_variant(df, w, seed), name)[0], "seed": seed})
scores = pd.DataFrame(results).groupby("model", sort=False)[["golden macro-F1", "golden accuracy", "neutral recall"]]
scores.agg(["mean", "std"]).round(3)

# %% [markdown]
# Read the table honestly:
# - Which change made the biggest difference? Is it bigger than the spread between seeds?
# - Did synthetic data help on the *real* golden set? If not, it doesn't go in. That's a valid
#   result, and a common one: 300 generated reviews against 17,000 real ones is a small nudge, and
#   generated text doesn't always look like your customers' text.
#
# We pick the candidate with the best mean golden macro-F1. Note what that means: **the golden set
# was just used to make a decision**, so it is now slightly optimistic for the winner. That's why
# the held-out set stays locked until after the choice is made.
#
# ### Per slice: the chosen model against lab 1

# %%
final_name = scores.mean()["golden macro-F1"].drop("lab 1 model").idxmax()
print("chosen:", final_name)
final_model, final_vec = train_variant(*variants[final_name], seed=0)
g = golden.assign(lab1=golden_scores(m1, v1, "")[1], lab2=golden_scores(final_model, final_vec, "")[1])
metrics.slice_report(g, "label", "lab1")[["n", "macro_f1"]].join(
    metrics.slice_report(g, "label", "lab2")[["macro_f1"]], lsuffix=" lab 1", rsuffix=" lab 2")

# %% [markdown]
# ### The held-out set, once
# Note what this measures. The held-out labels come from **star ratings**, the same proxy we just
# showed is often wrong for 3-star reviews. A model that labels a positive 3-star review "positive"
# is *right* but is scored as wrong here. That's why the golden set is the primary yardstick for
# this decision, and why the held-out labels deserve a relabelling pass of their own.

# %%
heldout = data.load_heldout(i_am_done_iterating=True)
for name, (m, v) in {"lab 1": (m1, v1), "lab 2": (final_model, final_vec)}.items():
    pred = [LABELS[i] for i in predict_proba(m, v.transform(heldout.text)).argmax(axis=1)]
    print(f"{name}: held-out accuracy {metrics.accuracy(heldout.label, pred):.3f}  macro-F1 {metrics.macro_f1(heldout.label, pred):.3f}")

# %% [markdown]
# ---
# ## Wrap-up: the loop
#
# ```
#   frame the problem ──► data: collect, clean, split, version ──► labels: guideline, golden set
#          ▲                                                                  │
#          │                                                                  ▼
#   monitor in production ◄── ship with a model card ◄── evaluate on golden ◄── train: loss, model
# ```
#
# What we did in each step today, and what you'd add before shipping:
#
# | Step | Today | Before production |
# |---|---|---|
# | Data | cleaning pipeline, dedup, stratified train/dev, versioning | group/time splits if the data has them |
# | Labels | guideline, double-rated golden set, label-noise detection | human review of golden disagreements; relabel the suspects |
# | Training | the maths; class weights | thresholds tuned to the cost of each mistake |
# | Evaluation | golden overall and per slice, held-out once | sign-off criteria agreed *before* looking |
# | Synthetic | targeted, decontaminated, train-only | keep only what measurably helps |
# | Shipping | | a **model card** (data, intended use, metrics per slice, known failures), monitoring for input drift and label drift, a schedule for refreshing the golden set |

# %% [markdown]
# ---
# ## Appendix: deriving the gradient
#
# For one review with logits $z$, probabilities $p = \text{softmax}(z)$ and true class $y$:
# $L = -\log p_y = -z_y + \log \sum_k e^{z_k}$.
#
# Differentiate with respect to $z_c$: the first term gives $-1$ if $c = y$, else 0. The second
# gives $e^{z_c} / \sum_k e^{z_k} = p_c$. So $\partial L / \partial z_c = p_c - [c = y]$, or in
# vector form, $p - y_{\text{one-hot}}$.
#
# Since $z = W^\top x + b$, the chain rule gives $\partial L / \partial W = x\,(p - y)^\top$ and
# $\partial L / \partial b = p - y$. Averaging over $n$ reviews (rows of $X$) gives
# $\frac{1}{n} X^\top (P - Y)$.
