# %% [markdown]
# # Lab 1: Train a sentiment classifier
#
# In the next two hours you'll train a neural network that reads a restaurant review and decides
# whether the customer is **negative**, **neutral** or **positive**. Then you'll use it, measure it,
# and try to break it.
#
# This lab is deliberately hands-on: the data is already prepared, and the theory is kept to what
# you need to make sense of what you see. **Lab 2** opens the box: where the data came from, what
# was wrong with it, and the maths behind training.
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

import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from mdlc import LABELS, data, metrics
from mdlc.baselines import LexiconClassifier, MajorityClassifier
from mdlc.nn import batches, encode_labels, evaluate, plot_history, predict_proba

pd.set_option("display.max_colwidth", 140)
data.build_all()  # downloads ~200 MB the first time, then takes about a minute
torch.manual_seed(0);

# %% [markdown]
# ---
# ## 1. The task (10 min)
#
# A **classifier** is a function: `f(review text) -> one of {negative, neutral, positive}`.
#
# Writing `f` by hand is hard. Try listing every way people express disappointment. Instead we'll
# show a program thousands of reviews **with the right answer attached** and let it find `f` itself.
# That is **supervised learning**, and the "right answers" are called **labels**.
#
# Where do our labels come from? Each review came with a 1–5 star rating:
# 1–2 ★ = negative, 3 ★ = neutral, 4–5 ★ = positive.
#
# Think about the use: say every *negative* review gets routed to a support agent. Which mistake
# costs more: an angry customer we miss, or a happy customer an agent reads for nothing?
# Keep your answer in mind for section 7.

# %% [markdown]
# ---
# ## 2. The data (15 min)
#
# Someone has already cleaned this dataset and split it for you. (Lab 2: you'll do that yourself,
# starting from a much messier file.)

# %%
train, dev = data.load_prepared()
print(f"train {len(train):,}   dev {len(dev):,}")
train.sample(5, random_state=3)[["stars", "label", "text"]]

# %% [markdown]
# ### Development data and evaluation data
#
# | | Set | Used for | How often we look |
# |---|---|---|---|
# | **Development** | **train** | the model learns from these | constantly |
# | | **dev** | we compare models and settings on these | as often as we like |
# | **Evaluation** | **held-out** | the final, honest score on realistic data | **once**, at the very end |
# | | **golden** | quality on hard cases, with carefully checked labels | rarely |
#
# The model must be scored on reviews it has never seen, otherwise we're measuring memory rather
# than understanding. And every time *we* look at a score and change something in response, that
# set leaks into our decisions. The dev set absorbs that; the evaluation sets are kept clean by
# looking at them as little as possible. The evaluation sets also come from a **separate source
# file**, checked so that no review in them is a copy of one in the development data.

# %% [markdown]
# ### Class balance
# How are the labels distributed?

# %%
mix = pd.DataFrame({name: df.label.value_counts(normalize=True) for name, df in [("train", train), ("dev", dev)]})
mix.loc[list(LABELS)].plot.bar(figsize=(7, 3), rot=0, title="label mix per split")
plt.ylabel("share of reviews"); plt.show()
mix.round(3)

# %% [markdown]
# Two things to notice:
#
# 1. **The classes are imbalanced.** Most reviews are positive and only about 1 in 9 is neutral.
#    That's normal for real data, and it changes how we must measure success (next section).
# 2. **Train and dev have the same mix.** That's on purpose. The split was *stratified*: each label
#    was divided 90/10 separately. A careless split can easily put most of the neutral reviews in
#    train and almost none in dev.

# %% [markdown]
# ---
# ## 3. The bar to beat (10 min)
#
# Before training anything, check what the dumbest reasonable approaches score:
#
# - **majority**: always answer "positive".
# - **lexicon**: count words from a hand-written list (+1 for "great", "delicious"…,
#   −1 for "rude", "bland"…).

# %%
baselines = {
    "majority": MajorityClassifier().fit(train.text, train.label),
    "lexicon": LexiconClassifier(handle_negation=True),
}
scoreboard = []
for name, m in baselines.items():
    pred = m.predict(dev.text)
    scoreboard.append({"model": name, "accuracy": metrics.accuracy(dev.label, pred),
                       "macro_f1": metrics.macro_f1(dev.label, pred)})
pd.DataFrame(scoreboard).set_index("model").round(3)

# %% [markdown]
# "Always positive" is right 62% of the time and it's completely useless: it never finds a single
# unhappy customer. **Accuracy hides that when classes are imbalanced.** So we also report:
#
# - **recall** for a class: of the reviews that really were negative, how many did we catch?
# - **precision** for a class: of the reviews we *called* negative, how many really were?
# - **F1**: combines the two, and is high only when both are.
# - **macro-F1**: the average F1 over the three classes. The rare *neutral* class counts as much as
#   *positive*, so a model can't score well by ignoring it.
#
# The **confusion matrix** shows everything at once: rows are the truth, columns are the prediction.

# %%
def show_confusion(y_true, y_pred, title=""):
    cm = metrics.confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=[f"true {l}" for l in LABELS], columns=[f"pred {l}" for l in LABELS])
    print(title)
    display(df)

show_confusion(dev.label, baselines["majority"].predict(dev.text), "majority baseline")
show_confusion(dev.label, baselines["lexicon"].predict(dev.text), "lexicon baseline")

# %% [markdown]
# **The bar:** macro-F1 ≈ 0.50 from a word list someone wrote in ten minutes.

# %% [markdown]
# ---
# ## 4. Turning text into numbers (15 min)
#
# A network does arithmetic, so each review has to become a list of numbers (a **vector**).
# We'll use **TF-IDF**:
#
# 1. Build a vocabulary of the 20,000 most useful words and two-word phrases ("not good").
# 2. Each review becomes a vector with one slot per vocabulary entry.
# 3. A slot is non-zero if the review uses that word. The value is higher when the word is frequent
#    in this review (**TF**, term frequency) but rare across all reviews (**IDF**, inverse document
#    frequency). "the" gets almost nothing; "disgusting" gets a lot.

# %%
vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
X_train = vectorizer.fit_transform(train.text)  # learn the vocabulary from TRAIN ONLY
X_dev = vectorizer.transform(dev.text)
y_train, y_dev = encode_labels(train.label), encode_labels(dev.label)
print("train matrix:", X_train.shape, f"({X_train.nnz / X_train.shape[0]:.0f} non-zero slots per review on average)")

# %% [markdown]
# > **Data-handling rule:** the vectorizer is *fitted* on the training set only, then *applied* to
# > everything else. If we built the vocabulary from all the data, information about the dev and
# > evaluation sets would leak into the model's inputs. Every step that learns something from data follows this
# > rule.
#
# Here is what one review looks like to the network:

# %%
vocab = vectorizer.get_feature_names_out()
i = 7
row = X_train[i]
print(train.text[i][:300], "\n")
pd.Series(row.data, index=vocab[row.indices]).sort_values(ascending=False).head(12).round(3).to_frame("tf-idf")

# %% [markdown]
# ---
# ## 5. The model (10 min)
#
# A **neural network** is a function with lots of adjustable numbers called **weights**. Ours:
#
# ```
# 20,000 TF-IDF numbers ──► Linear ──► 32 numbers ──► ReLU ──► Linear ──► 3 scores
#                                                                         (neg, neu, pos)
# ```
#
# - **Linear**: every output is a weighted sum of all the inputs. This is where the weights live.
# - **ReLU**: replaces negative numbers with 0. Without it, two Linear layers collapse into one.
#   It's what lets the network combine evidence in non-trivial ways.
# - The 3 output scores (**logits**) become probabilities with **softmax**: exponentiate each
#   one, then divide by the total, so they're positive and sum to 1.

# %%
def make_model(hidden: int = 32) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, len(LABELS)),
    )

torch.manual_seed(0)
model = make_model()
n_params = sum(p.numel() for p in model.parameters())
print(model, f"\n\n{n_params:,} weights")

# %% [markdown]
# Before training, the weights are random, so its "opinions" are noise:

# %%
probs = predict_proba(model, X_dev[:3])
pd.DataFrame(probs, columns=LABELS).assign(truth=dev.label[:3].values, text=dev.text[:3].str[:80].values).round(3)

# %% [markdown]
# ---
# ## 6. Training (30 min)
#
# Training is a loop that nudges the weights so the predictions get a bit less wrong each time:
#
# 1. take a small **batch** of reviews (64 at a time),
# 2. **forward pass**: run them through the model to get scores,
# 3. compute the **loss**: one number that says how wrong those predictions were,
# 4. **backward pass**: PyTorch works out, for every weight, which direction would reduce the loss
#    (the **gradient**),
# 5. the **optimizer** moves every weight a small step in that direction (the **learning rate**
#    sets the step size).
#
# One pass over all the training data is an **epoch**.
#
# Our loss is **cross-entropy**: roughly, *how little probability did the model give the correct
# label?* If the model puts 0.95 on the right answer, the loss is small (0.05). If it puts 0.01 on
# it, the loss is large (4.6). Lab 2 shows where this formula comes from and how to change it.
#
# **TODO 1:** write the five lines of one training step. You need:
# `optimizer.zero_grad()` (clear the gradients left over from the last step), `model(xb)`,
# `loss_fn(logits, yb)`, `loss.backward()` and `optimizer.step()`. Return the loss.

# %%
def train_step(model, optimizer, loss_fn, xb, yb) -> torch.Tensor:
    ### BEGIN SOLUTION
    optimizer.zero_grad()
    logits = model(xb)
    loss = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()
    return loss
    ### END SOLUTION

# %%
def fit(model, epochs=5, lr=1e-3, batch_size=64, loss_fn=None, verbose=True):
    """Train, recording train/dev loss and accuracy after each epoch. Keeps the best epoch's weights."""
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best = [], (float("inf"), None)
    for epoch in range(1, epochs + 1):
        for xb, yb in batches(X_train, y_train, batch_size, seed=epoch):
            train_step(model, optimizer, loss_fn, xb, yb)
        tr_loss, tr_acc = evaluate(model, X_train, y_train, loss_fn)
        va_loss, va_acc = evaluate(model, X_dev, y_dev, loss_fn)
        history.append(dict(epoch=epoch, train_loss=tr_loss, dev_loss=va_loss, train_acc=tr_acc, dev_acc=va_acc))
        if verbose:
            print(f"epoch {epoch}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | dev loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_loss < best[0]:
            best = (va_loss, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    if verbose:
        print(f"kept the weights from epoch {best[2]} (lowest dev loss)")
    return history

# %%
# Check: one step on one batch should lower the loss on that batch
torch.manual_seed(0)
_m = make_model(); _opt = torch.optim.Adam(_m.parameters(), lr=1e-2); _lf = torch.nn.CrossEntropyLoss()
_xb, _yb = next(batches(X_train, y_train, 64))
_before = _lf(_m(_xb), _yb).item()
train_step(_m, _opt, _lf, _xb, _yb)
assert _lf(_m(_xb), _yb).item() < _before, "loss did not go down: check the order of the five calls"
print("train_step works ✓")

# %%
torch.manual_seed(0)
model = make_model(hidden=32)
history = fit(model, epochs=6, lr=1e-3)
plot_history(history, "hidden=32, lr=0.001")

# %% [markdown]
# ### Reading the curves
# - **Training loss** keeps falling: the model fits the training reviews better and better.
# - **Dev loss** falls, bottoms out after an epoch or two, then **rises**. From that point on,
#   the model is memorising quirks of the training reviews that don't carry over to new ones.
#   That's **overfitting**.
# - That's why `fit` keeps the weights from the epoch with the lowest dev loss. This is called
#   **early stopping**, and it's only possible because we have a dev set.

# %% [markdown]
# ### Experiments (pick two, ~20 s each)
# Change one setting at a time and compare the curves with the run above.
#
# | Try | What to look for |
# |---|---|
# | `hidden=128, epochs=8` | more capacity: does it overfit sooner? |
# | `lr=1e-1` | step size too big: does the loss jump around? |
# | `lr=1e-5` | step size too small: does it learn anything in 6 epochs? |
# | `hidden=4` | too little capacity? |
# | `batch_size=512` | fewer, smoother steps per epoch |

# %%
torch.manual_seed(0)
experiment = make_model(hidden=128)
plot_history(fit(experiment, epochs=8, lr=1e-3, verbose=False), "experiment: hidden=128, 8 epochs")

# %% [markdown]
# ---
# ## 7. How good is it? (15 min)

# %%
def predict_labels(model, X):
    return [LABELS[i] for i in predict_proba(model, X).argmax(axis=1)]

dev_pred = predict_labels(model, X_dev)
scoreboard.append({"model": "network (ours)", "accuracy": metrics.accuracy(dev.label, dev_pred),
                   "macro_f1": metrics.macro_f1(dev.label, dev_pred)})
display(pd.DataFrame(scoreboard).set_index("model").round(3))
show_confusion(dev.label, dev_pred, "network, dev set")
metrics.classification_report(dev.label, dev_pred)

# %% [markdown]
# **Discuss:**
# - The network clearly beats the word list. Where does it still go wrong? Read the confusion
#   matrix row by row.
# - Look at the recall for **neutral**. The model rarely predicts the rare class, because the loss
#   rewards it mostly for getting the common classes right. Imbalance strikes again. (Lab 2:
#   fixing this by changing the loss.)
# - Remember your answer from section 1: which off-diagonal cell is the expensive mistake?
#   How many of those are there?
#
# ### The held-out set: once
# We've made all our choices using the dev set. Now the honest number, on 5,000 reviews from a
# separate source that nothing so far has touched. The loader makes you say you mean it.

# %%
heldout = data.load_heldout(i_am_done_iterating=True)
heldout_pred = predict_labels(model, vectorizer.transform(heldout.text))
print(f"HELD-OUT  accuracy {metrics.accuracy(heldout.label, heldout_pred):.3f}   "
      f"macro-F1 {metrics.macro_f1(heldout.label, heldout_pred):.3f}")

# %% [markdown]
# If it's close to the dev score, good: our choices didn't overfit to the dev set.
# Don't go back and tune now. If you did, the held-out set would become just another dev set.

# %% [markdown]
# ---
# ## 8. Play with it (15 min)

# %%
def classify(*reviews: str) -> pd.DataFrame:
    p = predict_proba(model, vectorizer.transform(reviews))
    df = pd.DataFrame(p, columns=LABELS).round(2)
    df.insert(0, "prediction", [LABELS[i] for i in p.argmax(axis=1)])
    df.insert(0, "review", reviews)
    return df

classify(
    "The pasta was incredible and the staff couldn't have been nicer.",
    "Waited an hour, food came cold, manager shrugged.",
    "It was fine. Nothing special, nothing wrong.",
)

# %% [markdown]
# What did it learn? A quick way to peek: feed the model each vocabulary entry *on its own* and
# see which ones push hardest towards each label.

# %%
single_words = sp.identity(len(vocab), format="csr", dtype=np.float32)
p = predict_proba(model, single_words)
lean = pd.DataFrame({"word": vocab, "pos - neg": p[:, 2] - p[:, 0], "neutral": p[:, 1]})
pd.concat([
    lean.nlargest(12, "pos - neg").word.reset_index(drop=True).rename("most positive"),
    lean.nsmallest(12, "pos - neg").word.reset_index(drop=True).rename("most negative"),
    lean.nlargest(12, "neutral").word.reset_index(drop=True).rename("most neutral"),
], axis=1)

# %% [markdown]
# Nobody wrote these lists. The network found them in 16,000 examples.

# %% [markdown]
# ---
# ## 9. Break it (15 min)
#
# **Your turn:** write reviews that you think will fool the model. Ideas: negation ("not bad at
# all"), sarcasm, mixed verdicts ("great food, awful service"), comparisons, very short reviews,
# typos, slang, emojis.

# %%
classify(
    "Not bad at all, actually.",
    "Oh great, another 45 minute wait for a cold burger. Fantastic.",
    "Food was amazing but the service was so rude I'm never coming back.",
    "Better than the place next door, which isn't saying much.",
    "meh",
    # add your own here
)

# %% [markdown]
# A more systematic check: the **golden set**. It's a few hundred reviews, picked on purpose to
# contain hard cases, whose labels were checked against a written guideline rather than taken from
# the stars. Each review is tagged with the kind of difficulty it contains (**slices**). The
# overall score is an average, and averages hide weak spots. (Lab 2: how a golden set is built.)

# %%
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    golden = data.load_golden()
print(f"{len(golden)} golden reviews, labels: {golden.attrs['label_source']}")
golden["pred"] = predict_labels(model, vectorizer.transform(golden.text))
metrics.slice_report(golden, "label", "pred").sort_values("macro_f1")

# %% [markdown]
# ---
# ## 10. Save it
# We'll compare against this model in lab 2.

# %%
out = data.DATA_DIR / "models"
out.mkdir(exist_ok=True)
torch.save(model.state_dict(), out / "lab1_model.pt")
with open(out / "lab1_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
pd.DataFrame(scoreboard).to_csv(out / "lab1_scoreboard.csv", index=False)
print("saved to", out)

# %% [markdown]
# ---
# ## Wrap-up
#
# In under two hours you:
# - trained a neural network that beats a hand-written rule by a wide margin,
# - learned why development and evaluation data are kept apart, and why the held-out set is touched once,
# - saw overfitting happen and stopped it early,
# - learned why accuracy lies when classes are imbalanced,
# - found the cases where it fails.
#
# **And now the uncomfortable part.** The "prepared" dataset you trained on started life as an
# export with **seven kinds of problems** in it, including about **600 reviews whose star ratings
# were flipped** (a glowing 5 ★ review stored as 1 ★). Those wrong labels are still in your
# training data. Your model learned from them, and your dev score was measured against them.
#
# **Lab 2:** start from the raw export, find what's wrong with it, learn the maths of what `fit`
# just did, and retrain a better model on data you can trust.
