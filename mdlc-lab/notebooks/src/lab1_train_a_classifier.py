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
# Someone has already cleaned this dataset and split it into three parts. (Lab 2: you'll do that
# yourself, starting from a much messier file.)

# %%
train, val, test = data.load_prepared()
print(f"train {len(train):,}   validation {len(val):,}   test {len(test):,}")
train.sample(5, random_state=3)[["stars", "label", "text"]]

# %% [markdown]
# ### Why three splits?
#
# | Split | Used for | How often we look |
# |---|---|---|
# | **train** | the model learns from these | constantly |
# | **validation** | we compare models and settings on these | as often as we like |
# | **test** | the final, honest score | **once**, at the very end |
#
# The model must be scored on reviews it has never seen, otherwise we're measuring memory rather
# than understanding. And every time *we* look at a score and change something in response, that
# set leaks into our decisions. That's why the test set is touched only once.

# %% [markdown]
# ### Class balance
# How are the labels distributed?

# %%
mix = pd.DataFrame({name: df.label.value_counts(normalize=True)
                    for name, df in [("train", train), ("validation", val), ("test", test)]})
mix.loc[list(LABELS)].plot.bar(figsize=(7, 3), rot=0, title="label mix per split")
plt.ylabel("share of reviews"); plt.show()
mix.round(3)

# %% [markdown]
# Two things to notice:
#
# 1. **The classes are imbalanced.** Most reviews are positive and only about 1 in 9 is neutral.
#    That's normal for real data, and it changes how we must measure success (next section).
# 2. **Every split has the same mix.** That's on purpose. The split was *stratified*: each label was
#    divided 80/10/10 separately. A careless split can easily put most of the neutral reviews in
#    train and almost none in test.

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
    pred = m.predict(val.text)
    scoreboard.append({"model": name, "accuracy": metrics.accuracy(val.label, pred),
                       "macro_f1": metrics.macro_f1(val.label, pred)})
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

show_confusion(val.label, baselines["majority"].predict(val.text), "majority baseline")
show_confusion(val.label, baselines["lexicon"].predict(val.text), "lexicon baseline")

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
X_val = vectorizer.transform(val.text)
X_test = vectorizer.transform(test.text)
y_train, y_val, y_test = encode_labels(train.label), encode_labels(val.label), encode_labels(test.label)
print("train matrix:", X_train.shape, f"({X_train.nnz / X_train.shape[0]:.0f} non-zero slots per review on average)")

# %% [markdown]
# > **Data-handling rule:** the vectorizer is *fitted* on the training set only, then *applied* to
# > validation and test. If we built the vocabulary from all the data, information about the test
# > set would leak into the model's inputs. Every step that learns something from data follows this
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
probs = predict_proba(model, X_val[:3])
pd.DataFrame(probs, columns=LABELS).assign(truth=val.label[:3].values, text=val.text[:3].str[:80].values).round(3)

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
    """Train, recording train/val loss and accuracy after each epoch. Keeps the best epoch's weights."""
    loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best = [], (float("inf"), None)
    for epoch in range(1, epochs + 1):
        for xb, yb in batches(X_train, y_train, batch_size, seed=epoch):
            train_step(model, optimizer, loss_fn, xb, yb)
        tr_loss, tr_acc = evaluate(model, X_train, y_train, loss_fn)
        va_loss, va_acc = evaluate(model, X_val, y_val, loss_fn)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=va_loss, train_acc=tr_acc, val_acc=va_acc))
        if verbose:
            print(f"epoch {epoch}: train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f}")
        if va_loss < best[0]:
            best = (va_loss, copy.deepcopy(model.state_dict()), epoch)
    model.load_state_dict(best[1])
    if verbose:
        print(f"kept the weights from epoch {best[2]} (lowest validation loss)")
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
# - **Validation loss** falls, bottoms out after an epoch or two, then **rises**. From that point on,
#   the model is memorising quirks of the training reviews that don't carry over to new ones.
#   That's **overfitting**.
# - That's why `fit` keeps the weights from the epoch with the lowest validation loss. This is
#   called **early stopping**, and it's only possible because we have a validation set.

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

val_pred = predict_labels(model, X_val)
scoreboard.append({"model": "network (ours)", "accuracy": metrics.accuracy(val.label, val_pred),
                   "macro_f1": metrics.macro_f1(val.label, val_pred)})
display(pd.DataFrame(scoreboard).set_index("model").round(3))
show_confusion(val.label, val_pred, "network, validation set")
metrics.classification_report(val.label, val_pred)

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
# ### The test set: once
# We've made our choices using validation. Now the honest number:

# %%
test_pred = predict_labels(model, X_test)
print(f"TEST  accuracy {metrics.accuracy(test.label, test_pred):.3f}   macro-F1 {metrics.macro_f1(test.label, test_pred):.3f}")

# %% [markdown]
# If it's close to the validation score, good: our choices didn't overfit to the validation set.
# Don't go back and tune now. If you did, the test set would become just another validation set.

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
# A more systematic check: reviews from a separate pool, tagged by the kind of difficulty they
# contain (**slices**). The overall score is an average, and averages hide weak spots.

# %%
hard = data.load_golden_candidates()
hard["label"] = hard.star_label  # star labels for now. Lab 2 replaces these with human-checked labels
hard["pred"] = predict_labels(model, vectorizer.transform(hard.text))
metrics.slice_report(hard, "label", "pred").sort_values("macro_f1")

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
# - learned why we split data three ways and why the test set is touched once,
# - saw overfitting happen and stopped it early,
# - learned why accuracy lies when classes are imbalanced,
# - found the cases where it fails.
#
# **And now the uncomfortable part.** The "prepared" dataset you trained on started life as an
# export with **seven kinds of problems** in it, including about **600 reviews whose star ratings
# were flipped** (a glowing 5 ★ review stored as 1 ★). Those wrong labels are still in your
# training data. Your model learned from them, and your validation score was measured against them.
#
# **Lab 2:** start from the raw export, find what's wrong with it, learn the maths of what `fit`
# just did, and retrain a better model on data you can trust.
