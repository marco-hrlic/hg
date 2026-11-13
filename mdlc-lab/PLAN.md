# MDLC Lab: Building a Sentiment Classifier, End to End

Status: **v0.2**. Session 1 is built and ready for a dry run; sessions 2 and 3 are planned.

## Decisions

| Decision | Choice |
|---|---|
| Format | 3 sessions × 2h |
| Data | Public: Yelp Review Full (1–5 stars), downloaded and rebuilt from a fixed seed, never committed |
| Audience | Developers comfortable with Python, **no ML background**. Maths is taught through intuition and code first; formulas follow, and derivations are optional extras |
| Runtime | CPU only, Python 3.10+, local or Colab. No API keys required (synthetic data ships pre-generated) |

## Goal

Developers go through one full model development lifecycle on a problem small enough to
train on a laptop CPU. They should leave able to:

- turn a vague request ("detect unhappy customers") into a labelling spec and a metric
- handle data properly: splits, leakage, deduplication, versioning, a locked test set
- explain the maths of a linear classifier: score → softmax → cross-entropy → gradient
- write a training loop, read loss curves, and spot overfitting
- choose or design a loss function for what the business actually cares about
- curate a golden dataset, and generate synthetic data without fooling themselves

## Guiding principle: evaluation comes before the model

The order is deliberate: **spec → data → golden set → baseline → model → iterate**.
Developers build the thing that measures success before they build the thing being measured.
This is the habit that carries over to real projects.

## Running example

Customer-review sentiment, **3 classes (negative / neutral / positive)**, with
a 1–5 star score available as a secondary target.

- 3 classes rather than 2, because "neutral" and "mixed" produce real disagreement
  between labellers. That disagreement is what makes the golden-set discussion concrete.
- Star scores let us compare classification against ordinal/regression losses in session 2.
- The dataset ships **deliberately dirty**: exact and near duplicates, the same product
  appearing in both train and test, HTML junk, a few percent of flipped labels, class imbalance.

## Model progression (each step must beat the one before it on the golden set)

1. Majority class and a keyword-lexicon baseline
2. TF-IDF + logistic regression **written from scratch in NumPy**
3. The same model in PyTorch, with autograd checked against the hand-derived gradient
4. Small MLP or embedding-bag model in PyTorch
5. *(Optional demo)* fine-tuned small transformer and/or a zero-shot LLM, as a reference ceiling

---

## Session 1 (2h): Framing, data, golden set

| Time | Block | Hands-on |
|---|---|---|
| 0:00–0:15 | Framing: what is a classifier, what decision it feeds, the cost of each error type | Write a one-paragraph problem spec |
| 0:15–0:40 | **Labelling exercise**: everyone labels the same 30 reviews from a draft guideline | Compute Cohen's/Fleiss' kappa, discuss disagreements, revise the guideline |
| 0:40–1:15 | Data handling: EDA, cleaning, exact and near-dup detection, leakage (group split by product/user), stratified splits, the test set as a locked box, dataset versioning (hash manifest) | Find the planted problems in the dirty dataset |
| 1:15–1:40 | **Golden dataset curation**: adjudication, slice coverage (negation, sarcasm, mixed, domain terms, short text), size and refresh policy, freezing | Build a ~150-item golden set with slice tags |
| 1:40–2:00 | Metrics: accuracy vs macro-F1, confusion matrix, per-slice metrics | Score the majority and lexicon baselines. This is the bar to beat |

## Session 2 (2h): The maths and the training loop

| Time | Block | Hands-on |
|---|---|---|
| 0:00–0:20 | Text → vectors: bag-of-words, TF-IDF, (briefly) embeddings. Vectors and dot products explained as "weighted word counts" | Vectorise the dataset; inspect one review as a vector |
| 0:20–0:50 | **Maths, intuition first**: the session 1 lexicon *is* a linear model with hand-picked weights (+1/−1). Learning means choosing the weights automatically. Score → softmax ("turn scores into probabilities") → cross-entropy ("how surprised were we by the right answer") → gradient ("nudge each weight by (prediction − truth) × feature") → gradient descent ("repeat many times") | Implement each piece as a small NumPy function, checked by a test cell. Interactive plot: loss as a function of one weight. The paper derivation is an optional appendix |
| 0:50–1:15 | Training loop: epochs and mini-batches, learning rate, train/val curves, overfitting, L2 regularisation, early stopping, seeds | LR sweep; overfit on purpose, then fix it |
| 1:15–1:30 | Port to PyTorch: gradient check against the NumPy version, then the MLP | Same results in about 15 lines |
| 1:30–2:00 | **Loss function design**: class weights, cost-sensitive loss (negative→positive is the expensive mistake), label smoothing for noisy labels; stars as classification vs regression (MSE) vs ordinal | Each group trains with a different loss; compare on the golden set |

## Session 3 (2h): Error analysis, synthetic data, shipping

| Time | Block | Hands-on |
|---|---|---|
| 0:00–0:25 | Error analysis: per-slice failures, confusion pairs, finding label errors with model loss | Find the weakest slices; find mislabelled training items |
| 0:25–1:15 | **Synthetic data**: LLM generation targeted at weak slices, prompt design for diversity, LLM-as-labeller vs human, filtering, dedup **against the golden/test set** (contamination), mixing ratios | Train real-only vs real+synthetic; evaluate on **real golden data only** |
| 1:15–1:35 | Thresholds and calibration: reliability plot, choosing an operating point from the cost of each error | Pick a threshold for the "escalate to human" use case |
| 1:35–1:50 | Shipping: model card, what to monitor (input drift, label drift), when to refresh the golden set | Fill in a model card template |
| 1:50–2:00 | Retro, plus the MDLC loop back to the beginning | |

---

## Deliverables

| Item | Status |
|---|---|
| `mdlc/` helper package (data build, cleaning, MinHash dedup, metrics, slices, baselines) | done, tested |
| Dirty dataset + answer key, golden candidates, 30-item exercise, locked held-out set | done (`make data`) |
| Session 1 notebook (student + solution), instructor notes | done, solution runs end to end in ~20 s |
| **Reference golden labels** for the 400 candidates, so sessions 2–3 don't depend on how well a cohort labelled | todo: needs 2 human labellers (~3h), or LLM draft labels + human review |
| Session 2 notebook + instructor notes | todo |
| Pre-generated synthetic data for session 3 (with the prompts used to make it) | todo |
| Session 3 notebook + instructor notes, model card template | todo |

## Open questions

1. LLM provider for the *optional* live synthetic-data demo in session 3 (the lab works without it).
2. Who labels the reference golden set? (See deliverables.)
