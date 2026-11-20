# MDLC Lab: Building a Sentiment Classifier, End to End

Status: **v0.3**. Two labs. Lab 1 is built and ready for a dry run. Lab 2's data block exists as a
draft; its theory, label-noise, synthetic-data and retrain blocks are still to build.

## Decisions

| Decision | Choice |
|---|---|
| Format | **2 labs × 2h** |
| Shape | Lab 1: the payoff first (prepared data, train, use, break). Lab 2: open the box (raw data, theory, retrain) |
| Data | Public: Yelp Review Full (1–5 stars), downloaded and rebuilt from a fixed seed, never committed |
| Audience | Developers comfortable with Python, **no ML background** |
| Runtime | CPU only, Python 3.10+, PyTorch, local or Colab. No API keys (synthetic data ships pre-generated) |

## Goal

Developers leave able to:

- turn a vague request into a labelled classification problem with a metric
- handle data properly: class balance, stratified splits, leakage, deduplication, versioning, a locked test set
- explain what training does: score → softmax → cross-entropy → gradient → update
- read loss curves, spot overfitting, use early stopping
- choose or design a loss function for what the business actually cares about
- curate a golden dataset, and use synthetic data without fooling themselves

## The story across the two labs

**Lab 1** trains a network on a "prepared" dataset and gets a satisfying result: 0.81 accuracy and
0.65 macro-F1, against 0.50 for a hand-written rule, in about 15 seconds of training. It ends with a
reveal: the prepared data came from an export with seven kinds of problems, and ~600 flipped labels
are still in it.

**Lab 2** starts from that raw export, teaches the data engineering and the theory behind what
lab 1 did, fixes the data and the loss, and retrains. Success means beating the lab 1 model on
human-checked golden data.

The lab 1 dataset is produced by the same reference cleaning pipeline that lab 2 teaches
(`mdlc.data.reference_clean`), so participants can compare their own cleaning with it.

## Where each topic lives

| Topic | Lab 1 (show it, use it) | Lab 2 (explain it, do it yourself) |
|---|---|---|
| Class balance | Look at the label mix; majority baseline at 62% accuracy is useless; neutral recall ≈ 0.19 | Why the loss ignores rare classes; class weights, resampling (train only), thresholds |
| Data splits | Roles of train/val/test; stratified mix; test touched once | Make them: stratification, the sorted-file trap, duplicates across splits, target leakage, group and time splits (theory), versioning |
| Metrics | Accuracy vs macro-F1, confusion matrix, precision/recall | Compute them yourself; slices; cost of each error type |
| Training | The loop (TODO: write the step), loss curves, overfitting, early stopping, hyperparameter experiments | The maths, from scratch in NumPy, checked against PyTorch |
| Loss function | "Cross-entropy = how little probability on the right answer" | Derivation; class weights; cost-sensitive loss; label smoothing |
| Data quality | Prepared for you (with a reveal at the end) | Hunt planted problems; find flipped labels using per-example loss |
| Golden set | Hard-case pool with slices shows where the model fails | Labelling agreement (kappa), guideline, curation principles |
| Synthetic data | Failures collected in "break it" | Targeted generation for weak slices, contamination check, evaluate on real data only |

---

## Lab 1 (2h): Train a classifier. *Built.*

| Time | Block | Hands-on |
|---|---|---|
| 0:00–0:10 | The task: what a classifier is, supervised learning, star → sentiment labels, the expensive mistake | Discussion |
| 0:10–0:25 | The data: prepared splits, **why three splits**, **class balance** per split | Plot the label mix |
| 0:25–0:35 | The bar to beat: majority and lexicon baselines; accuracy vs macro-F1; confusion matrix | Run baselines |
| 0:35–0:50 | Text → numbers: TF-IDF; vectorizer fitted on train only (first leakage rule) | Inspect one review as a vector |
| 0:50–1:00 | The model: Linear → ReLU → Linear, logits, softmax; 640k random weights | Look at an untrained model's predictions |
| 1:00–1:30 | Training: batches, forward, loss, backward, step; epochs; overfitting; early stopping | **TODO: write the training step**; two experiments (capacity, learning rate, batch size) |
| 1:30–1:45 | Evaluation: scoreboard vs baselines, per-class report (neutral recall), the test set *once* | Read the confusion matrix against the expensive mistake |
| 1:45–1:55 | Play and break it: classify your own reviews; learned word lists; per-slice scores | Write reviews that fool the model |
| 1:55–2:00 | Save the model; the reveal; lead-in to lab 2 | |

## Lab 2 (2h): Open the box. *Data block drafted; the rest to build.*

This is the dense one. To fit 2h, most code is provided and the exercises are short and focused.
The blocks marked *demo* are instructor-led walkthroughs with the code already in the notebook.

| Time | Block | Hands-on |
|---|---|---|
| 0:00–0:05 | Recap, and the raw export lab 1's data came from | |
| 0:05–0:40 | **Data engineering**: invalid labels, placeholders, HTML, exact/near duplicates (MinHash), target-leakage column, sorted file, imbalance; stratified splits; leakage across splits; group and time splits (theory); versioning; compare with lab 1's prepared data | 3 short TODOs (parse labels, dedup, stratified split), each scored against the answer key |
| 0:40–0:55 | **Labels and the golden set**: 10-item labelling exercise and Cohen's kappa; what makes a golden set; slices; the reference golden labels | Label 10 reviews; compute agreement |
| 0:55–1:25 | **The maths**: the lexicon *is* a linear model with hand-picked weights; softmax; cross-entropy from "probability of the right answer"; the gradient `(p − y)·x`; gradient descent | **TODO: softmax, cross-entropy and gradient in NumPy**, checked against PyTorch autograd |
| 1:25–1:40 | **Loss design and label noise**: class weights for neutral recall, cost-sensitive loss for the expensive mistake; per-example loss as a detector for flipped labels | Train with class weights; find flipped labels, scored against the answer key |
| 1:40–1:50 | **Synthetic data** *(demo)*: pre-generated examples aimed at weak slices (sarcasm, mixed, negation), dedup against eval data, mix ratio | Train with and without; evaluate on real golden data only |
| 1:50–2:00 | **Retrain and compare**: the lab 2 model vs the lab 1 model on the golden set and per slice; unlock the held-out set once; the MDLC loop and what to monitor in production | Final scoreboard |

---

## Deliverables

| Item | Status |
|---|---|
| `mdlc/` package: data build, reference cleaning, MinHash dedup, metrics, slices, baselines, PyTorch helpers | done, tested |
| Datasets: raw export + answer key, prepared splits, golden candidates, labelling exercise, locked held-out set | done (`make data`) |
| Lab 1 notebook (student + solution) and instructor notes | **done**, runs end to end in ~1 min |
| Lab 2 data block (from the old session 1 notebook) | draft: needs compressing to 35 min and comparing with lab 1's prepared data |
| Lab 2 theory, loss design and label-noise blocks | todo |
| Pre-generated synthetic data for lab 2 (with the prompts used to make it) | todo |
| **Reference golden labels** for the 400 candidates | todo: 2 human labellers (~3h), or LLM draft labels plus human review |
| Lab 2 instructor notes | todo |

## Open questions

1. Who labels the reference golden set? (See deliverables.)
2. Group and time splits are taught as theory only, because Yelp Review Full has no business, user
   or date fields. Switching to a dataset with them (e.g. Amazon Reviews 2023) would make them hands-on.
3. LLM provider for an *optional* live synthetic-data demo (the lab works without one).
