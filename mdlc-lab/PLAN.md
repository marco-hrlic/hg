# MDLC Lab: Building a Sentiment Classifier, End to End

Status: **v0.4**. Two labs, both built and ready for a dry run. The golden labels are LLM drafts
awaiting human review of 19 disputed items.

## Decisions

| Decision | Choice |
|---|---|
| Format | **2 labs × 2h** |
| Shape | Lab 1: the payoff first (prepared data, train, use, break). Lab 2: open the box (raw data, theory, retrain) |
| Data | Public: Yelp Review Full (1–5 stars), downloaded and rebuilt from a fixed seed, never committed |
| Audience | Developers comfortable with Python, **no ML background** |
| Runtime | CPU only, Python 3.10+, PyTorch, local or Colab. No API keys (synthetic data ships pre-generated) |
| Data sets | **Development**: train / dev (90/10, stratified) from Yelp's train file. **Evaluation**: golden (~400, checked labels, hard-case heavy) and held-out (5,000, production mix, locked) from Yelp's test file |
| Golden labels | LLM drafts from two independent raters, disagreements reviewed by a person |

## Goal

Developers leave able to:

- turn a vague request into a labelled classification problem with a metric
- handle data properly: class balance, stratified splits, leakage, deduplication, versioning, a locked test set
- explain what training does: score → softmax → cross-entropy → gradient → update
- read loss curves, spot overfitting, use early stopping
- choose or design a loss function for what the business actually cares about
- curate a golden dataset, and use synthetic data without fooling themselves

## The story across the two labs

**Lab 1** trains a network on a "prepared" dataset and gets a satisfying result: 0.82 accuracy and
0.68 macro-F1 on dev, against 0.50 for a hand-written rule, in about 15 seconds of training. It ends with a
reveal: the prepared data came from an export with seven kinds of problems, and ~600 flipped labels
are still in it.

**Lab 2** starts from that raw export, teaches the data engineering and the theory behind what
lab 1 did, fixes the data and the loss, and retrains. The lab 2 model beats lab 1 on the golden
set (macro-F1 0.719 → 0.747, mean of 3 seeds; spread about ±0.01). Synthetic data, honestly
measured, adds nothing beyond noise here, which is a lesson in itself.

The lab 1 dataset is produced by the same reference cleaning pipeline that lab 2 teaches
(`mdlc.data.reference_clean`), so participants can compare their own cleaning with it.

## Where each topic lives

| Topic | Lab 1 (show it, use it) | Lab 2 (explain it, do it yourself) |
|---|---|---|
| Class balance | Look at the label mix; majority baseline at 62% accuracy is useless; neutral recall ≈ 0.19 | Why the loss ignores rare classes; class weights, resampling (train only), thresholds |
| Data splits | Development (train, dev) vs evaluation (held-out, golden); stratified mix; held-out touched once | Make them: stratification, the sorted-file trap, independence of evaluation data, duplicates across sets, target leakage, versioning |
| Metrics | Accuracy vs macro-F1, confusion matrix, precision/recall | Compute them yourself; slices; cost of each error type |
| Training | The loop (TODO: write the step), loss curves, overfitting, early stopping, hyperparameter experiments | The maths, from scratch in NumPy, checked against PyTorch |
| Loss function | "Cross-entropy = how little probability on the right answer" | Derivation; class weights; cost-sensitive loss; label smoothing |
| Data quality | Prepared for you (with a reveal at the end) | Hunt planted problems; find flipped labels using per-example loss |
| Golden set | Hard-case pool with slices shows where the model fails | Labelling agreement (kappa), guideline, curation principles |
| Synthetic data | Failures collected in "break it" | Targeted generation for weak slices, contamination check, evaluate on real data only |

---

## Lab 1 (2h): Train a classifier. *Built.* `notebooks/lab1_train_a_classifier.ipynb`

| Time | Block | Hands-on |
|---|---|---|
| 0:00–0:10 | The task: what a classifier is, supervised learning, star → sentiment labels, the expensive mistake | Discussion |
| 0:10–0:25 | The data: **development vs evaluation data**, **class balance** | Plot the label mix |
| 0:25–0:35 | The bar to beat: majority and lexicon baselines; accuracy vs macro-F1; confusion matrix | Run baselines |
| 0:35–0:50 | Text → numbers: TF-IDF; vectorizer fitted on train only (first leakage rule) | Inspect one review as a vector |
| 0:50–1:00 | The model: Linear → ReLU → Linear, logits, softmax; 640k random weights | Look at an untrained model's predictions |
| 1:00–1:30 | Training: batches, forward, loss, backward, step; epochs; overfitting; early stopping | **TODO: write the training step**; two experiments (capacity, learning rate, batch size) |
| 1:30–1:45 | Evaluation: scoreboard vs baselines, per-class report (neutral recall), the held-out set *once* | Read the confusion matrix against the expensive mistake |
| 1:45–1:55 | Play and break it: classify your own reviews; learned word lists; per-slice scores on the golden set | Write reviews that fool the model |
| 1:55–2:00 | Save the model; the reveal; lead-in to lab 2 | |

## Lab 2 (2h): Open the box. *Built.* `notebooks/lab2_data_and_theory.ipynb`

This is the dense one. To fit 2h, most code is provided and the exercises are short and focused.
The blocks marked *demo* are instructor-led walkthroughs with the code already in the notebook.

| Time | Block | Hands-on |
|---|---|---|
| 0:00–0:05 | Recap, and the raw export lab 1's data came from | |
| 0:05–0:40 | **Data engineering**: invalid labels, placeholders, HTML, exact/near duplicates (MinHash), target-leakage column, sorted file, imbalance; development vs evaluation data; stratified train/dev split; leakage; versioning; compare with lab 1's prepared data | 3 short TODOs (parse labels, dedup, stratified split), each scored against the answer key |
| 0:40–0:55 | **Labels and the golden set**: 10-item labelling exercise and Cohen's kappa; star labels vs checked labels (most 3-star reviews aren't neutral); what makes a golden set; LLM raters with human review | Label 10 reviews; compare with the golden labels |
| 0:55–1:25 | **The maths**: the lexicon *is* a linear model with hand-picked weights; softmax; cross-entropy from "probability of the right answer"; the gradient `(p − y)·x`; gradient descent | **TODO: softmax, cross-entropy and gradient in NumPy**, checked against PyTorch autograd |
| 1:25–1:40 | **Loss design and label noise**: class weights for neutral recall, cost-sensitive loss for the expensive mistake; per-example loss as a detector for flipped labels | Train with class weights; find flipped labels, scored against the answer key |
| 1:40–1:50 | **Synthetic data** *(demo)*: 300 pre-generated examples aimed at weak slices, contamination check against evaluation data, the four rules | |
| 1:50–2:00 | **Retrain and compare**: candidates vs the lab 1 model on the golden set (3 seeds) and per slice; the held-out set once; the MDLC loop | Final scoreboard |

---

## Deliverables

| Item | Status |
|---|---|
| `mdlc/` package: data build, reference cleaning, MinHash dedup, metrics, slices, baselines, PyTorch helpers | done, tested |
| Datasets: raw export + answer key, prepared train/dev, golden candidates, labelling exercise, locked held-out set | done (`make data`) |
| Lab 1 notebook (student + solution) and instructor notes | done, runs in ~1 min |
| Lab 2 notebook (student + solution) and instructor notes | done, runs in ~4 min |
| Labelling guideline, LLM draft golden labels, review sheet, finalize script | done |
| Synthetic data (300 reviews) and the prompt used | done |
| **Human review of the golden labels** (19 disputed items, 101 spot-checks) | todo: see `golden_labels/README.md` |
| Dry run of both labs with a pilot group | todo |

## Open questions

1. After the dry run: is lab 2 too dense for 2h? Candidates to cut or move to optional reading: the
   appendix derivation, the learning-rate plot, the slice table in section 6.
