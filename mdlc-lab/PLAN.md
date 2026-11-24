# MDLC Lab: Building a Sentiment Classifier, End to End

Status: **v0.5**. Restructured around Claude Code: participants build with the agent from step
cards, verified by an independent harness. Both labs built. Both labs passed an agent dry run (an agent
playing the participant completed each from the cards alone). Golden labels await human review.

## Decisions

| Decision | Choice |
|---|---|
| Format | **2 labs × 2h** |
| Shape | Lab 1: the payoff first (prepared data, train, use, break). Lab 2: open the box (raw data, theory, retrain) |
| **Format** | **Participants build with Claude Code from step cards.** The agent writes the code; participants decide, predict and verify. The `lab` harness holds evaluation labels, scores submissions and checks each step. Notebooks remain as self-study material |
| Learning goal | Not "write it alone afterwards" but: see every component, learn the vocabulary and the ML way of working, so they can direct agents and research further |
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
lab 1 did, fixes the data and the loss, and retrains, comparing against lab 1's recipe retrained in
the same script with the same seeds. In the reference run the golden macro-F1 changes are **within
noise** (candidate 0: 0.733 ± 0.012; best: 0.741), while class weights clearly shift neutral recall and
held-out moves 0.695 → 0.719. The current lesson of 2.8 is therefore evaluation power and controlled
comparison, more than "lab 2 wins". (An earlier claim of 0.719 → 0.747 compared against a single
lab 1 run and is withdrawn.)

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

## Lab 1 (2h): Train a classifier

| Step | Min | Participants do (with Claude Code) | Verified by |
|---|---|---|---|
| 1.0 How we work | 10 | setup; the seven habits | |
| 1.1 Meet the data | 15 | profile train/dev; predict majority accuracy | journal |
| 1.2 Baselines | 20 | majority + keyword baselines; pick a metric | `lab check 1.2` |
| 1.3 Text to numbers | 15 | TF-IDF fitted on train only | verify questions |
| 1.4 Training | 25 | network + training loop, curves, early stopping | `lab check 1.4` |
| 1.5 Experiments | 15 | seed noise + two one-change experiments, predicted first | `lab check 1.5` |
| 1.6 Evaluate | 10 | golden + held-out submissions | `lab check 1.6` |
| 1.7 Break it | 10 | tricky reviews, error analysis; the reveal | `lab check 1.7` |

## Lab 2 (2h): Open the box

| Step | Min | Participants do (with Claude Code) | Verified by |
|---|---|---|---|
| 2.1 Audit | 15 | find row- and dataset-level problems in the raw export | `lab check 2.1` (answer key) |
| 2.2 Clean & split | 15 | cleaning decisions; stratified train/dev; manifest | `lab check 2.2` (leakage, dups, escapes, stratification) |
| 2.3 Labels & golden set | 15 | label 10 reviews by hand; stars vs golden; the 3-star relabelling result | `lab check 2.3` (kappa) |
| 2.4 The maths | 20 | softmax, cross-entropy, gradient, gradient descent in NumPy | `lab check 2.4` (vs PyTorch autograd) |
| 2.5 Loss design | 15 | class weights; ship decision; a reusable `train()` | `lab check 2.5` |
| 2.6 Label noise | 10 | out-of-fold loss → suspects | `lab check 2.6` (answer key) |
| 2.7 More data | 15 | learning curve at 18k / 50k / 100k reviews | `lab check 2.7` |
| 2.8 Synthetic data | 5 | instructor demo; contamination | `lab check 2.8` |
| 2.9 Retrain & compare | 10 | candidates 0/A/B/D × 2 seeds; paired test on dev; held-out once vs lab 1 | `lab check 2.9` (paired bootstrap) |

## Deliverables

| Item | Status |
|---|---|
| `mdlc/` package: data build, reference cleaning, MinHash dedup, metrics, slices, baselines, PyTorch helpers | done, tested |
| Datasets: raw export + answer key, prepared train/dev, golden candidates, labelling exercise, locked held-out set | done (`make data`) |
| `lab` harness, workspace (CLAUDE.md, settings, journal), 16 step cards, reference solution for every step | done, tested |
| Instructor guide and per-lab notes | done |
| Lab 1 and lab 2 as worked notebooks (self-study) | done |
| Labelling guideline, LLM draft golden labels, review sheet, finalize script | done |
| Synthetic data (300 reviews) and the prompt used | done |
| **Human review of the golden labels** (19 disputed items, 101 spot-checks) | todo: see `golden_labels/README.md` |
| Agent dry run of lab 1 | done: all checks passed first time; fixes applied (see instructor/GUIDE.md) |
| Agent dry run of lab 2 | done: all checks passed; fixes applied, incl. a controlled candidate 0 in 2.8 (see instructor/GUIDE.md) |
| Dry run of both labs with a pilot group | todo |

## Open questions

1. Pacing: with agents writing code, is 2h enough for lab 2's eight steps, including debriefs?
   Find out in the pilot. First candidates to make optional: 2.7 (synthetic data) as an instructor
   demo, and running 2.8 with one seed instead of three.
2. Should the guardrails be stronger (e.g. evaluation labels on a separate scoring server), or is
   the honour system plus file-tool block enough for an internal lab?
