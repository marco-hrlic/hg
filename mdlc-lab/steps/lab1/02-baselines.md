# 1.2 Baselines and metrics (20 min)

## Why this matters
A **baseline** is the simplest thing that could possibly work. It tells you what "good" means:
a model is worth its complexity only if it clearly beats the baseline.

**Accuracy** (fraction right) hides failures on rare classes. So we also use, per class:
- **precision**: of the reviews we *called* negative, how many were?
- **recall**: of the reviews that *were* negative, how many did we catch?
- **F1**: combines both; high only when both are high.
- **macro-F1**: the average F1 over the three classes, so the rare class counts as much as the common one.

The **confusion matrix** shows it all: rows are the truth, columns are predictions.

## Predict first (notes.md)
Majority baseline: accuracy = ? macro-F1 = ? Will a keyword-counting baseline beat it?

## Prompt
> Write `src/baselines.py` with two baselines, each predicting a label for every review in
> `data/lab1/dev.csv`:
> 1. **majority**: always predict the most common label *in train*;
> 2. **keywords**: count words from a short positive list and a short negative list, and decide the
>    label from the difference. [DECISION: when should it say "neutral"?]
>
> Save predictions as `outputs/dev_majority.csv` and `outputs/dev_keywords.csv` (columns `id,label`),
> then run `lab score` on each and explain the confusion matrices to me.

## Verify
`lab check 1.2`
- Why is the majority baseline's accuracy high but its macro-F1 so low? Find it in the confusion matrix.
- Decide, in notes.md: **which metric will we use to compare models from now on, and why?**

## ML thinking
**The baseline and the metric come first.** Without them, no number means anything.

## Go deeper
precision vs recall · F1 score · macro vs micro averaging · confusion matrix · baseline model
