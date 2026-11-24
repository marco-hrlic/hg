# 1.2 Baselines and metrics (15 min)
**Goal:** two dumb baselines and the metric we'll use from now on.

## Why this matters
A **baseline** is the simplest thing that could work; a model must clearly beat it.
**F1** combines precision (were our "negative" calls right?) and recall (did we catch the negatives?);
**macro-F1** averages it over the classes, so rare neutral counts as much as common positive.
The **confusion matrix**: rows = truth, columns = prediction.

## Predict
"Always positive" scores 62% accuracy. Its macro-F1?  **A** about 0.25  **B** about 0.50  **C** about 0.62

## Prompt
> Write `src/baselines.py` with two baselines for `data/lab1/dev.csv`:
> 1. **majority**: always the most common label in train;
> 2. **keywords**: score = positive words − negative words, from two short word lists.
>    [DECISION: when is it "neutral"?] **A** score exactly 0 (default) · **B** score between −1 and +1 ·
>    **C** only a tie with keywords present (no keywords → positive)
>
> Save `outputs/dev_majority.csv` and `outputs/dev_keywords.csv` (`id,label`), run `lab score` on
> each, and explain the confusion matrices in three lines. Save the keyword baseline's confusion
> matrix as a heatmap to `outputs/confusion_keywords.png`.

## Verify
`lab check 1.2` · Find the majority baseline's zero columns in its confusion matrix.
Then pick in `notes.md`: compare models from now on by **A** accuracy or **B** macro-F1?

## Takeaway
Baseline and metric come first. Without them no number means anything.

## Go deeper
precision vs recall · F1 · macro vs micro averaging · confusion matrix
