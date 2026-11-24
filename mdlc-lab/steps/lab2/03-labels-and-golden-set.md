# 2.3 Labels and the golden set (15 min, no Claude for part 1)
**Goal:** find out how good our labels are, and why neutral failed in lab 1.

## Why this matters
A **golden set** is labelled carefully: a written guideline, two independent raters, agreement
measured with **Cohen's kappa** (1 = perfect, 0 = chance), disagreements settled by a person.

## Part 1: label 10 reviews yourself (5 min, no Claude)
Read `data/lab2/label_these_10.csv` and label each **negative / neutral / positive**. The short version
of `data/lab2/GUIDELINE.md`: the overall verdict wins; a stated intention to return or not counts most;
neutral only if truly lukewarm or balanced. Save `outputs/my_labels.csv` (`item_id,label`).

## Predict
Of the golden set's 3★ reviews, what share actually read as neutral?  **A** about 80%  **B** about 50%  **C** about 30%

## Verify
`lab check 2.3` · Look at the **3★ column** of the stars-vs-golden table. What did lab 1 train on?

## Takeaway
Labels are data too, and usually the noisiest part. Measure them before trusting them.

## Go deeper
inter-annotator agreement · Cohen's kappa · annotation guidelines · LLM-as-annotator ·
*Ask Claude: "What is weak supervision, and when is a proxy label good enough?"*
