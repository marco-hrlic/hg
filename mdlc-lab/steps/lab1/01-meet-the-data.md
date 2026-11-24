# 1.1 Meet the data (10 min)
**Goal:** see what we're predicting and how the labels are spread.

## Why this matters
**Supervised learning** = learn `text → label` from examples. Labels come from stars:
1–2★ negative, 3★ neutral, 4–5★ positive. Train to learn, dev to compare; evaluation sets are hidden.

## Predict
A model that always says "positive" gets what accuracy on dev?  **A** 33%  **B** 60%  **C** 85%

## Prompt
> Write `src/explore.py`: review counts and label mix (%) for `data/lab1/train.csv` and `dev.csv`,
> median review length, and two example reviews per label.

## Verify
Is the label mix the same in train and dev? Would you agree with the example labels?

## Takeaway
Classes are **imbalanced**, so accuracy flatters lazy models. Next step: a better metric.

## Go deeper
class imbalance · stratified split · supervised learning ·
*Ask Claude: "Why would we want the same label mix in train and dev?"*
