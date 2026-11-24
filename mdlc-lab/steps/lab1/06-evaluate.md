# 1.6 Evaluate: golden and held-out (10 min)
**Goal:** the honest verdict on lab 1's model, and its weakest class.

## Why this matters
Dev picked our settings, so it flatters us. **Golden** (381 hard-case reviews, carefully labelled)
shows *where* it fails. **Held-out** (5,000 reviews) shows how good it is: **one look per lab**.

## Predict
Held-out macro-F1 compared with dev?  **A** clearly higher  **B** about the same (±0.02)  **C** clearly lower

## Prompt
> Write `src/predict.py`: load `outputs/model.pt` and `outputs/vectorizer.pkl`, predict
> `data/eval/golden_texts.csv` and `data/eval/heldout_texts.csv`, and save `outputs/golden_lab1.csv`
> and `outputs/heldout_lab1.csv` (`id,label`; `id` = `item_id` or `review_id`). Don't submit.

Then you run:
```
lab submit golden  outputs/golden_lab1.csv  --name lab1
lab submit heldout outputs/heldout_lab1.csv --name lab1
lab check 1.6
```

## Verify
Find the **neutral** row in the held-out report. Write its F1 in `notes.md`: lab 2 explains it.

## Takeaway
Decide on dev, evaluate once. Data you've tuned on can't tell you the truth.

## Go deeper
held-out test set · golden dataset · slice-based evaluation · benchmark contamination
