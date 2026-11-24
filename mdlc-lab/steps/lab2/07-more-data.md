# 2.7 More data: the learning curve (15 min)
**Goal:** decide, with a measurement, whether 82,000 more reviews are worth using.

## Why this matters
A **learning curve** plots quality against training-set size. Still rising at the right-hand end:
more data helps. Flat: spend the effort elsewhere (labels, model).

## Predict
Training on 100k reviews instead of 18k changes dev macro-F1 by…  **A** about +0.00  **B** about +0.02  **C** +0.05 or more

## Prompt
> Using `train()` from `src/train2.py`, train on `outputs/train.csv` plus the first N rows of
> `data/lab2/more_reviews.csv`, for N = 0, 32,000 and 82,000, early-stopping on `outputs/dev.csv`.
> Save `outputs/learning_curve.csv` (`n_train,dev_macro_f1,seconds`) and `outputs/learning_curve.png`.
> Run it in the background; the biggest run takes a few minutes.

## Verify
`lab check 2.7` · Is the curve still rising at the right-hand end?

## Takeaway
Measure whether a resource is worth it before paying for it.

## Go deeper
learning curves · scaling laws · active learning ·
*Ask Claude: "Would 100k reviews with 10% wrong labels beat 18k clean ones? How would we find out?"*
