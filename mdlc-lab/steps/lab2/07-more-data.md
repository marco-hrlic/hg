# 2.7 More data: the learning curve (15 min)

## Why this matters
So far we've trained on ~18,000 reviews. A second export of **82,000 more** has arrived
(`data/lab2/more_reviews.csv`: already cleaned, star-labelled, no overlap with anything else). Is
it worth using? More data costs storage, training time and labelling money in real projects, so
you measure before you decide.

A **learning curve** plots model quality against training-set size. If it's still rising at the
right-hand end, more data will help. If it's flat, spend your effort elsewhere (better labels, a
better model).

## Predict first (notes.md)
Training on 100k reviews instead of 18k: how much will dev macro-F1 change? +0.00? +0.02? +0.10?
And how much longer will training take?

## Prompt
> Using `train()` from `src/train2.py`, train on `outputs/train.csv` plus the first N rows of
> `data/lab2/more_reviews.csv`, for N = 0, 32,000 and 82,000 (about 18k, 50k and 100k reviews in
> total), always early-stopping on `outputs/dev.csv`. Record `n_train,dev_macro_f1,seconds` per run in
> `outputs/learning_curve.csv`, plot it to `outputs/learning_curve.png`, and tell me what the shape
> says. The biggest run takes a few minutes: run the script in the background.

## Verify
`lab check 2.7`
- Is the curve still rising? Roughly how much data would you need for the next +0.02?
- Ask Claude: *"Why does more data help a bag-of-words model more than it helps the keyword baseline?"*
- Ask: *"Would 100k reviews with 10% wrong labels beat 18k perfectly labelled ones? How would you find out?"*

## ML thinking
**Measure whether a resource is worth it before paying for it.** More data, a bigger model and
better labels are all costs.

## Go deeper
learning curves · scaling laws · data acquisition vs labelling cost · bias–variance trade-off · active learning
