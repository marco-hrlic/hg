# 2.6 Find wrong labels with a model (10 min)
**Goal:** a list of 600 likely mislabelled training reviews.

## Why this matters
A model is **surprised** (high loss) by a label that contradicts everything it learned. But it
memorises its own training data, so score each review with a model that didn't see it:
**out-of-fold** prediction (train on half, score the other half, swap).

## Predict
Of the 600 most surprising reviews, how many are planted flips?  **A** about 100  **B** about 300  **C** about 550

## Prompt
> Using `train()` from `src/train2.py`, compute 2-fold out-of-fold cross-entropy loss for every review
> in `outputs/train.csv`. Save the 600 highest-loss reviews to `outputs/suspects.csv` (`review_id,loss`)
> and show the top 5 with text and label.

## Verify
`lab check 2.6` · Read two suspects that weren't planted. Is their label right?
Then pick in `notes.md`: **A** drop the suspects (default) · **B** down-weight them · **C** send them for relabelling.

## Takeaway
Use models to debug data. The examples a model finds hardest are often the wrong ones.

## Go deeper
label noise · confident learning (cleanlab) · cross-validation ·
*Ask Claude: "Why not train once on everything and use the training loss?"*
