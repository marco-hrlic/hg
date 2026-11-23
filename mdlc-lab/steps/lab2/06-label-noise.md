# 2.6 Find wrong labels with a model (10 min)

## Why this matters
~600 training labels are flipped, and no row-level rule can find them. A trained model can: it
will be very **surprised** (high loss) by a review whose label contradicts everything it learned.
The catch: a model scores low loss on reviews it *trained* on, because it memorises them. So
score each review with a model that **didn't see it**: train on half, score the other half, swap.
That's **out-of-fold** prediction, the idea behind **cross-validation**.

## Predict first (notes.md)
Of the 600 most surprising reviews, how many will be planted flips?

## Prompt
> Find likely mislabelled reviews in `outputs/train.csv` with 2-fold out-of-fold loss: train the
> network on one half, compute each review's cross-entropy loss on the other half, swap. Save the
> 600 highest-loss reviews to `outputs/suspects.csv` (`review_id,loss`) and show me the top 10 with
> their text and label. Explain why we didn't just train once on everything and use training loss.

## Verify
`lab check 2.6`
- Read the suspects that *weren't* planted. Are they correctly labelled?
- **Decide** in notes.md: drop the suspects, down-weight them, or send them to people to relabel?

## ML thinking
**Use models to debug data.** The examples a model finds hardest are often the ones that are wrong.

## Go deeper
label noise · confident learning (cleanlab) · cross-validation · out-of-fold predictions · data-centric AI
