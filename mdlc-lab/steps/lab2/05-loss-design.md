# 2.5 Loss design (15 min)

## Why this matters
The loss is an **average over reviews**. With 62% positive and 12% neutral, getting neutral wrong
barely moves it, so the model learns to rarely say neutral. **The loss is where you tell the model
what you care about.**

**Class weights** multiply each review's loss by a weight for its true class:
- inverse frequency: rare classes weigh more, and every class counts equally in total;
- square root of that: a gentler middle ground.

The same tool encodes **business cost**: if missing an angry customer is the expensive mistake,
weigh negative higher.

## Predict first (notes.md)
With inverse-frequency weights, what happens to neutral recall? To positive recall? To macro-F1?

## Prompt
> Write `src/train2.py` with a reusable function `train(train_df, dev_df, class_weights=None, seed=0)`
> that trains the same network as lab 1: TF-IDF (20,000 terms, words and two-word phrases) fitted on
> `train_df` only; Linear → 32 → ReLU → Linear → 3; Adam, learning rate 0.001; batches of 64; 6 epochs;
> keep the epoch with the best dev macro-F1. It returns the model and vectorizer. Each run should take
> under a minute on a laptop CPU; if it's much slower, find out why (hint: convert only one batch at a
> time from sparse to dense). Steps 2.6 and 2.8 reuse this function.
>
> Then, on `outputs/train.csv` and `outputs/dev.csv`, train it three times with class-weighted
> cross-entropy: no weights, square-root inverse frequency, inverse frequency. Record dev macro-F1 and
> recall per class in `outputs/class_weights.csv` with columns
> `weighting,dev_macro_f1,recall_negative,recall_neutral,recall_positive`. Show me the weights you
> computed, and explain what the weighted loss formula does.

## Verify
`lab check 2.5`
- **Decide** in notes.md: for "route unhappy customers to a support agent", which weighting would
  you ship, and why? There's no single right answer. It's a product decision.
- Ask Claude: *"Besides class weights, what else handles imbalance? When would you use each?"*

## ML thinking
**A metric and a loss encode values. Choose them deliberately, with the people who own the outcome.**

## Go deeper
class imbalance · weighted cross-entropy · focal loss · resampling (oversampling, SMOTE) · cost-sensitive learning · decision thresholds
