# 2.5 Loss design (15 min)
**Goal:** a reusable `train()` function, and three class weightings compared.

## Why this matters
The loss averages over reviews, so rare neutral barely counts. **Class weights** multiply each
review's loss by a weight for its true class: that's how you tell the model what you care about.

## Predict
With inverse-frequency weights, neutral recall will…  **A** fall  **B** stay about the same  **C** roughly double

## Prompt
> Write `src/train2.py` with `train(train_df, dev_df, class_weights=None, seed=0)` returning
> `(model, vectorizer)`: TF-IDF (20,000 terms, words and two-word phrases) fitted on `train_df` only;
> Linear → 32 → ReLU → Linear → 3; Adam 0.001; batches of 64; 6 epochs; keep the epoch with the best
> dev macro-F1. Densify one batch at a time so a run takes under a minute. Steps 2.6, 2.7 and 2.9 reuse it.
> On `outputs/train.csv` / `outputs/dev.csv`, train with no weights, square-root inverse frequency and
> inverse frequency. Save `outputs/class_weights.csv`
> (`weighting,dev_macro_f1,recall_negative,recall_neutral,recall_positive`) and show the weights.

## Verify
`lab check 2.5` · Then pick in `notes.md`: to route unhappy customers to support, ship
**A** none, **B** sqrt or **C** inverse? (A product decision: there's no single right answer.)

## Takeaway
Weights don't make a model smarter. They move where its mistakes land.

## Go deeper
weighted cross-entropy · focal loss · resampling · decision thresholds ·
*Ask Claude: "Besides class weights, what else handles imbalance?"*
