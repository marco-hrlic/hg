# 2.9 Retrain and compare (15 min)
**Goal:** a fair, significance-tested answer to "is lab 2's model better than lab 1's?"

## Why this matters
Fair comparison: **same code and seeds** for every candidate, **two seeds** each, and a **paired
bootstrap** on the same items. Choose on dev, confirm once on held-out.

## Predict
Lab 2's best model vs lab 1 on held-out macro-F1?  **A** no significant difference  **B** +0.01 to +0.03  **C** +0.04 or more

## Prompt
> Write `src/compare.py` using `train()` from `src/train2.py`, early-stopping on `outputs/dev.csv`:
> - **0**: `outputs/train.csv` as is (lab 1's recipe)
> - **A**: + class weights. [DECISION: weights] your choice from 2.5 (default: sqrt)
> - **B**: A + the reviews in `outputs/suspects.csv` removed
> - **C**: B + `outputs/synthetic.csv`
> - **D**: B + all of `data/lab2/more_reviews.csv`
>
> Seeds 0 and 1 each, in the background. Write `outputs/dev_<candidate>_s<seed>.csv` for every run, and
> `outputs/golden_<candidate>_s0.csv` and `outputs/heldout_<candidate>_s0.csv` for seed 0. Don't submit.

Then compare on dev (`lab compare A 0`, `B 0`, `C B`, `D 0`), pick the winner, and submit it once:
```
lab submit heldout outputs/heldout_<winner>_s0.csv --name lab2
lab submit golden  outputs/golden_<winner>_s0.csv  --name lab2
lab check 2.9
```

## Verify
Which candidates beat 0 significantly? Then run `lab compare lab2 lab1 --on golden`: why can't the
golden set tell them apart?

## Takeaway
frame → data → labels → train → evaluate → ship with a **model card** → monitor **drift** → back to
data. You've touched every box. Pick one "go deeper" topic and ask an agent to teach it to you.

## Go deeper
statistical power · paired bootstrap · ablation study · model cards · data drift
