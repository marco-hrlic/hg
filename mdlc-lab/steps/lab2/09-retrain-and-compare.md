# 2.9 Retrain and compare (10 min)

## Why this matters
Time to find out whether everything we learned adds up to a better model than lab 1's. Three rules
make the comparison fair:
1. **Same code, same seeds.** Retrain lab 1's recipe *in the same script* as **candidate 0**.
2. **Several seeds.** Results vary between runs; train each candidate twice and average.
3. **A paired test on a big enough set.** Score every candidate on the *same* items and bootstrap the
   *difference*. Our golden set (381 items) is great for finding *where* a model fails, but too small
   to detect a 3-point gain. Dev (~2,000 items) and held-out (5,000) can.

Choose on dev. Confirm once on held-out, against your lab 1 submission.

## Predict first (notes.md)
Which candidate wins? By how much on held-out macro-F1? Will the difference be significant?

## Prompt
> Write `src/compare.py` using `train()` from `src/train2.py`. Candidates, all early-stopping on
> `outputs/dev.csv`:
> - **0**: `outputs/train.csv` as is (the lab 1 recipe);
> - **A**: class weights [DECISION: your choice from 2.5];
> - **B**: A + the suspects in `outputs/suspects.csv` removed;
> - **D**: B + all of `data/lab2/more_reviews.csv`.
>
> Train each with seeds 0 and 1 (8 runs; D is slow, so run everything in the background and tell me
> when it's done). For every run write dev predictions to `outputs/dev_<candidate>_s<seed>.csv`; for
> seed 0 also write `outputs/golden_<candidate>_s0.csv` and `outputs/heldout_<candidate>_s0.csv`.
> Don't submit anything; I'll do that.

Then compare on dev, paired:
```
lab compare A 0
lab compare B 0
lab compare D 0
```
Pick the winner. Submit its held-out predictions **once**, and its golden predictions for the slice view:
```
lab submit heldout outputs/heldout_<winner>_s0.csv --name lab2
lab submit golden  outputs/golden_<winner>_s0.csv  --name lab2
lab check 2.9
```

## Verify
- Which candidates are **significantly** better than 0 on dev? Which aren't?
- Is the lab 2 model significantly better than lab 1 on held-out? (`lab check 2.9` runs the paired test.)
- Run `lab compare lab2 lab1 --on golden`. Same models, but the golden set can't tell them
  apart. Why? Write down how big a golden set you'd need.

## Wrap-up: the loop
frame → data → labels & golden set → train → evaluate → ship with a **model card** → monitor for
**drift** → back to data. You've now touched every box. Pick one "go deeper" topic from today
and ask an agent to teach it to you properly.

## Go deeper
statistical power · paired bootstrap · controlled experiments · ablation study · model cards · data drift · MLOps
