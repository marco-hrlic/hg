# 2.8 Retrain and compare (10 min)

## Why this matters
Time to see whether better data and a better loss beat lab 1. Compare **candidates on the golden
set**: checked labels, hard cases included. Because results vary between runs, train each
candidate with **3 seeds** and compare means. Choose one, **then** use the held-out set, once.

## Predict first (notes.md)
Which change will help most: class weights, removing suspects, or synthetic data?

## Prompt
> Train these candidates on `outputs/train.csv` (dev for early stopping), each with seeds 0, 1, 2:
> A) class weights [your choice from 2.5]; B) A + suspects removed; C) B + `outputs/synthetic.csv`.
> For each seed, write golden predictions to `outputs/golden_<candidate>_s<seed>.csv`. Don't submit;
> I'll do that. Then write `outputs/golden_lab2.csv` and `outputs/heldout_lab2.csv` for the
> candidate I choose.

Then you submit each candidate's golden predictions (`lab submit golden FILE --name C-s0` …),
compare the means against `lab1` in `lab log`, pick one, and finally:
```
lab submit heldout outputs/heldout_lab2.csv --name lab2
lab check 2.8
```

## Verify
- Is the gain bigger than the seed-to-seed spread? Did synthetic data earn its place?
- Choosing on golden means golden is now slightly optimistic for the winner. That's why held-out
  comes last.

## Wrap-up: the loop
frame → data → labels & golden set → train → evaluate → ship with a **model card** → monitor for
**drift** → back to data. You've now touched every box. Pick one "go deeper" topic from today
and ask an agent to teach it to you properly.

## Go deeper
model selection · model cards · data drift and concept drift · ML monitoring · MLOps
