# 2.8 Retrain and compare (10 min)

## Why this matters
Time to see whether better data and a better loss beat lab 1. Two rules make the comparison fair:
- **Same code, same seeds.** Retrain lab 1's recipe *in the same script* as **candidate 0**. Comparing
  against a model built by different code would mix "better data and loss" with "different
  implementation".
- **Seeds.** Results vary between runs, so train each candidate with **3 seeds** and compare means.
  A difference smaller than the seed-to-seed spread isn't a difference.

Compare candidates on the **golden set** (checked labels, hard cases). Choose one, **then** use the
held-out set, once.

## Predict first (notes.md)
Which change will help most: class weights, removing suspects, or synthetic data? By how much?

## Prompt
> Write `src/compare.py` using `train()` from `src/train2.py`. Candidates, all trained on
> `outputs/train.csv` with `outputs/dev.csv` for early stopping:
> - **0**: no changes (the lab 1 recipe);
> - **A**: class weights [DECISION: your choice from 2.5];
> - **B**: A + the suspects from `outputs/suspects.csv` removed;
> - **C**: B + `outputs/synthetic.csv` added to train.
>
> Train each with seeds 0, 1 and 2 (12 runs: run it in the background and tell me when it's done).
> For each run, write golden predictions to `outputs/golden_<candidate>_s<seed>.csv`; for seed 0 also
> write held-out predictions to `outputs/heldout_<candidate>_s0.csv`. Don't submit anything; I'll do that.

Then you submit every golden file, naming each `<candidate>-s<seed>`:
```
for f in outputs/golden_*_s*.csv; do n=$(basename $f .csv); n=${n#golden_}; lab submit golden $f --name ${n/_s/-s}; done
lab check 2.8          # groups the seeds per candidate and compares each with candidate 0
```
Choose a candidate, then submit its seed-0 held-out file **once**:
```
lab submit heldout outputs/heldout_<candidate>_s0.csv --name lab2-<candidate>
```

## Verify
`lab check 2.8`
- Which candidates are **clearly** better than 0, and which are within noise?
- If nothing clearly beats candidate 0, that's a result too. Write down what you'd try next.
- Choosing on golden makes golden slightly optimistic for the winner. That's why held-out comes last.

## Wrap-up: the loop
frame → data → labels & golden set → train → evaluate → ship with a **model card** → monitor for
**drift** → back to data. You've now touched every box. Pick one "go deeper" topic from today
and ask an agent to teach it to you properly.

## Go deeper
controlled experiments · ablation study · model selection · model cards · data drift and concept drift · ML monitoring · MLOps
