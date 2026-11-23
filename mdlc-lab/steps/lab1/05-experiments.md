# 1.5 Experiments (15 min)

## Why this matters
Settings you choose rather than learn (hidden size, learning rate, epochs, batch size) are
**hyperparameters**. Finding good ones is experimentation, and it follows the scientific method:
one change at a time, a prediction before each run, everything recorded.

Results also vary from run to run, because the starting weights and batch order are random (the
**seed**). A difference smaller than that run-to-run variation isn't a real difference.

## Predict first
Before **each** experiment, write the prediction in the `predicted` column (see below).

## Prompt
> Make `src/train.py` configurable from the command line (hidden size, learning rate, epochs, seed,
> and an output tag). Experiment runs must **not** overwrite `outputs/dev_network.csv`, `model.pt` or
> `vectorizer.pkl` from step 1.4: write them under `outputs/experiments/` instead. Keep each run
> under ~2 minutes (hidden size ≤ 256, ≤ 8 epochs); run longer ones in the background.
> Then run these experiments, one change at a time from our current settings, and append one row per
> run to `outputs/experiments.csv` with columns `experiment,change,predicted,dev_macro_f1,notes`.
> I'll give you the prediction for each before you run it:
> 1. same settings, different seed (to measure the noise);
> 2. [DECISION: your first change, e.g. a much bigger hidden size];
> 3. [DECISION: your second change, e.g. learning rate ×50].

## Verify
`lab check 1.5`
- Which differences are bigger than the seed-to-seed difference from experiment 1?
- Ask Claude: *"Why did the large learning rate behave like that?"*

## ML thinking
**One change at a time, a prediction before each, and compare against noise.** An experiment
log beats memory.

## Go deeper
hyperparameter tuning · random seed and variance · grid search vs random search · experiment tracking (MLflow, Weights & Biases)
