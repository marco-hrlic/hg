# 1.5 Experiments (15 min)
**Goal:** three experiments, one change each, compared against noise.

## Why this matters
**Hyperparameters** are settings you choose (hidden size, learning rate). The random **seed** alone
moves the score; a difference smaller than that isn't real.

## Predict
For each experiment: will dev macro-F1 be **A** higher, **B** about the same, or **C** lower than now?
Tell Claude before each run; it goes in the `predicted` column.

## Prompt
> Make `src/train.py` configurable from the command line (hidden size, learning rate, epochs, seed,
> output tag). Experiment runs write under `outputs/experiments/`, never over step 1.4's files.
> Run, one change at a time from our current settings, and log each to `outputs/experiments.csv`
> (`experiment,change,predicted,dev_macro_f1,notes`):
> 1. same settings, another seed (measures the noise);
> 2. [DECISION] **A** hidden size 256 (default) · **B** hidden size 8 · **C** 2 epochs;
> 3. [DECISION] **A** learning rate ×50 (default) · **B** learning rate ÷10 · **C** batch size 512.

## Verify
`lab check 1.5` · Which changes were bigger than the seed-to-seed difference?

## Takeaway
One change at a time, a prediction before each, compared against noise.

## Go deeper
hyperparameter tuning · seeds and variance · experiment tracking ·
*Ask Claude: "Why did the large learning rate behave like that?"*
