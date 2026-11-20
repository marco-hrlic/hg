# Lab 2: instructor notes

Lab 2 is the dense one: theory-heavy, with 6 short TODOs. To fit in 2h, keep exercises tight and
present the theory tables rather than reading them out. The solutions notebook runs in ~4 min on
4 CPU cores; the training cells in sections 4 and 6 take ~1 and ~2.5 min, so start them, then talk.

## Before the session
- `make data` and the lab 1 notebook run on every machine (lab 2 loads the saved lab 1 model; if
  it's missing, it retrains the same recipe automatically).
- Ideally, finish the human review of the golden labels first (`golden_labels/README.md`).
  Otherwise lab 2 uses the agreed LLM draft labels and says so.

## Timing

| Block | Time | TODOs | Notes |
|---|---|---|---|
| 1. Data engineering | 35 min | 1 parse labels, 2 exact dedup, 3 stratified split | Scores come from the answer key. 1e (theory: development vs evaluation data, leakage, imbalance) is the core 10 minutes. Point out the cleaned set equals lab 1's prepared data. |
| 2. Labels and golden set | 15 min | label 10 reviews | The 3-star crosstab is the moment: most 3-star reviews aren't neutral. |
| 3. The maths | 30 min | 4 softmax, 5 cross-entropy, 6 gradient | Draw z → softmax → loss on a whiteboard. Run the learning-rate plot live. The NumPy model matching lab 1's network is a good discussion. |
| 4. Loss design and label noise | 15 min | | Start the class-weights cell, then explain the weighted loss while it runs. |
| 5. Synthetic data | 10 min | | Demo. Show `synthetic/PROMPT.md`. Stress the four rules. |
| 6. Retrain and compare | 10 min | | Start the comparison cell before the synthetic-data block's discussion ends. |
| Wrap-up | 5 min | | The MDLC loop table. |

## Reference numbers (one run, 4-core CPU)

| | value |
|---|---|
| NumPy linear model (lr 10, 1500 steps), dev | accuracy 0.817, macro-F1 0.681 |
| Dev neutral recall, class weights none / sqrt / inverse | 0.30 / 0.50 / 0.62 |
| Dev macro-F1, class weights none / sqrt / inverse | 0.680 / 0.707 / 0.700 |
| Flipped-label suspects (top 600 by out-of-fold loss) | 296 of the 600 planted flips caught |

Golden macro-F1 (381 items, draft labels), mean of 3 seeds:

| model | golden macro-F1 |
|---|---|
| lab 1 model | 0.719 |
| + class weights (sqrt) | 0.737 |
| + class weights, noise removed | **0.747** (chosen) |
| + class weights, noise removed, synthetic | 0.741 |

Held-out macro-F1: lab 1 0.695 → lab 2 0.701.

A separate 3-seed experiment, with selection on dev macro-F1 throughout, gave the same picture:
lab 1 recipe 0.714; noise removed 0.733; sqrt weights 0.732; inverse weights 0.700 (too
aggressive); synthetic only 0.720; all three combined 0.753 ± 0.012. Synthetic data's effect is
small and within run-to-run noise. Present that honestly: it's the lesson of rule 3.

## Things people ask
- *Why is the held-out gain smaller than the golden gain?* Held-out labels are star-derived, so the
  3-star proxy problem penalises a model that reads 3-star reviews correctly. The golden set is the
  better yardstick, which is why section 6 says so before the held-out cell.
- *Wasn't choosing the final model on golden a form of tuning?* Yes, a mild one, and the notebook
  says so. That's why the held-out set is opened only after the choice.
- *The suspects list has non-planted items.* Mostly 3-star reviews that aren't neutral: real
  label errors that nobody planted.
