# Lab 1: instructor notes

## Before the session
- Everyone runs `make install && make data` beforehand. The ~200 MB download and the PyTorch install
  are the only steps that can fail on the day. For laptops, the CPU build of PyTorch is much smaller:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
- Run `make check` on the projector machine. Lab 1 runs end to end in about a minute on 4 CPU cores.

## Timing

| Block | Time | Notes |
|---|---|---|
| 1. The task | 10 min | Get the room to name the expensive mistake (missing an angry customer). It comes back in section 7. |
| 2. The data | 15 min | Splits and class balance. Stress *why* the test set is touched once. |
| 3. The bar to beat | 10 min | The majority-baseline confusion matrix is the moment accuracy stops being trusted. |
| 4. Text → numbers | 15 min | Point out the vectorizer is fitted on train only. It's the first leakage rule they meet. |
| 5. The model | 10 min | Keep it concrete: 640k weights, all random right now. |
| 6. Training | 30 min | TODO 1 is the only required coding exercise. Leave time for two experiments. |
| 7. How good is it? | 15 min | Neutral recall ≈ 0.19 is the discussion. Run the test set once, together. |
| 8. Play | 15 min | Let people type their own reviews. The learned word lists usually get a reaction. |
| 9. Break it | 15 min | Collect the best failures from the room. They're good material for lab 2's synthetic-data block. |
| Wrap-up | 5 min | The reveal: ~600 flipped labels are still in the data they trained on. |

## Reference numbers (seed 0, 4-core CPU)

| model | val accuracy | val macro-F1 |
|---|---|---|
| majority | 0.621 | 0.255 |
| lexicon (with negation) | 0.692 | 0.502 |
| network, hidden=32, best of 6 epochs (epoch 3) | 0.811 | 0.648 |

Test set: accuracy 0.810, macro-F1 0.654. Per-class recall: negative 0.78, neutral 0.19, positive 0.94.

Typical failures in section 9: sarcasm ("Oh great, another 45 minute wait… Fantastic." → positive 0.78),
"Not bad at all" → neutral, comparisons. On the hard-case pool, the sarcasm and mixed-signal slices score lowest.

## If something goes wrong
- Results differ slightly between machines (thread scheduling in PyTorch). Differences of ±0.01 are normal.
- A participant stuck on TODO 1 can copy the five lines from `solutions/`. Everything after it depends on it.
