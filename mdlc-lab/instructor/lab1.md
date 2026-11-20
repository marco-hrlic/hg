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
| 2. The data | 15 min | Development data (train, dev) vs evaluation data (held-out, golden), and class balance. Stress *why* the held-out set is touched once. |
| 3. The bar to beat | 10 min | The majority-baseline confusion matrix is the moment accuracy stops being trusted. |
| 4. Text → numbers | 15 min | Point out the vectorizer is fitted on train only. It's the first leakage rule they meet. |
| 5. The model | 10 min | Keep it concrete: 640k weights, all random right now. |
| 6. Training | 30 min | TODO 1 is the only required coding exercise. Leave time for two experiments. |
| 7. How good is it? | 15 min | Neutral recall ≈ 0.28 is the discussion. Run the held-out set once, together. |
| 8. Play | 15 min | Let people type their own reviews. The learned word lists usually get a reaction. |
| 9. Break it | 15 min | Collect the best failures from the room. They're good material for lab 2's synthetic-data block. |
| Wrap-up | 5 min | The reveal: ~600 flipped labels are still in the data they trained on. |

## Reference numbers (seed 0, 4-core CPU)

| model | dev accuracy | dev macro-F1 |
|---|---|---|
| majority | 0.621 | 0.255 |
| lexicon (with negation) | 0.680 | 0.503 |
| network, hidden=32, best of 6 epochs | 0.816 | 0.680 |

Held-out: accuracy 0.836, macro-F1 0.695. Golden (381 items, LLM draft labels): accuracy 0.858,
macro-F1 0.719. Dev per-class recall: negative 0.79, neutral 0.28, positive 0.93.

The golden score is *higher* than dev even though the golden set is full of hard cases. Its labels
were checked against the text, while dev labels come from star ratings, and many 3-star reviews
aren't neutral at all. If someone asks, that's a perfect lead-in to lab 2.

Typical failures in section 9: sarcasm ("Oh great, another 45 minute wait… Fantastic." → positive),
"Not bad at all" → neutral, comparisons.

## If something goes wrong
- Results differ slightly between machines (thread scheduling in PyTorch). Differences of ±0.01 are normal.
- A participant stuck on TODO 1 can copy the five lines from `solutions/`. Everything after it depends on it.
