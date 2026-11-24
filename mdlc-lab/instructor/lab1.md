# Lab 1: Train a classifier (instructor notes)

Steps: `lab steps`. Cards: `steps/lab1/`. General format: [GUIDE.md](GUIDE.md).

| Step | Time | Concept to introduce | Debrief question |
|---|---|---|---|
| 1.0 How we work | 10 | the seven habits; who does what | — |
| 1.1 Meet the data | 10 | supervised learning, labels, development vs evaluation data, imbalance | Hands up: A, B or C for "always positive"? (B: 62%) |
| 1.2 Baselines | 15 | baseline; precision, recall, F1, macro-F1; confusion matrix | Which metric did you pick, and why? |
| 1.3 Text to numbers | 10 | features, TF-IDF, sparse vectors; **fit on train only** | What did your agent fit the vectorizer on? Did anyone's get it wrong? |
| 1.4 Training | 25 | network, loss, gradient, optimizer, epoch; overfitting; early stopping | Show two people's curves. Where did dev loss turn? |
| 1.5 Experiments | 15 | hyperparameters, seeds and noise, one change at a time | Was any change bigger than the seed-to-seed difference? |
| 1.6 Evaluate | 10 | golden set, slices, held-out discipline | Held-out vs dev: A, B or C? Then point at neutral F1: "lab 2 explains this." |
| 1.7 Break it | 10 | error analysis; limits of bag-of-words | Best failure in the room. Then the reveal (600 flipped labels) |

## Reference numbers (reference solution, seed 0)
| | value |
|---|---|
| majority baseline, dev | accuracy 0.621, macro-F1 0.255 |
| keyword baseline, dev | accuracy 0.680, macro-F1 0.503 |
| network (hidden 32), dev | accuracy 0.816, macro-F1 0.680; dev loss lowest at epoch 3 of 6 |
| golden (381 items, draft labels) | accuracy 0.858, macro-F1 0.719; weakest slices: sarcasm, short |
| held-out | accuracy 0.836, macro-F1 0.695 |

Participants' numbers will vary by a few points depending on their decisions; anything with dev
accuracy above ~0.92 deserves a leakage hunt (a good teaching moment if it happens).
