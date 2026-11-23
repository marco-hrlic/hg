# Lab 2: Open the box (instructor notes)

Steps: `lab steps`. Cards: `steps/lab2/`. General format: [GUIDE.md](GUIDE.md). Theory-heavier
than lab 1: spend the instructor minutes on the concepts, and let agents do the typing.

| Step | Time | Concept to introduce | Debrief question |
|---|---|---|---|
| 2.1 Audit | 20 | data quality, near-duplicates, target leakage | Who found the sorted file? The leaky column? |
| 2.2 Clean & split | 15 | cleaning rules as decisions; stratified split; versioning; group/time splits (theory) | Repair or drop "5 stars"? Defend it. |
| 2.3 Labels & golden set | 15 | proxy labels, agreement (kappa), guidelines, golden-set properties, LLM raters | The 3-star column: what does it mean for lab 1's model? |
| 2.4 The maths | 25 | linear model, softmax, cross-entropy, gradient, gradient descent; learning rate | Did the hand-written linear model match lab 1's network? |
| 2.5 Loss design | 15 | class weights, cost-sensitive loss; metrics encode values | Which weighting would you ship? (No right answer) |
| 2.6 Label noise | 10 | out-of-fold loss, cross-validation, data-centric debugging | What were the non-planted suspects? |
| 2.7 Synthetic data | 10 | the four rules; contamination | Would you believe a customer wrote these? |
| 2.8 Retrain & compare | 10 | seeds, model selection, held-out once; the MDLC loop | Was synthetic data worth it? |

Steps 2.5, 2.6 and 2.8 each train several models (1–3 min on a laptop). Introduce the next
concept while they run.

## Reference numbers (reference solution)
| | value |
|---|---|
| audit (2.1) | all row-level problems caught except label noise (0 of 600) |
| clean + split (2.2) | 17,892 train / 1,988 dev, identical to lab 1's prepared data |
| stars vs golden (2.3) | of 59 three-star golden items: 17 neutral, 29 positive, 13 negative |
| NumPy linear model (2.4) | dev accuracy ≈ 0.82, macro-F1 ≈ 0.68 (matches lab 1's network) |
| class weights none / sqrt / inverse (2.5) | neutral recall 0.28 / 0.45 / 0.59; macro-F1 0.680 / 0.704 / 0.700 |
| suspects (2.6) | 308 of 600 are planted flips; 204 of the other 292 are 3-star reviews |
| final (2.8): sqrt weights + suspects removed | golden macro-F1 0.719 → 0.736; held-out 0.695 → 0.719 |

Across 3 seeds, synthetic data's effect was within noise (±0.01). Present that honestly: it's
the point of rule 3 on the card.
