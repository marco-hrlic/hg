# Lab 2: Open the box (instructor notes)

Steps: `lab steps`. Cards: `steps/lab2/`. General format: [GUIDE.md](GUIDE.md). Theory-heavier
than lab 1: spend the instructor minutes on the concepts, and let agents do the typing.

| Step | Time | Concept to introduce | Debrief question |
|---|---|---|---|
| 2.1 Audit | 15 | data quality, near-duplicates, target leakage | Who found the sorted file? The leaky column? |
| 2.2 Clean & split | 15 | cleaning rules as decisions; stratified split; versioning; group/time splits (theory) | Repair or drop "5 stars"? Defend it. |
| 2.3 Labels & golden set | 15 | proxy labels, agreement (kappa), guidelines, golden-set properties, LLM raters | The 3-star column, and the relabelling result: ~1,450 wrong training labels |
| 2.4 The maths | 20 | linear model, softmax, cross-entropy, gradient, gradient descent; learning rate | Did the hand-written linear model match lab 1's network? |
| 2.5 Loss design | 15 | class weights, cost-sensitive loss; metrics encode values | Which weighting would you ship? (No right answer) |
| 2.6 Label noise | 10 | out-of-fold loss, cross-validation, data-centric debugging | What were the non-planted suspects? |
| 2.7 More data | 15 | learning curves; the cost of data | Is the curve still rising? Worth another 100k? |
| 2.8 Synthetic data | 5 | *demo*: the four rules; contamination | Would you believe a customer wrote these? |
| 2.9 Retrain & compare | 15 | controlled comparison, seeds, **paired tests and statistical power** | Why can held-out see the gain and golden can't? |

Steps 2.5–2.7 and 2.9 train models in the background (2.7 and 2.9 take ~4 and ~7 minutes on 4
cores). Introduce the next concept while they run.

## Reference numbers (reference solution, 4-core CPU)
| | value |
|---|---|
| audit (2.1) | all row-level problems caught except label noise (0 of 600) |
| clean + split (2.2) | 17,892 train / 1,988 dev, identical to lab 1's prepared data |
| stars vs golden (2.3) | of 59 three-star golden items: 17 neutral, 29 positive, 13 negative |
| 3-star relabelling (2.3) | of 2,316 three-star reviews: 37% positive, 36% neutral, 25% negative, 1% ambiguous |
| NumPy linear model (2.4) | dev accuracy ≈ 0.82, macro-F1 ≈ 0.68 (matches lab 1's network) |
| class weights none / sqrt / inverse (2.5) | neutral recall 0.28 / 0.45 / 0.59; macro-F1 0.680 / 0.704 / 0.700 |
| suspects (2.6) | ~300 of 600 are planted flips; most of the rest are 3-star reviews |
| learning curve (2.7) | dev macro-F1 0.680 / 0.699 / 0.707 at 18k / 50k / 100k reviews (25 / 66 / 134 s per run) |
| 2.9, dev, paired vs candidate 0 (2 seeds) | A (sqrt weights) **+0.024 [+0.007, +0.043]**; B (+ suspects removed) +0.015 [−0.000, +0.031]; D (+ more data) **+0.047 [+0.026, +0.070]** |
| 2.9, held-out, lab 2 (D) vs lab 1 | **+0.048 [+0.034, +0.063]**, significant (0.695 → ~0.74) |
| 2.9, golden, same two models | +0.031 [−0.025, +0.090]: not distinguishable |

**How to present 2.9.** The lab 2 model is significantly better than lab 1 on held-out, driven
mostly by more data and class weights. Removing suspects is borderline on its own. The golden set
can't confirm the same gain because 381 items (31 neutral) are too few to detect ~3 points. That's
the statistical-power lesson, and the reason step 2.9 compares on dev and held-out. The golden
set's job is *where* the model fails (per slice), not *whether* a few-point gain is real.

**Why we didn't train on the relabelled 3-star reviews.** We tried: with corrected labels only ~4%
of training reviews are neutral, the model nearly stops predicting neutral, and golden macro-F1
*fell* by 0.08 (significant). With class weights it only got back to level. Correct labels alone
don't fix a class the model barely sees, and fixing labels changes the class balance too. It's a good
answer if someone asks "why not just fix the labels?"

## Why we don't plant more label noise to make 2.8 "work"
We tested whether a higher flip rate would make "remove the suspects" produce a clear gain
(candidate 0 vs B, no class weights, 3 seeds each, macro-F1):

| flipped labels in train | golden 0 → B | held-out 0 → B |
|---|---|---|
| 3% (as shipped) | 0.733 → 0.707 | 0.688 → 0.691 |
| 10% | 0.694 → 0.737 | 0.648 → 0.649 |
| 20% | 0.687 → 0.662 | 0.603 → 0.572 |

The direction isn't consistent: about half of the removed suspects are genuinely hard but correctly
labelled reviews, and dropping them costs as much as dropping the flips gains. So the dataset stays as
it is, and 2.8 teaches controlled comparison and evaluation power rather than a guaranteed win. If
someone asks "so how *would* you get a real gain?", the honest answer from step 2.3 is: fix the
proxy labels (relabel the 3-star reviews against the guideline), and grow the golden set so a
2-point gain is measurable.
