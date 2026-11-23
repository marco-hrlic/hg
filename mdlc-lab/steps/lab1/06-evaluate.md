# 1.6 Evaluate: golden and held-out (10 min)

## Why this matters
We used dev to make every decision, so the dev score is a little optimistic. Our choices fit it.
The **evaluation sets** give the honest verdict:
- **golden set** (~380 reviews): labels assigned against a written guideline by two independent
  raters, with disagreements sent to a person (lab 2 shows how),
  **deliberately rich in hard cases** (negation, sarcasm, mixed opinions), each tagged with its
  **slice**. Answers "where does it fail?"
- **held-out set** (5,000 reviews): realistic mix, labels from stars. Answers "how good in
  production?" **One submission per lab.** Every look-then-change makes it less honest.

## Predict first (notes.md)
Held-out macro-F1 vs your dev macro-F1: higher, lower, the same? Which golden slice will be worst?

## Prompt
> Using `outputs/model.pt` and `outputs/vectorizer.pkl` from step 1.4, write `src/predict.py`
> that predicts labels for `data/eval/golden_texts.csv` and `data/eval/heldout_texts.csv` and saves
> `outputs/golden_lab1.csv` and `outputs/heldout_lab1.csv` with columns `id,label` (`id` = the file's
> `item_id` or `review_id`). Don't submit; I'll do that.

Then you run:
```
lab submit golden  outputs/golden_lab1.csv  --name lab1
lab submit heldout outputs/heldout_lab1.csv --name lab1
lab check 1.6
```

## Verify
- Was your held-out prediction right? Why might dev and held-out differ?
- Which slices are weakest? Note them; they're the target for lab 2.

## ML thinking
**Decide, then evaluate once.** Evaluation data you've tuned on can't tell you the truth any more.

## Go deeper
held-out test set · golden dataset · slice-based evaluation · overfitting to the test set · benchmark contamination
