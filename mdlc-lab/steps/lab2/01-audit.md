# 2.1 Audit the raw data (15 min)

## Why this matters
Lab 1's "prepared" data started as `data/lab2/reviews_raw.csv`: a raw export of 20,700 reviews.
Real data always arrives like this. **Data quality problems cost more than modelling problems**,
and they're invisible in an average score. Before cleaning anything, **audit**: find and count the
problems, then decide what to do about each.

Kinds of problem to look for:
- **row level**: invalid labels, empty or placeholder texts, markup, **duplicates** (exact, and
  **near-duplicates**: edited or truncated copies), the same text with **conflicting labels**;
- **dataset level**: how the file is ordered, columns that shouldn't be used as features
  (**target leakage**: a column that gives the answer away), class balance.

## Predict first (notes.md)
How many different kinds of problem will you find? Which one would hurt the model most?

## Prompt
> Audit `data/lab2/reviews_raw.csv` (read every column as text first). Write `src/audit.py` that
> looks for row-level problems: invalid star values, empty/placeholder texts, HTML, exact duplicates
> after normalising case and punctuation, duplicates with conflicting ratings, and near-duplicates
> (explain how you detect those efficiently). Write every flagged row to `outputs/issues.csv`
> (`review_id,issue`). Then look at the dataset as a whole (row order, every column, label balance)
> and tell me what else is wrong. Don't fix anything yet.

Write the dataset-level problems you found in notes.md, **then** run the check.

## Verify
`lab check 2.1` scores your findings against the list of problems that were planted, then
reveals the dataset-level ones.
- Ask Claude: *"How does near-duplicate detection with MinHash work, in three sentences?"*
- Ask: *"If we kept `stars_display` as a feature, what would the dev score look like, and why
  would it be useless in production?"*

## ML thinking
**Look before you clean, and count before you fix.**

## Go deeper
data profiling · near-duplicate detection (MinHash, LSH) · target leakage · data validation (Great Expectations, pandera)
