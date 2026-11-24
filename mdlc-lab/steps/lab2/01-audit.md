# 2.1 Audit the raw data (15 min)
**Goal:** a counted list of what's wrong with the raw export, before fixing anything.

## Why this matters
Lab 1's data came from `data/lab2/reviews_raw.csv` (20,700 rows). Problems hide at **row level**
(bad labels, junk text, **duplicates**) and **dataset level** (file order, **target leakage**: a column
that gives the answer away).

## Predict
How many *kinds* of problem are planted?  **A** 2–3  **B** 4–6  **C** 7 or more

## Prompt
> Write `src/audit.py` for `data/lab2/reviews_raw.csv` (read every column as text). Flag invalid
> stars, empty/placeholder texts, HTML and escaped characters, exact duplicates (after normalising
> case and punctuation), duplicates with conflicting stars, and near-duplicates. Write the flags to
> `outputs/issues.csv` (`review_id,issue`). Then check row order, every column and the label balance,
> and list dataset-level problems in five lines or fewer. Don't fix anything.

Write the dataset-level problems in `notes.md`, **then** run the check.

## Verify
`lab check 2.1` scores your flags against the planted problems and reveals the dataset-level ones.

## Takeaway
Look before you clean, count before you fix.

## Go deeper
data profiling · near-duplicates (MinHash, LSH) · target leakage · data validation ·
*Ask Claude: "How does MinHash find near-duplicates, in three sentences?"* ·
*"What would `stars_display` do to the dev score?"*
