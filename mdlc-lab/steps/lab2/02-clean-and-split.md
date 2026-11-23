# 2.2 Clean and split (15 min)

## Why this matters
Every cleaning action is a **decision** with consequences: drop or repair? keep one copy of a
duplicate, or none? Write the decisions down. They're part of the model.

Then split the data into **train** and **dev**:
- **stratified**: each label divided separately, so both splits have the same label mix;
- **no copies across the split**: a review (or a near-copy of it) in both train and dev makes the
  dev score measure memory;
- never "the last 10% of the file". Look at how the file is ordered.

Evaluation data is already separate: it comes from a different source and was deduplicated
against this pool. Keep it that way.

## Predict first (notes.md)
How many of the 20,700 rows will survive cleaning?

## Prompt
> Write `src/clean.py`. My cleaning decisions:
> - invalid star values: [DECISION: drop, or try to repair "5 stars" → 5?]
> - placeholder/empty texts: drop
> - HTML and literal "\n": [DECISION]
> - exact duplicates: keep one copy
> - duplicates with conflicting ratings: [DECISION]
> - near-duplicates: [DECISION]
> - columns: [DECISION: which columns to keep]
>
> Map stars to labels (1–2 negative, 3 neutral, 4–5 positive). Log how many rows each rule removes.
> Then do a stratified 90/10 split with a fixed seed into `outputs/train.csv` and `outputs/dev.csv`
> (`review_id,text,label`). Save a manifest with row counts, label mix and a hash of each file.

## Verify
`lab check 2.2` runs a dozen checks: leakage, duplicates across the split, stratification.
- Ask Claude: *"Why do we hash the output files?"*
- Ask: *"Our data has no customer or date columns. If it had, how should we split?"* (group split, time split)

## ML thinking
**Every cleaning rule is a modelling decision: write it down and version the result.**

## Go deeper
stratified split · group k-fold · time-based split · data versioning (DVC) · data lineage
