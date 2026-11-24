# 2.2 Clean and split (15 min)
**Goal:** a clean, versioned train/dev split with no copies across it.

## Why this matters
Every cleaning rule is a **decision**: write it down. Split **stratified** (same label mix on both
sides), with **no copies across the split**, and never "the last 10% of the file".

## Predict
How many of the 20,700 rows survive?  **A** about 20,500  **B** about 19,900  **C** about 18,000

## Prompt
> Write `src/clean.py` with these rules, logging how many rows each removes:
> - placeholder/empty texts: drop · exact duplicates: keep one copy
> - [DECISION: invalid stars] **A** drop (default) · **B** repair the obvious ones ("5 stars" → 5)
> - [DECISION: HTML, `\n`, `\"`] **A** repair (default) · **B** drop the row
> - [DECISION: same text, conflicting labels] **A** drop all copies (default) · **B** keep the first
> - [DECISION: near-duplicates] **A** keep the longest copy (default) · **B** keep all
> - [DECISION: columns] **A** `review_id,text` + label (default) · **B** every column
>
> Map stars to labels (1–2 negative, 3 neutral, 4–5 positive). Stratified 90/10 split, fixed seed,
> into `outputs/train.csv` and `outputs/dev.csv` (`review_id,text,label`). Save `outputs/manifest.json`
> with row counts, label mix and each file's SHA-256.

## Verify
`lab check 2.2` runs a dozen checks: leakage, copies across the split, stratification.

## Takeaway
Every cleaning rule is a modelling decision. Write it down, version the result.

## Go deeper
stratified split · group and time splits · data versioning (DVC, Data Version Control) ·
*Ask Claude: "Why hash the output files?"* · *"If we had customer IDs, how should we split?"*
