# 1.3 Text to numbers (15 min)

## Why this matters
Models do arithmetic, so each review must become a list of numbers: a **vector** of **features**.
We'll use **TF-IDF**: build a vocabulary (e.g. the 20,000 most useful words and two-word phrases);
each review becomes a vector with one slot per vocabulary entry. The value is high for words that
are frequent in *this* review but rare overall. "the" ≈ 0; "disgusting" is high.

**The first leakage rule:** anything that *learns from data* (here, the vocabulary and the word
rarities) is **fitted on train only**, then *applied* to dev and evaluation data. Fitting it on
everything lets information about the evaluation data leak into the model's inputs.

## Predict first (notes.md)
How many of the 20,000 slots will be non-zero for a typical review?

## Prompt
> Write `src/features.py`: a function that fits a TF-IDF vectorizer on the train texts
> [DECISION: vocabulary size, and single words only or also two-word phrases?] and transforms
> train and dev. Print the matrix shapes, the average number of non-zero values per review, and,
> for one example review, its 10 highest-weighted terms. Explain what you fitted on what, and why.

## Verify
- Ask Claude: *"Show me the line where the vectorizer is fitted. What would go wrong if it were
  fitted on train + dev?"*
- Ask: *"Why is the matrix stored as 'sparse'?"*

## ML thinking
**Fit on train, apply everywhere else.** This one rule prevents a whole family of bugs.

## Go deeper
bag of words · TF-IDF · n-grams · sparse matrix · data leakage in preprocessing · word embeddings (the modern alternative)
