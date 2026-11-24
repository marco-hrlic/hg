# 1.3 Text to numbers (10 min)
**Goal:** every review becomes a vector of numbers, with no leakage.

## Why this matters
**TF-IDF**: one slot per vocabulary term, high for words frequent in *this* review but rare overall.
**Fit on train only**, then apply to dev and evaluation data; anything else leaks.

## Predict
With a 20,000-term vocabulary, how many slots are non-zero for a typical review?
**A** about 10  **B** about 100  **C** about 5,000

## Prompt
> Write `src/features.py` with a function that fits a TF-IDF vectorizer on train texts only and
> transforms train and dev. [DECISION: vocabulary] **A** 5,000 · **B** 20,000 (default) · **C** 50,000.
> [DECISION: terms] **A** single words · **B** words and two-word phrases (default).
> Print the matrix shapes and the average number of non-zero values per review.

## Verify
Ask Claude to show the one line where the vectorizer is fitted. `lab check 1.4` verifies it too.

## Takeaway
Fit on train, apply everywhere else.

## Go deeper
bag of words · n-grams · sparse matrix · word embeddings ·
*Ask Claude: "What goes wrong if the vectorizer is fitted on train + dev?"*
