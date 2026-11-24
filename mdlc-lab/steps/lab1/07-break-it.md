# 1.7 Break it (10 min)
**Goal:** find how the model fails, not just how often.

## Why this matters
Averages hide failures. **Error analysis** shows what to fix and whether it's safe to use.

## Predict
How many of the eight starter reviews below will fool it?  **A** 0–2  **B** 3–5  **C** 6–8

## Prompt
> Write `src/classify.py` that prints the predicted label and the three probabilities for any texts.
> [DECISION: test reviews] **A** the starter set below (default) · **B** your own.
> Save `outputs/break_it.csv` (`text,expected,predicted`). Then list the 10 terms the model links
> most strongly to each label (probe: feed each vocabulary term alone, rank by probability).

Starter set (expected label in brackets):
"Not bad at all." (positive) · "Oh great, another hour of waiting." (negative) ·
"Amazing food, but so rude we're never coming back." (negative) · "This place slaps." (positive) ·
"Better than next door, which isn't saying much." (neutral) · "Mid." (neutral) ·
"It was fine. Nothing special." (neutral) · "I wouldn't say it was good." (negative)

## Verify
`lab check 1.7` · Look at the top neutral terms. Do they describe a verdict, or hedging words?

## Takeaway
Find the failure modes before your users do.

**The reveal:** the data you trained on had seven kinds of problems and ~600 flipped labels.
Lab 2 starts from the raw export.

## Go deeper
error analysis · behavioural testing (CheckList) · transformers ·
*Ask Claude: "Why can't a bag-of-words model understand sarcasm?"*
