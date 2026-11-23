# 1.7 Break it (10 min)

## Why this matters
Averages hide failures. Knowing *how* a model fails tells you what to fix, and whether it's safe
to use. This is **error analysis**.

## Predict first
Write down three reviews you're sure will fool it, and what it will say.

## Prompt
> Write `src/classify.py` that takes review texts and prints the predicted label with the three
> probabilities. Run it on these: [DECISION: your tricky reviews: sarcasm, negation ("not bad at
> all"), mixed verdicts, comparisons, slang, emoji]. Save `outputs/break_it.csv` with columns
> `text,expected,predicted`. Then show me the words the model associates most strongly with each
> label (a network has no single weight per word, so probe it: feed each vocabulary term on its own
> and rank by predicted probability), and explain one of its failures.

## Verify
`lab check 1.7`
- Ask Claude: *"Why can't a bag-of-words model understand sarcasm? What kind of model could?"*

## The reveal
The "prepared" data you trained on came from an export with **seven kinds of problems**, and
**~600 of its labels are flipped** (5 ★ reviews stored as 1 ★). Your model learned from them.
**Lab 2:** start from the raw export, fix the data, learn the maths, retrain.

## ML thinking
**Find the failure modes before your users do.**

## Go deeper
error analysis · behavioural testing (CheckList) · bag-of-words limitations · transformers and contextual embeddings
