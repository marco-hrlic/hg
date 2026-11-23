# 1.1 Meet the data (15 min)

## Why this matters
We want a **classifier**: a function `review text → negative | neutral | positive`. Instead of
writing rules, we show a program thousands of reviews **with the right answer** (the **label**)
and let it learn the function. That's **supervised learning**. Our labels come from star ratings:
1–2 ★ negative, 3 ★ neutral, 4–5 ★ positive.

The data is already split:
| | Set | Used for |
|---|---|---|
| **Development** | `train` | the model learns from it |
| | `dev` | you compare models and settings on it, as often as you like |
| **Evaluation** | `golden`, `held-out` | the final verdict. Texts only; the harness keeps the labels |

Development and evaluation data come from different source files, so no review is in both.

## Predict first (notes.md)
If a "model" always answered **positive**, what accuracy would it get on dev?

## Prompt
> Load `data/lab1/train.csv` and `data/lab1/dev.csv`. Write `src/explore.py` that prints: the number
> of reviews in each, the label distribution (counts and percentages) in each, review length
> statistics, and three random example reviews per label. Explain what the label distribution means
> for how we should measure a model.

## Verify
- Is the label mix the same in train and dev? Ask Claude: *"Why would we want that, and what is a
  stratified split?"*
- Check your prediction: what fraction of dev is positive?
- Ask: *"What is class imbalance and why does it make accuracy misleading?"*

## ML thinking
**Look at the data before modelling.** Read the examples Claude printed. Would *you* agree with
the labels?

## Go deeper
class imbalance · stratified sampling · train/dev/test split · supervised learning
