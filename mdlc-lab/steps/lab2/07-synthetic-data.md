# 2.7 Synthetic data (10 min)

## Why this matters
The model is weak on sarcasm, mixed verdicts and lukewarm reviews, and real examples are rare.
An LLM can write targeted examples. That's **synthetic data**. It can help, and it can quietly
fool you. The rules:
1. **Train only. Never evaluate on it.** A score on LLM-written text measures how well you imitate
   the LLM, not how you do on customers.
2. **Check for contamination**: no synthetic review may be a near-copy of an evaluation item.
3. **Measure on real evaluation data**, with and without. Keep it only if it helps.
4. **Keep it a minority** of the training data, and record the prompt and model used.

## Predict first (notes.md)
Will 100–300 synthetic reviews measurably improve the golden-set score?

## Prompt
> [DECISION: generate fresh or use the pre-generated set?]
> - *Fresh:* write 100 realistic reviews for our weak slices (sarcasm, mixed verdicts with a clear
>   lean, negation, lukewarm, comparisons), varied in business type, length and tone, labelled
>   according to `data/lab2/GUIDELINE.md`. Save to
>   `outputs/synthetic.csv` (`text,label`) and save your prompt to `outputs/synthetic_prompt.md`.
> - *Pre-generated:* copy `data/lab2/synthetic_pregenerated.csv` to `outputs/synthetic.csv` (columns `text,label`).
>
> Then show me five examples and the label mix.

## Verify
`lab check 2.7` checks contamination against the evaluation texts and the synthetic share.
- Read ten synthetic reviews. Would you believe a customer wrote them? Are the labels right?
- Ask Claude: *"What is model collapse, and why does it matter when training on generated data?"*

## ML thinking
**Synthetic data is a hypothesis, not an improvement. Test it on real data.**

## Go deeper
synthetic data generation · data augmentation · contamination · distribution shift · model collapse
