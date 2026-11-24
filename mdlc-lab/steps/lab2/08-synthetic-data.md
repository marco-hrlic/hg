# 2.8 Synthetic data (5 min, instructor demo)
**Goal:** 300 LLM-written reviews for our weak slices, checked for contamination.

## Why this matters
**Synthetic data** can fill rare cases, and can quietly fool you. Rules: train only, never evaluate
on it; no near-copies of evaluation items; keep it a minority; keep it only if real data says it helps.

## Predict
Next to 100,000 real reviews, 300 synthetic ones will…  **A** clearly help  **B** make no measurable difference  **C** hurt

## Prompt
> [DECISION] **A** copy `data/lab2/synthetic_pregenerated.csv` to `outputs/synthetic.csv`, columns
> `text,label` (default) · **B** write 100 fresh reviews for sarcasm, mixed verdicts, negation, lukewarm and
> comparisons, labelled by `data/lab2/GUIDELINE.md`, to `outputs/synthetic.csv`, with the prompt in
> `outputs/synthetic_prompt.md`.
> Show three examples and the label mix.

## Verify
`lab check 2.8` checks contamination and the synthetic share. Would you believe a customer wrote them?
Step 2.9 tests whether they help.

## Takeaway
Synthetic data is a hypothesis, not an improvement. Test it on real data.

## Go deeper
data augmentation · contamination · distribution shift ·
*Ask Claude: "What is model collapse?"*
