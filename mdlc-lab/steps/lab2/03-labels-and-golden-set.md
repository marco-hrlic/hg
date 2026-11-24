# 2.3 Labels and the golden set (15 min, no Claude for the first part)

## Why this matters
Every model learns from labels and is judged by labels. **How good are ours?**

A **golden set** is a small evaluation set whose labels were assigned carefully:
- a written **labelling guideline** with rules for hard cases (`data/lab2/GUIDELINE.md`);
- **at least two independent raters**; their agreement is measured with **Cohen's kappa**
  (agreement beyond chance: 1 = perfect, 0 = chance level);
- disagreements **adjudicated** by a person;
- **hard cases included on purpose**, tagged by **slice**;
- kept **independent** of training data, small, frozen and versioned.

Ours was drafted by two different LLMs labelling independently (κ = 0.95), with disagreements sent
to a human. That's a legitimate way to scale labelling, *if* you measure and check it.

## Part 1: label 10 reviews yourself (no Claude)
Read `data/lab2/label_these_10.csv`. For each, decide negative / neutral / positive using the
guideline. Write `outputs/my_labels.csv` with columns `item_id,label`. Don't look at stars.

## Predict first (notes.md)
Of the 3-star reviews in the golden set, what share read as *neutral* when you actually read them?

## Verify
`lab check 2.3`: your agreement with the golden labels, and the stars-vs-golden table.
- Look at the 3-star column. What does it say about the labels lab 1 trained on?
- The check also shows what happened when an LLM rater relabelled **all 2,316 three-star reviews**
  in the development data: only about a third read as neutral. That's ~1,450 wrong training labels.
  (In our tests, training on the corrected labels alone didn't improve the model: neutral became
  so rare that the model nearly stopped predicting it. Fixing labels changes the class balance too.)
- Ask Claude: *"What is weak supervision, and when is a proxy label like star ratings good enough?"*

## ML thinking
**Labels are data too, and usually the noisiest part.** Measure agreement before trusting them.

## Go deeper
inter-annotator agreement · Cohen's / Fleiss' kappa · annotation guidelines · LLM-as-annotator · weak supervision
