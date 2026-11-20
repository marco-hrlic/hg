# Golden labels

Labels for the 400 golden candidates (`data/golden/candidates.csv`, built by `make data`).
Only item ids and labels are committed here, never review text.

## How the draft was made
1. A written guideline: [GUIDELINE.md](GUIDELINE.md).
2. Two independent LLM raters labelled all 400 items without seeing the star ratings.
   Rater A and rater B were different models, and B read most blocks in reverse order.
3. `merge_drafts.py` combined them into `draft_labels.csv`:

| status | items | meaning | in the draft golden set? |
|---|---|---|---|
| `agreed` | 280 | same label, both confident | yes |
| `agreed_low_conf` | 101 | same label, at least one rater unsure | yes; spot-check |
| `needs_review` | 13 | the raters disagree | no, until a person decides |
| `ambiguous` | 6 | both raters said ambiguous | no |

Agreement between the raters: Cohen's kappa 0.95 on the 392 items both gave a sentiment label.

Agreed labels vs the star-derived labels: 85% match. Of the 59 three-star reviews in the agreed set,
only 17 read as neutral (29 positive, 13 negative): the "3 stars = neutral" mapping is the largest
source of label noise in the dataset.

## Human review (to do)
1. Open `data/golden/review_sheet.csv` (120 rows, sorted: disagreements first, then ambiguous,
   then low-confidence agreements).
2. Fill `final_label` (negative / neutral / positive / ambiguous), `reviewer` and `reviewer_note`.
   The 19 disputed items matter most. For the 101 spot-checks, change only the ones you disagree with.
3. Run `python golden_labels/finalize.py` and commit `golden_labels.csv`.

Until `golden_labels.csv` exists, `mdlc.data.load_golden()` uses the agreed drafts and warns that
they are unreviewed.
