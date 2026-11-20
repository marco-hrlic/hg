# Lab 2 (draft): instructor notes for the data-engineering block

## Before the session
- Everyone runs `make data` beforehand. The download is the only step that can fail on the day.
  Compare `data/manifest.json` hashes if results differ between machines.
- Run `make check` on the projector machine.
- Create a shared sheet with columns `item_id, label, rater` for the 30-item exercise, and a second
  one from `golden/labelling_sheet.csv` split into blocks of ~40 items per team.
- Don't hand out `raw/answer_key.csv`. Participants score themselves with `data.score_findings`.

## Timing and what to expect

| Block | Time | Watch for |
|---|---|---|
| Framing | 15 min | Push teams to name the *expensive* mistake. It drives the loss design in session 2. |
| Labelling exercise | 25 min | Room kappa is usually 0.4–0.6 on these 30 (22 of them were picked to be debatable). Spend the time on the guideline rewrite, not the number. |
| Data handling | 35 min | Exact dedup after normalising catches 300/300 exact, 60/60 conflicting and only ~150/300 near-duplicates. MinHash at 0.7 gets 296/300. The ~20 "flagged but not planted" are placeholder texts and one natural duplicate. A good discussion point: the answer key isn't perfect either. |
| Golden set | 25 min | Teams label ~40 items each, two people per item. Collect into `golden/golden_labels.csv` (`item_id,label`); rows labelled `ambiguous` are dropped automatically. |
| Metrics and baselines | 20 min | Majority: ~62% val accuracy but ~0.25 macro-F1. Lexicon: ~0.67 / ~0.48. Negation handling helps the negation slice. |

## Reference numbers (star labels standing in for golden labels)

| model | val acc | val macro-F1 | golden acc | golden macro-F1 |
|---|---|---|---|---|
| majority | 0.621 | 0.255 | 0.425 | 0.199 |
| lexicon | 0.672 | 0.475 | 0.570 | 0.479 |
| lexicon + negation | 0.692 | 0.502 | 0.590 | 0.506 |

The golden set deliberately over-samples hard cases, so its scores are lower than validation.
That's expected and worth pointing out: the two sets answer different questions.

## Known gaps
- The ~600 flipped labels are not detectable in session 1. They are the hook for session 3.
- Slice tags are regex heuristics. Expect false positives ("no" in "no wait at all").
