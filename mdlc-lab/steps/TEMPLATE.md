# Step card template

Cards are read during a live, timed session. Keep each card to about one screen.
`tests/test_lab_harness.py` requires the three headings marked *(required)*.

```markdown
# X.Y Title (N min)
**Goal:** one line: what participants will have at the end.

## Why this matters            (required)
Two or three lines. Name one or two concepts in bold. The instructor says the rest out loud.

## Predict
One multiple-choice question with three options: **A** … **B** … **C** …
Pick the question whose answer surprises people. Never an open number.

## Prompt
> What Claude should build. Short, with exact file names.
> [DECISION: what to choose] **A** option (default) · **B** option · **C** option

## Verify                      (required)
`lab check X.Y`, plus at most two quick things to look at.

## Takeaway
One line. The habit or idea to remember.

## Go deeper                   (required)
keyword · keyword · *Ask Claude: "…?"* (optional follow-up questions live here, not in Verify)
```

Rules of thumb:
- Every `[DECISION]` has lettered options and a marked default, so "default" is a valid answer.
- One prediction per step at most. Skip it when there's no real surprise.
- Each prediction has a matching line in `workspace/notes.md`.
- No more than one table per card.
- Spell out each acronym on the card where it first appears in the course: "TF-IDF (term frequency ×
  inverse document frequency)".
- Ask for a picture where it replaces reading (a diagram, heatmap or plot), but not on every step:
  every PNG is a file to open during the session.
