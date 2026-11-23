# How we work (10 min)

**Goal of these two labs:** see every major part of building an ML model, learn the vocabulary,
and learn the *ML way of working*, so that afterwards you can direct agents like Claude Code and
research any of these topics yourself. You are not expected to write this code by hand. Claude
writes it; **you make the decisions, predict the results, and check the work.**

## The ML way of working: seven habits
1. **Decide how you'll measure success before building anything.** The evaluation comes first.
2. **Start with the dumbest baseline.** A model is only as good as the margin by which it beats it.
3. **Predict, then run.** Write down what you expect. The surprises are where you learn.
4. **Change one thing at a time** and record every experiment.
5. **Small differences are often noise.** Re-run with another random seed before believing a 1-point gain.
6. **Suspect results that look too good.** It's usually leakage: the answer slipped into the input.
7. **Look at the data and at the mistakes**, not just the averages. Data quality beats model cleverness.

## Your tools
| Tool | What it's for |
|---|---|
| **Claude Code** | writes and runs the code; explains it when you ask |
| **`lab`** | the harness: step cards, checks, scores. It holds the evaluation labels; nobody else sees them |
| **`notes.md`** | your lab journal: predictions, results, decisions |

## Each step card has
- **Why this matters**: the concept and the words for it.
- **Predict first**: write it in `notes.md` *before* running anything.
- **Prompt**: paste it into Claude Code. Fill every `[DECISION: ...]` yourself. That's the part
  that's yours.
- **Verify**: `lab check`, plus questions to ask Claude about what it built. Agents make
  plausible-looking mistakes; checking is part of the job.
- **Go deeper**: what to search or ask later.

## Do now
```
pip install -e ..        # from workspace/, once
lab setup                # exports the data
claude                   # start Claude Code in this folder
```
Then `lab step 1.1`.
