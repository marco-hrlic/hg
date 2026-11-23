# Instructor guide

## What the labs are for
Participants won't be able to build this project alone afterwards, and that's not the goal. The
goal is that they **see every major component**, **learn the vocabulary**, and pick up **the ML way
of working**: evaluation first, baselines, predict-then-run, one change at a time, suspicion of good
results, looking at data and errors. That's enough to direct agents and research the rest.

Claude Code writes the code. Participants make the decisions, predict outcomes, and verify.

## Format of every step (5–25 min)
1. **Instructor, 2–4 min:** introduce the concept from the card's "Why this matters". Draw it.
2. **Participants:** write the prediction in `notes.md`, fill in the `[DECISION]`s, paste the
   prompt into Claude Code, then run `lab check`.
3. **Instructor, 2–3 min:** debrief. Ask two or three people what they predicted vs got; read one
   surprising result aloud. This is where the learning happens. Don't skip it to save time.

If someone falls behind: `lab catchup <step>` writes reference outputs so they can continue
with the next step.

## Before the session
- Every participant has Claude Code working, Python 3.10+, and has run, from the repo:
  `cd mdlc-lab && pip install -e . && cd workspace && lab setup`. (Downloads ~200 MB. PyTorch CPU build:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`.)
- They open Claude Code **inside `workspace/`**, so its `CLAUDE.md` and `.claude/settings.json` apply.
- For lab 2: finish the human review of the golden labels (`golden_labels/README.md`).

## Guardrails, and how strong they are
- Evaluation labels and the answer key never enter `workspace/`; the harness reads them.
- `workspace/.claude/settings.json` sets `blockReadsOutsideWorkingDirectories`, so Claude Code's
  file tools can't read them. Shell commands aren't sandboxed (the agent has to run `lab`), so a
  determined agent or person could still read them. `CLAUDE.md` tells the agent not to. Treat it as
  an honour system with a fence, and say so: *in real projects, evaluation discipline is mostly
  culture too.*
- Held-out submissions are rationed (two in total, one per lab). Every submission is logged before
  its score is shown. `lab log` shows the history, including forced extra looks.

## What the checks catch (and what agents tend to get wrong)
| Mistake | Caught by |
|---|---|
| Vectorizer fitted on train + dev | a "suspiciously good" warning in 1.4 if dev accuracy > 0.92; mainly the card's verify questions |
| Using the `stars_display` column | 2.2 (leaky columns) |
| Splitting by file position on a sorted file | 2.2 (stratification) |
| Duplicates across train/dev, conflicting copies kept | 2.2 |
| Unstable softmax, wrong gradient | 2.4 (tests against PyTorch autograd) |
| Synthetic data copying evaluation items | 2.7 (near-duplicate check) |
| Peeking at held-out repeatedly | `lab submit` refuses after the allowance |

## Self-study material
`notebooks/` has the same content as two worked notebooks (solutions in `solutions/`). Good
for participants who want to read the full code afterwards, or for preparing the session.
