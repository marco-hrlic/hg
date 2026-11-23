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
| Vectorizer fitted on train + dev | 1.4: vocabulary and IDF weights recomputed from train must match the saved vectorizer |
| Dev data leaking into training some other way | 1.4: "suspiciously good" warning above 0.92 dev accuracy |
| Using the `stars_display` column | 2.2 (leaky columns) |
| Splitting by file position on a sorted file | 2.2 (stratification) |
| Duplicates across train/dev, conflicting copies kept | 2.2 |
| Unstable softmax, wrong gradient | 2.4 (tests against PyTorch autograd) |
| Synthetic data copying evaluation items | 2.7 (near-duplicate check) |
| Peeking at held-out repeatedly | `lab submit` refuses after the allowance |

## Agent dry run (lab 1)
An agent playing a participant (and its Claude Code) completed steps 1.1–1.7 from the cards
alone: every check passed on the first attempt, at 3–6 tool calls per step. Its scores (dev
macro-F1 0.685, golden 0.720, held-out 0.701) match the reference solution. Fixes made from its
report: explicit file names for the saved model and vectorizer, id-column conventions, experiments
writing to their own folder and staying under ~2 minutes each, a hint for finding a network's
strongest words, and a real fit-on-train-only check.

## Agent dry run (lab 2)
An agent completed steps 2.1–2.8 from the cards alone; every check passed. What it taught us:
- **Compare like with like.** Its own lab 2 candidates didn't beat the lab 1 *reference* model on
  golden (0.704–0.714 vs 0.719), while held-out improved (0.695 → 0.717). Implementations differ by a
  few points, which is as large as the effects we're measuring. Step 2.8 now retrains lab 1's recipe
  as **candidate 0 in the same script with the same seeds**, and `lab check 2.8` reports each
  candidate's difference from it against the seed noise. This is a strong teaching moment: if
  someone's results are "within noise", that *is* the finding.
- The source also escapes quotes (`\"`), not just newlines. Cleaning now decodes both, and check 2.2
  fails if escapes remain.
- Lab 2 no longer depends on lab 1's code: step 2.5 has the agent write one reusable `train()`
  function (with a "should take under a minute" hint) that 2.6 and 2.8 reuse.
- Agent-written training can be slow (3.5 min per run in the dry run vs ~20 s in the reference). The
  cards now give the exact settings and the sparse-to-dense hint. Steps 2.5, 2.6 and 2.8 should run
  in the background while you introduce the next concept.

## Self-study material
`notebooks/` has the same content as two worked notebooks (solutions in `solutions/`). Good
for participants who want to read the full code afterwards, or for preparing the session.
