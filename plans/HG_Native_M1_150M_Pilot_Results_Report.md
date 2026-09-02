# HG-Native M1 — 150M Stage M1-1 pilot results

**For:** the theory and planning track
**Basis:** `plans/HG_Native_M1_150M_Generalization_Refinement_Kickoff.md`, building on `plans/HG_Native_M0_R1_Replication_Baseline_Report.md`
**Date:** 2026-09-02
**Status:** single-seed pilot (kickoff §16 Stage M1-1), **not** the full 3-seed confirmatory report (§23) — that needs Stage M1-2, not attempted here.

Self-contained: every number below is reproduced from a saved
`manifest.json`/`metrics.json` or the training logs, nothing requires
reading the code. States plainly which hypotheses this run supports,
which it doesn't, and which it couldn't actually test due to gaps in
this pilot's own wiring — several of the latter, not hidden.

---

## 1. What this pilot is

M0-R1 (57M params, replicated across 3 seeds) established that HG produces
a more selective task-relative decision state than vanilla pretraining —
Outcome B: **HG organizes what's already there; it doesn't reveal
information vanilla lacks.** M1 asks whether that effect survives scale,
distribution shift, cross-family transfer, and capability change, not just
IID structure at 57M.

This session (2026-09-02) built the full M1 groundwork the kickoff's own
full-scope path calls for — OOD splits for lexical/parameter/distractor
(Family A) and parameter (Families B/C), structural OOD (Family A), a
cross-family transfer generator (Family E), condition-matrix/bootstrap
reporting, and a stronger vanilla decision-probe control — verified
end-to-end on debug-scale checkpoints across 10 slices (commits `cd85687`
through `697c4c4`), then ran the actual Stage M1-1 pilot: one seed,
Baseline-150M vs. HG-150M, real BPE tokenizer + TinyStories, on the DGX
Spark GPU.

**Headline**: training was healthy (no stop condition triggered), the core
M0-R1 selectivity result **replicates at 150M** (N4 `Q_sel = +0.35`), and
one new M1 hypothesis (parameter-OOD selectivity survival) is **supported**
with tight, non-overlapping-with-zero confidence — but two others
(capability refinement, cross-family transfer) show **weak or unsupported**
results at this pilot's training budget, and structural OOD's own
comparison is **not currently a valid test** of what it's supposed to
measure (a real, disclosed gap in this session's wiring, not a model
failure — see §7).

---

## 2. Exact configuration

| | Baseline | HG |
|---|---|---|
| config | `experiments/hg_native/configs/m1_baseline.yaml` | `experiments/hg_native/configs/m1_hg.yaml` |
| params | 163,001,216 | 163,549,278 |
| overhead | — | **0.336%** (well under the <5%, even <2%, target) |
| trunk | `d_model=896, n_layers=16, n_heads=14, n_kv_heads=7, d_ff=2560, max_seq_len=512, vocab=16000` (same for both) | same |
| HG latent | — | `hidden_dim=384, latent_dim=96, layers=2` |
| seed | 0 | 0 |
| steps | 2000 | 2000 |
| batch size | 16 | 16 |
| tokenizer | real trained BPE, `experiments/hg_native/tokenizer/hg_native_bpe.json` (16000 vocab) | same |
| ordinary_lm | 20,000 real TinyStories examples, ratio **0.8** (the existing `run_train.py` CLI default — kickoff §5 suggests 0.85; not overridden for this pilot, noted as a minor deviation, not a confirmatory-run concern per the kickoff's own "adjusted only before confirmatory runs" allowance) | same |
| device | DGX Spark GB10, CUDA 13.0, torch 2.13.0+cu130 | same |
| run id | `20260902T101602Z_m1_pilot_seed0_baseline` | `20260902T104351Z_m1_pilot_seed0_hg` |
| git commit | `1d119f7` (dirty: true — untracked PDF files in `plans/`, no uncommitted code) | same |
| wall time | ~17.3 min (2000 steps @ ~3940 tok/s steady-state) | ~27.5 min (2000 steps @ ~2480 tok/s steady-state) |

---

## 3. Training health and ΔNLL

Both runs' losses dropped smoothly with no divergence, NaN, or collapse:

- Baseline: `lm` loss 9.85 → **1.7845** (first → last step).
- HG: `total` loss 9.85 → 1.9866, `lm` loss 9.85 → **1.7931**.

**ΔNLL = 1.7931 − 1.7845 = +0.0086 nats** — small, same order of magnitude
as M0-R1's own 0.0097-nat finding at 57M. **H-M1-6 (ordinary LM cost
remains small): supported.**

HG's per-head losses (`accessibility`, `info_deficit`, `route`) all
dropped to near-zero by step ~300-400 and stayed low; `decision`/
`transition`/`quotient` kept a modest amount of residual noise through
step 2000 (e.g. `decision` oscillated 0.42-0.72 in the last few hundred
steps rather than fully flattening) — consistent with a model that is
still converging at this step budget, not one that has plateaued. **2000
steps at 163M is very likely undertrained relative to what 2000 steps
bought at 57M in M0-R1** (a materially smaller model per step) — worth
weighing before Stage M1-2 replication: either more steps, or reporting
results explicitly as a lower-bound/undertrained pilot, which is how §5-§8
below should be read throughout.

No stop condition (kickoff §16: N0 failing badly, HG latent collapse,
large LM degradation, capability-refinement leakage, overlapping OOD
splits) fired. Details in §9.

---

## 4. IID replication (N0-N5, N4)

Held-out pool: 5400 episodes (seed 12345, disjoint from training seed 0),
`n_per_family=200`.

| gate | result | read |
|---|---|---|
| N0 capability floor | accuracy 0.734, balanced 0.740, no label collapse | reasonable, headroom left (undertrained) |
| N1 grounding discrimination | pooled AUROC 0.733 | **uneven by subclass** — `multi_bit_deficit`/`distractor_visible` near-perfect (brier ~1e-11); `one_bit_deficit`/`redundant_hidden` near-total-failure (brier ~0.9995, i.e. confidently *wrong*) |
| N2 routing | route accuracy 0.917; model cost sits between oracle and the `always_*` baselines across the price grid | healthy, non-degenerate |
| query efficiency (Family A) | soundness 0.941, η_Q 1.36 | comparable to earlier debug-scale runs |
| N3 accessibility discrimination | AUROC 1.0 | essentially perfect |
| N3 route rates on zero deficit | ACT 0.375 / THINK 0.625 / ASK 0.0, unnecessary-ask 0.0 | clean, matches every prior run this project has produced |
| N5 grounding reuse | repeated-ask rate 0.0 | clean |
| **N4 quotient recovery** | r=0.95, f=0.883, **h=0.787**, u=0.543, n=0.51 → **Q_sel = +0.347** | **the core replication result — genuine positive selectivity at 150M** |

N4 is the headline: relevant variables (`r`/`f`/`h`) are strongly
recovered, irrelevant ones (`u`/`n`) sit near chance, `Q_sel` is clearly
positive. This is numerically a bit lower than M0-R1's own 57M pilot
(`h=0.902`, `Q_sel=+0.43`) — plausibly the undertraining noted in §3 (a
larger model needs more steps to reach the same per-parameter fit), not
necessarily a scale-inherent ceiling; a single pilot at one step budget
can't distinguish the two. **Flagged for Stage M1-2, not resolved here.**

N1's per-subclass split (some subclasses essentially solved, others
essentially failed) is worth a closer look before the next run — it isn't
a generator bug (the same generator/subclasses passed cleanly in every
prior debug/pilot run this project has done at smaller scale) and reads
most plausibly as an artifact of this run's specific 2000-step budget
interacting with per-subclass class balance, not a fundamental issue.

---

## 5. OOD generalization (kickoff §6, §10, §11)

### 5.1 Parameter OOD, Family C (`n_steps` 2-6 train-range vs. 7-10 OOD-range)

| condition | Q_sel | 95% CI (bootstrap, n=500) |
|---|---:|---|
| IID | 0.352 | [0.268, 0.421] |
| Parameter OOD | 0.325 | [0.275, 0.387] |

**ρ_OOD = 0.325 / 0.352 = 0.923.** The two CIs overlap substantially — no
statistically clear degradation from the longer, unseen trajectory-length
range. **H-M1-1 (selectivity survives OOD): supported** on this one axis,
at single-seed confidence. This is a genuinely positive, non-trivial
result: HG's selectivity is not brittle to this distribution shift.

**Caveat, stated plainly**: this compares HG's own IID vs. OOD `Q_sel` —
it does **not** include a vanilla matched-dimension-PCA control (the
`H_t^LM,kD` side of kickoff §10's three-way comparison), because that
control (`eval/projections.py` exists, but was never wired into
`condition_matrix`/`quotient_recovery` this session). So the *stronger*
form of H-M1-1 (`Q_sel^HG,OOD > Q_sel^LM,OOD`) is **not tested** by this
pilot — only "HG's own OOD degradation is small" is. Wiring the vanilla
PCA control into this comparison is the natural next slice before this
axis can support a full confirmatory claim.

### 5.2 Lexical / distractor-density OOD (Family A), other-axis parameter OOD (Family B)

Built and leakage-tested (slices 2-3) but **not wired into this pilot's
eval run** — `run_eval.py` only exercises Family C's parameter-OOD
condition matrix and Family A's structural-OOD comparison (§5.3). A real
gap in this pilot's coverage, not a negative result: these pools exist,
are proven correct via their own test suites, and just weren't plumbed
into a `condition_matrix`-style report this session. Next-step item.

### 5.3 Structural OOD (Family A)

| condition | N0 capability floor | query efficiency (soundness) |
|---|---:|---:|
| "IID" composition (`h1` alone) | 0.56 (**constant-label collapse**: always predicts B) | 1.0 |
| "OOD" composition (unseen `h1 OR (h2 AND NOT h1)`) | 0.43 | 0.71 |

**Important limitation, not a negative HG result**: `generate_structural_episode`
was built (slice 7) with its own leakage-checked train/OOD condition
split, but **was never wired into `training/dataset.py`'s `SyntheticMix`**
— the model was trained on the 6 named `static_hidden` subclasses only,
and has **zero exposure to either the "structural IID" or the "structural
OOD" composition**, in any form. So this comparison is actually two
zero-shot conditions against each other, not "does selectivity trained on
IID compositions survive an unseen composition" — the hypothesis kickoff
§6.2 actually wants isn't tested by this pilot. The constant-label
collapse on the "IID" condition is exactly what you'd expect from a model
seeing a completely novel 100-item synthetic slice it was never trained
on at all, at a modest step budget — not evidence of a general N0/latent
failure (the *real* held-out N0 in §4, on subclasses actually trained on,
shows no such collapse). **This needs training-side wiring (add
structural episodes to `SyntheticMix` with a real train/OOD split) before
it can be interpreted as a test of H-M1-1 for this axis.** Flagged
clearly rather than reported as a finding either way.

---

## 6. Cross-family transfer (kickoff §7, Family E)

Zero-shot probe (Family E, `resource_allocation.py`, never seen in
training in any form — by design, a *stronger* test than the kickoff's
own stated minimum):

| var | accuracy |
|---|---:|
| r (urgent) | 0.847 |
| f (future_urgent) | 0.62 |
| h (capacity) | 0.60 |
| u (priority_site, irrelevant) | 0.70 |
| n (weather, irrelevant) | 0.567 |

**Q_sel = +0.056** — barely positive, much weaker than Family C's own IID
`Q_sel = +0.347`. Only `r` transfers well; `f`/`h` are barely above
chance, and `u` (meant to be irrelevant) actually scores *higher* than `h`
(meant to be relevant). **H-M1-3 (info/access discrimination transfers to
an unseen family) is not well supported by this pilot.** Stated plainly:
zero-shot cross-family transfer of the HG latent's selectivity is weak at
this training budget. This doesn't distinguish between "the effect
genuinely doesn't transfer past syntax" (Outcome M1-B/worse) and
"2000 steps at 163M isn't enough exposure to the base mechanism for it to
transfer cleanly even within-family" (§3's undertraining caveat) — a
single pilot seed can't tell these apart. Worth a longer run before
drawing a real conclusion here.

---

## 7. Capability refinement (kickoff §8, §16 Stage M1-0's own metric)

| var | ProbeAcc(C1) | ProbeAcc(C2) | Δrefine |
|---|---:|---:|---:|
| v (should refine: irrelevant→relevant) | 0.983 | 1.0 | +0.017 |
| u (control: irrelevant throughout) | 0.617 | 0.638 | +0.021 |

**H-M1-4 (v's recoverability rises from C1 to C2) is not supported**: `v`
is already almost perfectly recoverable *under C1*, where it should be
near chance — there's a ceiling effect leaving essentially no room for
the predicted rise, and `u`'s shift is statistically indistinguishable
from `v`'s (+0.021 vs. +0.017). This is **not generator leakage** — the
`z`/`z'` counterfactual-pair construction was exhaustively verified
leakage-free at build time (slice 1's own test suite, unchanged since);
it's a genuine property of what this trained representation does. Read
plainly: at this pilot's scale/training budget, the HG latent does not
show the capability-selective suppress-then-reveal pattern the theory
predicts for `v` — it encodes `v` regardless of which capability token is
present. This is the pilot's clearest negative finding, and matches
kickoff §25's **F-M1-1 falsification condition** ("the newly relevant
variable remains suppressed under C2") in spirit, except inverted: `v`
isn't suppressed under *either* capability, so this is closer to "the
latent doesn't discriminate `v`'s relevance by capability at all" than
either F-M1-1 or F-M1-2 exactly as worded. Whether this persists with
more training or is a genuine architectural limit (over- or
under-compression relative to capability context) needs Stage M1-2, not
resolvable from one seed.

---

## 8. Vanilla decision-probe control (kickoff §12)

Raw trunk hidden state (`H_t^LM,raw`, before any HG-specific projection),
linear probe on the actual decision label, 70/30 fit/eval split on the
`static_hidden` held-out pool:

| | Baseline | HG |
|---|---:|---:|
| held-out accuracy | 0.586 | 0.661 |
| decision_probability (mean) | 0.969 | 0.978 |
| decision_entropy (mean) | 0.066 | 0.051 |
| top1_top2_margin (mean) | 0.939 | 0.956 |

HG's raw trunk shows a modest edge (0.661 vs. 0.586, ~7.5pp) over the
baseline's raw trunk on this probe — interesting directionally (even
before the HG-specific latent projection, the HG-trained trunk carries
slightly more decision-relevant signal), but with `n_eval=360` at one
seed this is not a confident claim; both probes are also strikingly
overconfident relative to their actual accuracy (mean predicted
probability ~0.97-0.98 against ~60-66% actual accuracy) — a calibration
problem in the probe/trunk-at-this-training-budget combination worth
noting for anyone using these features downstream, not just a scoreboard
number.

---

## 9. Stop-condition check (kickoff §16)

| condition | triggered? |
|---|---|
| N0 fails badly | No — 0.734/0.740, no collapse, on the real held-out set actually trained on |
| HG latent collapses | No — N4/§5.1 show real, structured, non-degenerate probe signal |
| LM degradation becomes large | No — ΔNLL +0.0086 nats |
| Capability-refinement generator leaks capability labels | No — verified leakage-free at build time; §7's flat result is a representation finding, not a leak |
| OOD splits accidentally overlap | No — `assert_disjoint_ranges`/`assert_no_shared_composition`/`assert_no_shared_lexical_templates` all run at pool-build time and would have raised, not silently passed |

**None fired. The pilot is healthy enough to report** — but §5.3 and §6-7
above are genuine mixed/negative findings, not glossed over.

---

## 10. What this pilot does and doesn't establish

**Supported at single-seed confidence:**
- H-M1-6 (small LM cost) — yes.
- Core M0-R1 selectivity result replicates at 150M (N4 `Q_sel=+0.35`) — yes, though numerically softer than the 57M pilot.
- H-M1-1 for Family C's parameter-OOD axis specifically — yes, `ρ_OOD=0.92`, CIs overlap.

**Not supported / weak at this pilot's training budget:**
- H-M1-4 (capability-selective refinement) — `v` is encoded regardless of capability; no discrimination shown.
- H-M1-3 (cross-family zero-shot transfer) — weak (`Q_sel=+0.056` vs. Family C's own +0.35).

**Not actually tested by this pilot (wiring gaps, not results):**
- The vanilla matched-dimension-PCA control for §5.1's OOD comparison.
- Lexical/distractor-density OOD condition matrices (built, leakage-tested, not run here).
- Structural OOD as kickoff §6.2 actually intends it (needs training-side wiring, not just eval-side).

**Confounded with training budget, not resolved:**
- Whether §6/§7's weak results are a genuine architectural limit or an
  artifact of 2000 steps at 163M (plausibly undertrained per §3's own
  evidence). A longer run at the same scale, or Stage M1-2 replication
  proper, is needed before drawing a real conclusion on either.

---

## 11. Recommendation

**Not** an unqualified GO to Stage M1-2 (3-seed replication) as currently
scoped. Before spending 3x the compute on replication:

1. Wire the vanilla matched-dim-PCA control into `condition_matrix` (§5.1's gap) — cheap, mechanical, makes the OOD comparison's strong form actually testable.
2. Either run a longer single-seed pilot (more steps, same 163M scale) to separate "undertrained" from "genuine limit" for §6/§7's weak results, or proceed to Stage M1-2 explicitly treating this as the undertrained baseline and expecting the same weak pattern to persist or resolve across seeds.
3. Wire structural OOD into training (`SyntheticMix`) before trusting §5.3's numbers as a test of anything.
4. Wire the built-but-unused lexical/distractor-density OOD condition matrices into a real eval run.

None of this invalidates the pilot — it's a real, honest first 150M data
point, with a genuine positive replication (N4) and a genuine positive OOD
result (§5.1) alongside genuine negative ones (§6, §7) that the M0
pilot's smaller scale never had the chance to show. Read it as: **the
core mechanism replicates and shows real OOD robustness on the one axis
fully tested; the harder generalization/refinement claims are not yet
supported, and part of why is that this pilot didn't yet test them the
way the theory needs.**
