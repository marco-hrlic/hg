# HG-Native M0-R1 — Replication + Matched Vanilla Baseline: results

**For:** the theory and planning track
**Basis:** `plans/HG_Native_M0_R1_Experiment_Kickoff.md`, `plans/HG_Native_M0_Technical_Spec_v0_1.md`
**Date:** 2026-09-02

Self-contained: every number below is reproduced from a saved run directory,
`comparison.json`, or `hg_eval_metrics.json` under `results/m0_r1/` (mirrored
into this checkout; checkpoints themselves live on the DGX Spark and are
gitignored, per the project's existing convention). Nothing here requires
reading the code.

---

## 1. The question this run answers

The M0 pilot (`plans/HG_Native_M0_Pilot_Results_Report.md`) showed HG-native
training produces strong-looking metrics at one seed, with an explicit caveat:
no replication, and no matched vanilla control existed for any HG head. This
run closes both gaps. Primary question, unchanged from the kickoff doc:

> **Does HG learn a decision-relevant representation and routing policy that
> matched vanilla pretraining does not?**

**Bottom line: yes, replicated across 3 seeds, on every measured axis.** The
one asterisk is *why* — see §4's per-variable breakdown, which points to
Outcome B (a cleaner decision state, not more information overall) rather
than Outcome A (raw information advantage).

---

## 2. Provenance and exact configs

| | |
|---|---|
| branch | `claude/hg-v2-e1-e2` |
| source commit for the new seed-1/seed-2 runs | `0454a58aad3b5bc852bb106aa583bc6b2dc0d6cb` |
| tree state at deploy | **dirty** (this session's own uncommitted changes were on disk when the runs launched — see §8) |
| frozen config | `experiments/hg_native/configs/pilot_baseline.yaml` / `pilot_hg.yaml`, unchanged from the pilot |
| trunk | vocab 16000, d_model 640, 10 layers, 10 heads / 5 KV heads, d_ff 1792, max_seq_len 256 |
| params | baseline 56,947,840 / HG 57,204,862 (0.45% overhead, unchanged) |
| tokenizer | the same trained byte-level BPE from the pilot (`experiments/hg_native/tokenizer/hg_native_bpe.json`) |
| steps / batch | 2000 / 16, `ordinary_lm` ratio 0.8, 20,000 TinyStories examples — identical to the pilot |
| seeds | 0 (reused pilot checkpoints), 1, 2 (new) |
| device | DGX Spark GPU (GB10) |

Every new run's `manifest.json` now carries a real `git_commit`/`git_dirty`
(the §17 provenance fix — see §8), unlike the pilot's two runs, which are
still `null` and stay that way (not rerun solely for this, per the kickoff
doc's own instruction).

Nothing about the frozen config, task families, or architecture was changed
for this run. No hyperparameters were tuned based on seed 0.

---

## 3. Training health, capability floor, ΔNLL

### 3.1 ΔNLL (final `lm` loss, HG − baseline)

| seed | baseline final lm loss | HG final lm loss | ΔNLL |
|---|---:|---:|---:|
| 0 (pilot) | 1.7586 | 1.7683 | +0.0097 |
| 1 | 1.7521 | 1.7608 | +0.0087 |
| 2 | 1.7941 | 1.8116 | +0.0175 |
| **mean ± std** | | | **+0.0120 ± 0.0048** |

Replicated: HG's language-modeling cost is small and consistent across
seeds, well inside "must not catastrophically damage ordinary LM
performance" (spec §10.1).

### 3.2 N0 capability floor (HG decision head vs. chance)

| seed | n | accuracy | balanced accuracy | constant-label collapse |
|---|---:|---:|---:|---|
| 0 | 2200 | 0.742 | 0.784 | False |
| 1 | 2200 | 0.764 | 0.754 | False |
| 2 | 2200 | 0.774 | 0.743 | False |
| **mean ± std** | | **0.760 ± 0.016** | **0.760 ± 0.021** | |

Healthy and stable across seeds — well above chance (0.5), no collapse.
**There is no baseline-model equivalent of N0**: a baseline has no decision
head to score at all (`eval.predict.predict()` requires `variant == "hg"` by
construction — the baseline never had a decision target to train against).
This is not a gap in this run; it is exactly the structural asymmetry the
rest of this report's vanilla *proxies* exist to work around.

---

## 4. N4 — quotient recovery: HG latent vs. matched vanilla controls

For each seed: `sequential_quotient` held-out episodes (n=500, seed 12345),
probe-fit split for PCA-64 drawn from a disjoint pool (seed 12845, "fit-seed"
= eval-seed + 500 — see kickoff §4.2's "do not fit PCA on the evaluation
split"). Both raw and PCA-64 controls read the vanilla model's own trunk
hidden state at the *same* decision-token position HG's latent is built
from (`eval.latents.extract_hidden_states`, added this pass — see §9).

### Required summary table

| seed | HG Q_sel | Vanilla raw Q_sel | Vanilla 64D Q_sel | Δ HG-vs-64D | 95% CI |
|---|---:|---:|---:|---:|---|
| 0 | 0.437 | −0.042 | −0.062 | 0.520 | [0.402, 0.559] |
| 1 | 0.472 | −0.084 | −0.067 | 0.543 | [0.417, 0.559] |
| 2 | 0.362 | −0.016 | −0.039 | 0.440 | [0.353, 0.496] |
| **mean ± std** | **0.424 ± 0.056** | **−0.047 ± 0.034** | **−0.056 ± 0.014** | **0.501 ± 0.054** | |

`ΔQ_sel_raw` (HG vs. raw, not shown in the required columns but computed
identically): mean 0.498 ± 0.038, seed values 0.504 / 0.532 / 0.457, all
95% CIs excluding 0.

**Replicated, large, and the CI excludes 0 in every seed** — the pilot's
single-seed 0.431 was not a fluke and is not explained by parameter count
or the extra 640→64 dimensionality reduction on its own (the PCA-64 control
undergoes the exact same reduction and still scores ≈0, in fact slightly
negative).

### 4.1 Per-variable breakdown — why the aggregate number is what it is

| seed | probe | HG acc | vanilla PCA-64 acc |
|---|---|---:|---:|
| 0 | r (relevant) | 1.000 | 0.953 |
| 0 | f (relevant) | 0.967 | 0.913 |
| 0 | h (relevant) | 0.893 | 0.907 |
| 0 | u (irrelevant) | 0.560 | **1.000** |
| 0 | n (irrelevant) | 0.473 | **0.973** |
| 1 | r | 0.947 | 0.940 |
| 1 | f | 0.993 | 0.900 |
| 1 | h | 0.887 | 0.900 |
| 1 | u | 0.460 | **1.000** |
| 1 | n | 0.480 | **0.960** |
| 2 | r | 1.000 | 0.953 |
| 2 | f | 0.987 | 0.940 |
| 2 | h | 0.900 | 0.900 |
| 2 | u | 0.633 | **1.000** |
| 2 | n | 0.567 | **0.940** |

This is the important nuance the aggregate Q_sel hides: **the vanilla
64D representation decodes `r`/`f`/`h` almost as well as HG does** (0.90–0.95
vs. HG's 0.89–1.00) — it is not an information-poor representation. What
drives its Q_sel to ≈0 is that it decodes the *irrelevant* variables `u`/`n`
**even better** (0.94–1.00, essentially perfect), while HG suppresses them to
near-chance (0.46–0.63). Framed against the kickoff doc's §7 pre-registered
outcomes, this is **Outcome B, not Outcome A**: vanilla pretraining learns a
rich, undifferentiated world state that happens to retain everything visible
in context (relevant and irrelevant alike); HG's auxiliary training
compresses the same trunk representation down to a task-relative decision
state that keeps what the task needs and actively discards what it doesn't.
Per the kickoff doc, **this counts as a successful, if more specific, result**
— not "HG sees more," but "HG organizes what both models can see."

---

## 5. N1 — information-deficit: HG head vs. vanilla confidence proxy

Vanilla proxy `g_info(x)`: a logistic regression (`eval.confidence.fit_logistic_proxy`)
over 4 label-free features (max next-token probability, top1-top2 margin,
next-token entropy, hidden-state norm — see §9.2 for why "action probability"
had to be redefined this way for a model with no action head), fit on a
disjoint `static_hidden` pool (fit-seed), evaluated on the same 600 held-out
`static_hidden` episodes HG is scored on.

| seed | HG AUROC | vanilla-proxy AUROC | ΔAUROC | 95% CI |
|---|---:|---:|---:|---|
| 0 | 1.000 | 0.674 | 0.326 | [0.287, 0.369] |
| 1 | 1.000 | 0.946 | 0.054 | [0.036, 0.073] |
| 2 | 1.000 | 0.915 | 0.085 | [0.065, 0.107] |
| **mean ± std** | 1.000 | 0.845 ± 0.148 | **0.155 ± 0.149** | |

Replicated (CI excludes 0 in all 3 seeds) but with real seed-to-seed
variance the pilot's single seed couldn't show: seed 0's gap is 4-6x larger
than seeds 1-2. HG is at ceiling (AUROC 1.0) every seed; the vanilla
confidence proxy is a genuinely non-trivial baseline (0.67-0.95, well above
chance) but never matches HG. This is the honest form of N1: HG wins
consistently, the *margin* is not stable across seeds, and pooling would
have hidden that.

---

## 6. N3 — accessibility: HG head vs. vanilla confidence proxy

Same construction as N1, `g_access(x)` fit/evaluated on `accessibility`
episodes (n=800 held-out, zero information-deficit by family construction).

| seed | HG AUROC | vanilla-proxy AUROC | ΔAUROC | 95% CI |
|---|---:|---:|---:|---|
| 0 | 1.000 | 0.980 | 0.020 | [0.013, 0.027] |
| 1 | 1.000 | 0.788 | 0.212 | [0.182, 0.242] |
| 2 | 1.000 | 0.997 | 0.003 | [0.001, 0.005] |
| **mean ± std** | 1.000 | 0.922 ± 0.114 | **0.078 ± 0.116** | |

Replicated (CI excludes 0 every seed) but the smallest and most
seed-variable of the three head-level comparisons — seed 1's vanilla proxy
happened to do much worse than seeds 0/2's. HG is again at ceiling every
seed. Directionally consistent, magnitude not stable.

---

## 7. N2 — routing: HG router vs. vanilla confidence router

Both routers make one fixed λ_B=λ_K=1 decision per episode (matching how
`eval.predict.predict()` actually calls the HG model — `c_t` defaults to
zero, so HG's own router is not re-decided per price point either), then
both fixed route assignments are scored across the full 8×8 price grid
(λ_B, λ_K ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}), reusing `eval.routing.route_value_report`
identically for both.

| seed | fraction of 64 cells HG's expected cost is lower |
|---|---:|
| 0 | 1.00 |
| 1 | 1.00 |
| 2 | 1.00 |

HG's routing wins on **every cell of the price grid, every seed** — a
stronger, cleaner result than the pilot's single-lambda oracle-matching
number. The vanilla confidence router (built from the same `g_info`/`g_access`
proxies as §5/§6, per the kickoff doc's spec-§13 "vanilla + confidence
router" control) is a real, non-trivial baseline — it is not a strawman
(§5/§6 show it clears 0.67-1.0 AUROC on its own) — and HG's router still
dominates it everywhere measured.

---

## 8. N5 — reuse (context-only, growing-context caveat preserved)

Per kickoff §16: M0's reuse mechanism is retained growing context, not
HG-only memory, unchanged from the pilot.

| seed | HG repeated-ask rate | vanilla-router repeated-ask rate |
|---|---:|---:|
| 0 | 0.000 | 1.000 |
| 1 | 0.000 | 1.000 |
| 2 | 0.000 | 1.000 |

A stark, fully replicated split: HG's router **never** re-asks for a fact
already disclosed earlier in the same trajectory's context (n=400 each
seed); the vanilla confidence router **always** does. This is not evidence
vanilla pretraining is incapable of noticing "this was already answered" —
it is evidence that the specific, generic 4-feature confidence proxy this
run built (§5's `g_info`, unchanged for later turns) never learns to. HG's
`info_deficit` head, trained with a direct supervised target, does. Read
this as "explicit supervision buys reliable reuse where a generic confidence
signal did not," not as "vanilla representations cannot support reuse in
principle" — the latter is a stronger claim than this experiment tests.

---

## 9. What changed in the code this pass, and two honest caveats

### 9.1 New/changed code (all additive; existing tests unchanged and still pass)

- `eval.latents.extract_hidden_states` — raw trunk `h_t` at the decision
  position, for either variant (kickoff §4.1).
- `eval.projections.fit_pca` / `fit_random_projection` — dimension-matched
  vanilla controls (kickoff §4.2).
- `eval.confidence.confidence_features` / `fit_logistic_proxy` — the vanilla
  `g_info`/`g_access` proxies (kickoff §8).
- `eval.vanilla_router.vanilla_route_predictions` — the confidence-based
  ACT/THINK/ASK router (kickoff §9-11), receiving no generator ground truth.
- `training.manifest.deploy_provenance` + `scripts/spark_deploy.sh`'s
  `write_deploy_provenance` — the git-SHA-survives-a-`.git`-less-rsync fix
  (kickoff §17); confirmed working in every new manifest this pass (§2).
- `training.trainer.train(..., log_interval=...)` — live per-step progress
  logging (kickoff §18); used for every run this pass (see `r1_pipeline.log`).
- `experiments/hg_native/run_eval_r1.py` — the new script that produces
  everything in §4-§8 from a baseline/HG checkpoint pair.
- 748 local (torch-free) + 806 full-suite (Spark, torch included) tests pass,
  up from 721/775 at the pilot — all new tests, zero regressions.

### 9.2 Confidence-feature substitution (a genuine interpretation call)

The kickoff doc's example confidence features ("max action probability,
action entropy, chosen-action probability...") presuppose a model that
produces action probabilities. A baseline model never does: it has no
action head, and `serialize_episode`'s decision position is the prompt's own
EOS token, never trained toward an action token either. `eval.confidence.py`
substitutes the analogous *next-token-distribution* statistics (max prob,
margin, entropy) plus hidden-state norm — computable for either variant,
using no ground truth. This is documented in the module's own docstring as a
substitution, not silently. It is a legitimate confidence signal (§5/§6 show
it clears well above chance) but it is not identical to what an
action-having model's confidence would look like, and a different feature
set might close some of §5-§6's remaining gap.

### 9.3 The pilot's seed-0 checkpoints were briefly lost, then recovered without retraining

Mid-run, `results/pilot_20260901/` was found deleted from the Spark's repo
checkout (present on 2026-09-01, gone by the time this run needed it on
2026-09-02 — cause not established; not caused by this session's own rsync,
which excludes `/results` from `--delete` consideration). Byte-identical
copies survived in two other places — this local checkout's own
`results/pilot_20260901/` (never git-tracked, but never deleted locally
either) and a `/tmp` scratch copy still on the Spark from the original
session — confirmed by matching MD5 checksums before use. Both seed-0
checkpoints were restored from there; **seed 0 was not retrained**, per the
kickoff doc's explicit instruction, and this incident does not affect any
number in this report. Recorded here per the instruction to stop and
document any issue found along the way, even one that didn't end up
blocking the protocol. Checkpoints (900MB+ across all seeds) remain
gitignored and Spark/local-only, as before.

---

## 10. Outcome call (kickoff §7)

**Outcome B: vanilla is representationally rich, HG is more selective.**
Not Outcome A (vanilla is not near-zero on the relevant variables — it's
close to HG on `r`/`f`/`h`) and not Outcome C (the gap is real, replicated,
and not explained by dimension-matching). Per the kickoff doc, Outcome B
"counts as a successful result": HG's auxiliary training demonstrably
reorganizes the shared trunk representation into a task-relative decision
state, on top of a vanilla trunk that already encodes the relevant variables
about as well, but without discarding the irrelevant ones.

## 11. Confirmatory success criteria (kickoff §22) — scored

1. Healthy N0 across seeds — **met** (§3.2).
2. Replicated positive ΔQ_sel_64 — **met**, all 3 seeds, CI excludes 0 (§4).
3. Effect not explained by dimension/parameter count — **met**: the PCA-64
   control undergoes the identical 640→64 reduction and still scores ≈0.
4. HG info/accessibility discrimination beats a reasonable vanilla
   confidence control — **met** on both, though N3's margin is seed-variable
   (§5, §6).
5. HG routing has lower expected cost over a nontrivial price-grid region —
   **exceeded**: 100% of cells, all seeds (§7).
6. Ordinary LM degradation small and quantified — **met**, +0.012 ± 0.005
   nats (§3.1).

**5 of 6 criteria cleanly met; the 6th (info/access discrimination) is met
on point estimates every seed but with a magnitude that varies 4-80x across
seeds** — worth flagging precisely because it's the kind of thing a
single-seed pilot cannot show.

## 12. Scale decision (kickoff §23)

Per §23: GO requires at least 2 of {grounding-awareness advantage,
accessibility-separation advantage, dynamic quotient/reuse advantage},
replicated, with matched vanilla controls and no unexplained LM degradation.

**This run supports all three**, replicated:
- grounding-awareness (N1): replicated, ΔAUROC > 0 all seeds (§5)
- accessibility-separation (N3): replicated, ΔAUROC > 0 all seeds (§6)
- reuse (N5): replicated, stark and consistent (§8)

with matched vanilla controls throughout and LM degradation quantified and
small (§3.1).

**Recommendation: GO to ~150M (spec §23's M1 preview)**, carrying forward
two open items rather than treating them as blockers:
- N1/N3's seed-to-seed magnitude variance (§5, §6) — worth a wider
  confidence-feature set or more seeds before trusting a specific effect
  size, not before trusting the *direction*.
- §9.2's confidence-feature substitution is a real design choice, not a
  spec-mandated one — an ablation on feature choice would strengthen how
  much weight the N1/N3 margins specifically can bear.

Neither open item is a HOLD-level "causal source unclear" concern (kickoff
§23's HOLD criterion): the N4/N2/N5 results are large, structural, and
directionally rock-solid across seeds; only two of six numbers have
wide seed-to-seed spread, and both still point the same direction every time.
