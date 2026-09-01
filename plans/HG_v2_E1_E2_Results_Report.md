# HG-v2 — E1, E2 and E3 results

**For:** the theory and planning track
**Basis:** `plans/HG_v2_Improved_Experiment_Plan_FIXED.md`, sections 4–10 and 19
**Date:** 2026-09-01

Self-contained: every number below is reproduced from a saved run directory,
and nothing requires reading the code.

---

## 1. Branch and commit

| | |
|---|---|
| branch | `claude/hg-v2-e1-e2` (from `claude/hg-v2-phase0-phase1`) |
| commit | `325192d` — *Add E1 and E2/E3 from the improved experiment plan* |
| test suite | 429 passed, 2 skipped (`uv run python -m pytest`) |
| E1 run | `results/20260901T114119Z_hg_v2_e1_frontier` (no GPU, ~1 min) |
| E2/E3 run | `results/20260901T120721Z_hg_v2_e2_integrated_qwen3b` (DGX Spark, ~32 min) |

One provenance caveat, stated rather than hidden: the E2 run was launched
before the commit, so its `manifest.json` records the *previous* commit. Every
file on its execution path was verified byte-identical (sha256) to the
committed version, so the run did execute `325192d`'s code. The E1 run was
re-run after committing and its manifest is correct.

---

## 2. Files added and modified

**New:** `src/hg/grounding/bounded_query.py` (E1), `src/hg/grounding/integrated.py`
(E2/E3), `src/hg/metrics/calibration.py`, `src/hg/queries/quality.py`,
`experiments/hg_v2/run_e1.py`, `experiments/hg_v2/run_e2.py`, three configs,
three test modules.

**Modified, all additively:** `hg.grounding.config` (two task kinds, two gate
types), `hg.inference.prompts` (two query-synthesis templates),
`hg.queries.lm_query_synthesizer` (variant registry, cap policy),
`hg.routing.decision_state` / `grounding_router` (the feature-name list is now
a parameter, so a Phase 1 router's saved weights keep their meaning),
`hg.queries.synthetic` (a reproducibility fix, §12), `experiments/hg_v2/analyze.py`,
the README, the reuse audit, the Makefile.

---

## 3. Gate H0 — theorem-frontier validation

**PASSED**, 374 checks. `results/20260901T114119Z_hg_v2_e1_frontier/report.md`.

Every `(n, r)` for `n ∈ {8,16,32,64}` and `r = 0..⌈log₂n⌉` was simulated at
`b*(n,r) = ⌈n/2^r⌉` bits and again at `b*−1`. The `n = 64` row, which is the
table in plan §4:

| r | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b*` predicted | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
| `b` observed | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
| accuracy at `b*` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| accuracy at `b*−1` | 0.994 | 0.984 | 0.969 | 0.938 | 0.870 | 0.748 | 0.495 |

Phase 0's two results are the endpoints of this curve, so the old result is
contained in the new one rather than replaced.

Two methodological points worth carrying back into the theory:

**The converse control needs its own item count.** One bit short of the bound
fails with probability `(s−b)/2s`, which is 1/128 at `n=64, r=0`. At a fixed
400 items there is a 4% chance of observing no failure at all and reporting the
bound as *not* tight. Each cell is now sized to buy 20 expected failures, so
"tight here" is backed by equal evidence at every point of the frontier.

**Truncation alone does not establish the converse.** Running one particular
`b*−1` protocol leaves open the objection that a cleverer code might succeed.
That is closed separately by pigeonhole, tested over random encoders: whenever
the message alphabet is smaller than `2^s`, two fiber assignments collide, and
two that collide differ at some coordinate — which is an index the AI can be
holding.

### E1b — the lossy frontier

The three claims are kept apart, and only one is gated.

| quantity | status in this run |
|---|---|
| finite converse `b ≥ s(1−h₂(ε))` | exact at every blocklength; **gated**; respected in all 110 cells |
| asymptotic rate `n·2^{−r}(1−h₂(ε))` | reported, **not** gated — at `s ≤ 10` it is not a prediction about any single cell |
| achievability | best-found codebook; an **upper** bound on `b*`, never a claim of finite-length optimality |

The finite-blocklength penalty behaves as the asymptotic claim would suggest
without the run being able to confirm the limit: at `ε = 0.10` the gap between
the best-found code and the converse is 0.469 bits per source symbol at `s ≤ 4`
and 0.219 at `s = 8`.

---

## 4. Gate H1 — integrated routing benefit

**NOT MET.** 1 of 10 prices, 2 required.

Frozen `Qwen2.5-3B-Instruct`, 1000 items, 464 fitting the router and 536
scoring it. λ = 0.15:

| arm | acc | acc\|A | acc\|B | bits | ask rate | sound | waste | C = L+λB | regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `b0_no_grounding` | 0.3470 | 0.095 | 0.588 | 0.000 | 0.000 | 0.502 | 0.000 | 0.6530 | +0.393 |
| `b1_self_consistency` | 0.3601 | 0.095 | 0.613 | 0.000 | 0.000 | 0.502 | 0.000 | 0.6399 | +0.380 |
| `b2_generic_ask` | 0.7575 | 0.752 | 0.763 | 5.000 | 1.000 | 1.000 | 2.509 | 0.9925 | +0.453 |
| `b3_oracle_route_oracle_query` | 0.8097 | 0.779 | 0.839 | 0.463 | 0.463 | 0.744 | 0.220 | 0.2597 | 0.000 |
| `b4_always_ask_synth` | 0.7593 | 0.767 | 0.752 | 2.440 | 1.000 | 1.000 | 1.215 | 0.6067 | +0.237 |
| `b5_learned_route_synth` | 0.7593 | 0.767 | 0.752 | 2.440 | 1.000 | 1.000 | 1.215 | 0.6067 | +0.237 |
| `b6_learned_route_oracle_query` | 0.7649 | 0.779 | 0.752 | 1.000 | 1.000 | 1.000 | 0.502 | 0.3851 | +0.125 |
| `b7_oracle_route_synth` | 0.8041 | 0.767 | 0.839 | 1.159 | 0.457 | 0.743 | 0.541 | 0.3697 | 0.000 |

**B5 is identical to B4 at every price up to λ = 0.15** — the learned router
asks on every item — and at λ ≥ 0.30 it acts on every item, where the bit saving
is large but final distortion collapses to B0's. The one winning price, λ = 0.20,
saves 0.093 bits [0.049, 0.149]. The router is a step function on this task: it
never produces an intermediate policy.

---

## 5. Why H1 failed — two separable causes

### 5.1 Asking helps even where it is provably unnecessary

The finding that reframes the gate. Computed from the counterfactuals every
episode carries:

| items | n | error acting | error asking | value of asking |
|---|---:|---:|---:|---:|
| withheld variable decides | 267 | 0.487 | 0.004 | **+0.483** |
| prompt already decides | 269 | 0.818 | 0.476 | **+0.342** |

On items where `H(J | M_A) = 0` — the rule closes from the prompt alone, and the
traveller's preferences are information-theoretically irrelevant — disclosing
those preferences still cuts the model's error rate from 0.818 to 0.476.

Part of grounding's value at this capability is therefore **prompt scaffolding,
not information**. A router that correctly declines to ask on those items pays
for it in distortion. Pricing the four policies from the same episodes:

| λ | never | always | structural | clairvoyant | structural wins |
|---:|---:|---:|---:|---:|:---:|
| 0.05 | 0.6530 | 0.3627 | 0.4736 | 0.2538 | no |
| 0.10 | 0.6530 | 0.4847 | 0.5349 | 0.3118 | no |
| 0.15 | 0.6530 | 0.6067 | 0.5962 | 0.3697 | yes |
| 0.30 | 0.6530 | 0.9728 | 0.7800 | 0.4642 | yes |

`structural` asks exactly where the deficit is nonzero — the best *any* router
could do from AI-visible state, since `needs_grounding` is determined by the
prompt on this task. It does not beat always-ask below λ = 0.15, and where it
does, it wins by 0.010 at λ = 0.15. **There was almost nothing for a router to
recover at this model's capability**, whatever estimator it used.

Note also that this makes the oracle-route arms (B3, B7) weaker references than
they look: the gap between `structural` and `clairvoyant` is how much of their
advantage is clairvoyance about realized noise rather than exploitable structure.

### 5.2 The LM's internal signal carries no routing information

Independently, and confirming falsification **F1**:

| | |
|---|---|
| Brier | 0.2223 |
| base-rate variance (a constant predictor) | 0.2266 |
| ECE / MCE | 0.0370 / 0.2412 |
| Spearman(p̂, V_ask) | 0.062–0.180 |
| p̂ mean / sd / range | 0.666 / 0.059 / [0.405, 0.743] |
| p̂ on items needing grounding | 0.6699 |
| p̂ on items not needing grounding | 0.6619 |

The estimator is well *calibrated* to the base rate and has essentially no
*discrimination*: its Brier score improves on a constant predictor by 0.004. The
truth it should track goes the other way — acting is wrong 48.7% of the time on
items that need grounding and 81.8% on items that do not — and p̂ separates those
two populations by 0.008 in the wrong direction.

This is exactly the split plan §2.3 asks to be measured, and it is why routing
regret alone would have been uninformative: a flat estimator and a good one can
produce the same route at most prices.

### 5.3 Factorial decomposition (B4–B7), λ = 0.15

| quantity | bits |
|---|---:|
| routing gap `B5 − B7` | +1.282 |
| synthesis gap `B5 − B6` | +1.440 |

Both gaps are large, so neither ingredient of B5 was the sole problem — but §5.1
says the routing half of that is largely unreachable and §6 says the synthesis
half is fully reachable, and was reached.

---

## 6. Gate H2 — minimality interventions

**PASSED.** All three interventions improved, and the result is stronger than
the gate required.

Comparator `q0_baseline` is the unmodified synthesizer, 2.440 bits, soundness
1.000, accuracy 0.7593. All arms ask on every item, so the query is the only
thing that differs.

| variant | bits | saved vs Q0 | 95% CI | sound | Δsound | acc | ΔD | ΔD 95% CI | η_Q | α minimal | full escalation |
|---|---:|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| `q0_baseline` | 2.440 | — | — | 1.000 | — | 0.7593 | — | — | 0.640 | 0.300 | 0.321 |
| `q1_minimality` | 1.817 | +0.623 | [+0.515, +0.739] | 1.000 | 0.000 | 0.7593 | +0.0000 | [0, 0] | 0.796 | 0.386 | 0.201 |
| `q2_budget` | **1.000** | +1.440 | [+1.287, +1.582] | 1.000 | 0.000 | 0.7649 | −0.0056 | [−0.0131, 0] | **1.000** | 0.498 | 0.000 |
| `q3_hard_cap` | **1.000** | +1.440 | [+1.287, +1.582] | 1.000 | 0.000 | 0.7649 | −0.0056 | [−0.0131, 0] | **1.000** | 0.498 | 0.000 |

(η_Q here is the plan §1.3 form, `(B_generic − B_arm)/(B_generic − B_oracle)`
with `B_generic = 5.000` and `B_oracle = 1.000`. The run's `analysis.json` also
reports an η_Q against B3, whose oracle *route* puts its cost below one bit.)

**One sentence of prompt closed the entire remaining gap.** "You may ask about
at most one of the traveller's preferences" takes the synthesized query from
2.440 bits to exactly 1.000 on every one of 536 items, at soundness 1.000 — the
model names one variable and it is always the deciding one. η_Q goes from 0.640
to 1.000: the synthesized query is now the oracle query.

Three further observations:

- **Enforcement was unnecessary.** Q3 (budget + structural truncation) is
  byte-identical to Q2 (budget, unenforced) in every metric: the truncation
  never fires because the model already complies. Falsification F2 — that
  minimality pressure breaks soundness — did not occur, and could not have,
  since no query needed trimming.
- **Decision quality improved slightly** (−0.0056 loss, CI [−0.0131, 0]), so
  this is not a cost/quality trade.
- **`α minimal` is not the share of one-variable queries.** A query counts as
  minimal only when its cost is at or below that item's `B*`, and `B*` is zero
  on an item whose prompt already decides — so a one-variable query there is
  still above the optimum. That is why Q2 shows `α = 0.498` with *every* query
  naming one variable: 0.498 is essentially the share of items that needed
  grounding at all. The remaining half is not a synthesis failure; it is the
  routing failure of §5, counted in the query metric.
- **The mixture moved as H-C predicts.** Query cardinality distributions:
  Q0 `{1: 322, 3: 42, 5: 172}` → Q1 `{1: 426, 4: 2, 5: 108}` → Q2/Q3 `{1: 536}`.
  The full-escalation mode drains from 0.321 to 0.201 to 0.000. The Phase 2
  bimodality was a fallback-policy effect, and it is removable by instruction.

---

## 7. Query-cost distribution and waste rate

`b4_always_ask_synth` at λ = 0.15: mean 2.440 bits, distribution over variables
named `{1: 322, 3: 42, 5: 172}`, soundness 1.000, exact-gold rate 0.601.

Waste — bits spent on items the prompt already determined — is 1.215 bits per
item, **49.8% of all bits spent**. Under Q2 it falls to 0.502 bits per item.

The honest caveat is §5.1: on this model those bits are not decision-value
waste, because they buy +0.342 accuracy on exactly those items. The waste rate
remains the right *information* accounting and the wrong proxy for what routing
can recover at 3B.

---

## 8. Replication of gate G2

At λ = 0.15, `b4_always_ask_synth` versus `b2_generic_ask`:

| | previous G2 run | this run |
|---|---:|---:|
| generic escalation | 5.00 bits | 5.000 bits |
| synthesized query | 2.52 bits | 2.440 bits |
| oracle query | 1.00 bits | 1.000 bits |
| η_Q | 0.62 | 0.640 |
| final loss, synth vs generic | equal | 0.2407 vs 0.2425 |

**Falsification F3 did not occur**: the G2 compression result replicates on a
fresh run of the corrected task, with an independent train/eval split and a
different item sample.

---

## 9. Capability floor (plan §2.4)

`Qwen2.5-3B-Instruct` **passes**, but with one result that matters for
interpretation.

| criterion | observed | verdict |
|---|---|---|
| fully grounded accuracy materially above chance | 0.765 | pass |
| balanced across correct-action classes | 0.779 / 0.752 | pass |
| grounding materially improves | 0.347 → 0.765 (+0.418) | pass |
| constant-label behaviour absent | says A 0.507 when grounded | pass |

The result to carry forward: **ungrounded accuracy is 0.347, below chance on a
binary task**, with `acc|A = 0.095` and "says A" of 0.257. Without the
constraints block the model has a strong label bias; with it, the bias
disappears and both classes land near 0.76. This is the same phenomenon as
§5.1 seen from the other side, and it is the mechanism behind H1's failure.

---

## 10. Scoring-infrastructure audit (plan §16)

The `_label_logprob` fix from the G2 run now has an executable regression:
`tests/test_label_scoring.py` checks, per chat template, that the bare label is
the correct continuation after templating, that the spaced form is correct for a
raw prompt, and that appending a label does not re-tokenize the prompt at the
seam. It passes against the real `Qwen2.5-3B-Instruct` and `Qwen2.5-1.5B-Instruct`
tokenizers on the Spark; it skips where `transformers` or the checkpoint is
unavailable.

No new scoring defects were found. One unrelated reproducibility defect was:
see §12.

---

## 11. Assessment — is the HG-v2 architecture hypothesis stronger or weaker?

**Mixed, and the two halves moved in opposite directions.**

**Stronger on query synthesis.** The claim that a system can identify residual
decision-relevant state and ask for it minimally is now *fully* demonstrated on
a frozen off-the-shelf model: 5.000 → 1.000 bits at soundness 1.000 and no loss
of decision quality, η_Q = 1.000. There is no remaining oracle gap to close on
this task. That is a stronger result than G2's 1.98×, and it came from one
sentence of instruction rather than from training.

**Weaker on routing — as an empirical matter at 3B, not as a theory.** The
architecture separates *whether* to ask from *what* to ask, and the second
module now works perfectly while the first has nothing to work with. Two
independent reasons, and it is important that they are independent:

1. The model's decision quality depends on being given information it does not
   informationally need (+0.342 accuracy on items with zero residual deficit).
   This is a confound in the `C = L + λB` objective itself: the objective
   assumes the grounded answer's value is its information content, and at this
   capability that assumption is false. No router can be blamed for it.
2. The frozen model's internal signals carry essentially no information about
   whether acting now is wrong (Brier 0.2223 against a constant predictor's
   0.2266). This is F1, and it is a fact about 3B-scale introspection.

The plan's own falsification list anticipated (2) and not (1). I would treat (1)
as the more consequential finding: it predicts that the *routing* half of HG-v2
only becomes measurable on a model whose rule-following does not improve from
irrelevant context — which is a capability threshold, and a testable one.

---

## 12. Defects found and fixed

- **`NoisyQuerySynthesizer` was not reproducible across processes.** It keyed
  its RNG on `hash(item_id)`, and Python salts string hashing per process, so
  two runs of the same config produced different query sets. The arms stayed
  internally consistent and the saved-artifact round-trip still reproduced, so
  nothing failed loudly — but between-run comparisons were invalid. Now a
  stable digest; verified identical across `PYTHONHASHSEED`. This affected the
  simulated Phase 2 arm only, never the LM synthesizer or any gate result.
- **Routing regret was degenerate for the ACT-only arms** in the first cut of
  the E2 code: with no ask counterfactual, never-asking is trivially optimal
  and B0's regret was 0 by construction. The oracle query is now used as their
  ask reference. Found before the reported run, which was restarted.

---

## 13. Recommendation on the 7B model

**Run it, and change what it is for.**

The original motivation — "does the grounding requirement fall with capability"
— is now secondary. The sharper question this run produced is:

> Does the value of information-theoretically irrelevant context fall with
> capability?

Measure, on the same task and with no new machinery (the arms already exist):

- `value_of_asking` on items with zero residual deficit. At 3B it is +0.342. If
  it approaches 0 at 7B, routing becomes measurable and H1 can be retested
  honestly. If it does not, H1 is not a well-posed gate on this task family at
  any scale and the task needs changing, not the model.
- Ungrounded accuracy and label bias. 3B is *below chance* ungrounded; a model
  that is merely at chance would already change the routing picture.
- Whether Q2's one-line budget still yields exactly the gold query. If η_Q = 1.0
  replicates at 7B, H4 (cross-model persistence) is satisfied for the synthesis
  half on the strength of two capable models.

Cost is about 35 minutes on one GB10 for the full 12-arm run at 1000 items.

**Do not** begin H6 (LoRA, learned critics, joint training). H1 is unmet, and
the reason is a property of the model and the task rather than of the modules
that would be trained.

---

## 14. What is outstanding

1. **H3/H4 — the 7B run**, reframed as above.
2. **H5 — a second task family.** Now more urgent than it was: §5.1 is a
   property of *this* task's prompt structure, and whether it survives a
   different causal structure is the question that decides whether H1 is
   retestable at all. A task where the irrelevant variables are not
   presentable as a "constraints" block would isolate scaffolding from
   information.
3. **A router with a usable signal.** If §5.1 resolves at 7B, the estimator
   still has to improve. The features here are the model's own vote agreement
   and label distribution; neither discriminates. Worth trying before any
   training: the AI-visible structural feature (`concession_faster`) as a
   diagnostic ceiling, and the model's own *stated* uncertainty as a feature.
