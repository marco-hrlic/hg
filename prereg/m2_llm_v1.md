# M2-LLM preregistration, v1

Protocol: *M2-LLM Experimental Protocol* v1.0, section 11.

**Primary question.** Does a clean systematic residual remain after the best
practical use of internal model-family and data diversity?

Written before Phase 1 is run. Analysis code implementing every quantity below
is in `src/hg/`; the mapping is given per prediction so the numbers cannot be
chosen after the fact.

---

## Design being preregistered

| | |
|---|---|
| Ensemble size | E = 5 |
| Primary comparison | HS → DD |
| Conditions | HS (one family, shared slice, seeds vary), DD (five families, disjoint slices) |
| Evaluation items | ≥ 2,000 held-out, target 2,000–5,000 |
| Anchors | raw annotator panels, ≥ 3 annotators per item |
| Primary subset | `A_i = unanimous` |
| Adaptation | LoRA, rank 16, one recipe across families, equal per-member example budget |
| Inference | greedy; label-ranking scoring where available |
| Resampling | item-level bootstrap, paired for comparisons, 2,000 draws |

## Primary outcome

`C_DD^clean` = P(all five members wrong | anchor unanimous), with a paired
bootstrap CI. Reported alongside `C_HS^clean` and the gap.

## Predictions

| # | Statement | Decision rule fixed in advance | Implementation |
|---|---|---|---|
| **P1** | Diversity reduction: `C_DD < C_HS` | Supported iff the paired bootstrap CI for `C_HS^clean − C_DD^clean` excludes zero and is positive | `metrics.bootstrap.paired_bootstrap` |
| **P2** | A clean residual is measurable: `C_DD^clean > 0` | Supported iff the bootstrap CI for `C_DD^clean` excludes zero. **Magnitude is deliberately not preregistered** (protocol §11) | `metrics.bootstrap.bootstrap_metric` |
| **P3** | Contested concentration: `Q_contested` rises with internal diversity | Supported iff `Q_contested(DD) > Q_contested(HS)` | `metrics.core.contested_share_of_residual` |
| **P4** | Error conversion: family diversity converts consensus-wrong into disagreement more than it merely removes error | Supported iff (gain in clean `D_E`) / (reduction in `C^clean`) ≥ 1 | `metrics.core.error_conversion` |
| **P5** | Active learning under diversity: disagreement sampling matches or beats generic anchored targeting unless a clean shared core survives | Supported iff localization surplus ≤ +0.05 at the median budget | `routing.strategies.run_race` |
| **P6** | Conditional failure-memory benefit: if a large clustered clean core survives DD, targeted anchored correction beats disagreement sampling | Evaluated **only** at gate C or D; supported iff surplus > +0.05 there | `routing.strategies.LocalizationRace.interpret` |

## Decision gates (protocol §16)

| Gate | Condition | Next action |
|---|---|---|
| A | `C_DD^clean ≈ 0` (≤ 0.002, or CI includes zero) | Heterogeneous internal ensembles + adjudication for contested items |
| B | `> 0` but still falling with added diversity | Extend the diversity axis; do not claim an information-limited floor |
| C | Stable clean residual survives (curve has flattened) | Test localization and targeted external correction |
| D | Residual survives and grows relatively with capability | Scaling analysis of the information and harm races |

Gate assignment is computed by `report.decision.decision_gate`, not chosen by
hand.

---

## Two commitments added during implementation

These are recorded here because both were discovered while building the
analysis, before any real run, and both constrain how the results may be read.

### 1. The survival-curve plateau test is not sufficient at E = 5

The protocol reads the residual survival curve's shape to separate gate A from
gate C. Implementation showed that at E = 5 the curve has **not flattened in
either case**. On the simulator control (`configs/experiments/sim_discrimination.yaml`),
with a genuine information-limited core versus none at all:

| | with a real core | no core |
|---|---|---|
| `C_5^clean` | 0.030 | 0.015 |
| trailing relative slope | 28% | 44% |
| survival shape | still_falling | still_falling |

Both arms return gate B. The shape test cannot separate them, and the endpoint
level differs by only a factor of two — not enough to rule on.

**Commitment.** Every reported clean residual is accompanied by
`excess_verdict`: the observed survival curve compared against the fitted
hierarchical model's own curve over the same family subsets. That model has one
shared item-difficulty factor with conditional independence given it, so its
`C_k` must fall to zero as `k` grows — it cannot represent a floor. On the same
control:

| | with a real core | no core |
|---|---|---|
| observed/predicted ratio, k = 1…5 | 1.00 → 1.42 | 1.00 → 1.06 |
| ratio growth | 0.42 | 0.06 |
| verdict | residual exceeds shared difficulty | consistent with shared difficulty |

A nonzero `C_5^clean` carrying `consistent_with_shared_difficulty` will **not**
be reported as a candidate residual, however nonzero it looks.

### 2. P5 and P6 are not testable on a binary task

With two labels, "every member is wrong" and "every member agreed" are the same
event. A consensus failure therefore has zero vote entropy, and the
disagreement scorer ranks its own targets last — so any strategy beats
disagreement sampling for arithmetic reasons rather than epistemic ones.

**Commitment.** P5 and P6 are evaluated only on task families with ≥ 3 labels
(§5.2 families B and E qualify; a binary harmfulness label does not). On binary
tasks the race is reported with `verdict: not_applicable`, and the informative
comparison is uncertainty versus failure memory. The harmfulness continuity
benchmark (§5.1) therefore tests P1–P4 only.

---

## Analyses that are exploratory, not preregistered

- Decomposition of the endpoint gap into data-overlap and family components (HD, DS)
- Within- versus between-family error correlation
- Any localization race on tasks below gate C
- Small-model economics (§17); depends on a persistent residual being found first

## Deviations

Record any deviation from this document here, dated, with the reason, before
looking at the affected result.

| Date | Deviation | Reason |
|---|---|---|
| — | — | — |
