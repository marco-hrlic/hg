# human-grounding — M2-LLM

Implementation of the *M2-LLM Experimental Protocol* v1.0.

**The question.** Does a clean systematic residual remain after the best
practical use of internal model-family and data diversity? That is the
empirical hinge between "diversify internally" and "acquire genuinely external
grounding" — and the v0.4 result made it live, because apparent common-mode
error may be information-limited or may just be an artifact of internal
homogeneity.

**The design.** Five model families, controlled training-data overlap,
human-anchored evaluation. Diversity is increased in steps and what remains is
measured: disagreement, clean consensus failure, contested anchors, and the
residual value of external targeting.

> Research discipline: first exhaust internal information → measure what
> genuinely remains → spend external information on that residual.

---

## Quick start

```bash
make install          # analysis pipeline, CPU only, no model weights
make test             # 162 tests, ~7s
make phase0           # the three-family infrastructure pilot (§19)
```

Phase 0 runs on a simulator backend and needs no GPU. It exists to validate the
pipeline end to end and to check the analysis against a known ground truth
before any GPU time is spent. **Nothing it prints is evidence about LLMs.**

For the real experiment on a DGX Spark:

```bash
make install-gpu
hg validate --config configs/experiments/m2_llm_endpoints.yaml   # check the panel first
hg run      --config configs/experiments/m2_llm_endpoints.yaml
```

---

## What this repo does

Everything is driven by config files, never notebook state. A run writes a
self-contained directory:

```
results/<timestamp>_<experiment>/
├── config.resolved.yaml     the fully inlined config that produced this run
├── manifest.<cond>.json     members, checkpoints, adapter hashes, measured O_jk
├── diagnostics.json         split sizes, overlap checks, adapter bookkeeping
├── predictions/<cond>.jsonl item-level rows  ← the artifact that matters
├── matrices/<cond>.npz      derived error matrices (a cache, not the truth)
├── bootstrap/               resampled metric draws
├── figures/                 plots, with CSV fallback if matplotlib is absent
├── analysis.json            every computed quantity
└── report.md                warnings, decision gate, evidence
```

The prediction matrix is the expensive artifact, not the summary plot (§7). So
`hg analyze --run-dir <dir>` re-runs the whole analysis over saved rows with no
model calls — a new dependence measure costs seconds, not GPU-hours.

### Layout

| Path | Role |
|---|---|
| `configs/models/` | one file per family: checkpoint, backend, lineage |
| `configs/datasets/` | task, label space, anchor source, split policy |
| `configs/experiments/` | conditions, seeds, bootstrap draws |
| `src/hg/inference/` | backends (simulator, transformers, vLLM), versioned prompts, canonicalization |
| `src/hg/finetune/` | LoRA with a common per-member budget, adapter cards |
| `src/hg/anchors/` | raw panels → reference label + anchor status |
| `src/hg/diversity/` | splits, overlap `O_jk`, residual survival |
| `src/hg/metrics/` | §8 metrics, dependence models, bootstrap |
| `src/hg/routing/` | localization strategies and the targeting race |
| `src/hg/report/` | decision gates, prediction verdicts, figures |
| `experiments/m2_llm/` | phase runners |
| `prereg/` | preregistered predictions and decision rules |

The protocol's `src/{inference,finetune,...}` layout is kept as
`src/hg/{inference,finetune,...}` so the code is an installable package;
subsystem names are unchanged.

---

## The four conditions (§4)

| | same family | five families |
|---|---|---|
| **shared slice** | `HS` | `DS` |
| **disjoint slices** | `HD` | `DD` |

Run the endpoints first (`HS` → `DD`), then fill in `HD` and `DS` only if the
gap is informative. `configs/experiments/m2_llm_endpoints.yaml` and
`m2_llm_decomposition.yaml` respectively.

Two things the code enforces rather than trusting:

- **Lineage, not name, defines a family.** Five variants of one foundation
  model are not diversity (§3). `hg validate` warns when a `DS`/`DD` panel
  reuses a lineage.
- **`O_jk` is measured, not declared.** Jaccard overlap of the actual training
  slices goes into every manifest, and a "disjoint" condition whose overlap is
  nonzero is flagged — otherwise data-sharing dependence gets attributed to
  representation sharing.

## Anchors (§6)

Panels are stored as raw annotator judgments, one JSONL row per item:

```json
{"item_id": "h-0001", "x": "...", "judgments": ["harmful", "harmful", "not_harmful"]}
```

Never a collapsed majority label. The reference label `y_i^ref` and the anchor
status `A_i ∈ {unanimous, majority, contested}` are derived from the panel, and
`C_E^clean` — the primary residual proxy — conditions on `unanimous`. Without
the raw panel that subset cannot be reconstructed, and a residual made of human
disagreement is indistinguishable from one made of missing information.

## Metrics (§8)

`C_E`, `C_E^clean`, `D_E`, `Q_contested`, `Δ_div`, localization surplus, plus
the invisible-error rate (wrong *and* internally silent).

Two deliberate choices:

- **An empty subset returns NaN, never 0.** "No unanimous items survived the
  filter" and "no unanimous item defeated the ensemble" are opposite
  conclusions.
- **Disagreement is defined on predictions, not correctness.** Members can be
  wrong the same way (invisible) or differently (addressable). Separating those
  is what P4 is about.

## Dependence (§9)

The naive independence null `∏ p_j` is reported for context only. The
inferential baseline is a hierarchical model with the item factor integrated
out:

```
E_ij ~ Bernoulli(σ(α_j + u_i)),   u_i ~ N(0, σ_u²)
```

fitted by marginal MLE (Gauss–Hermite quadrature). It already contains unequal
model quality and ordinary correlated difficulty, so `excess_over_hierarchical`
is the part neither explains.

The joint per-item fit is also available but is **not** the baseline: it
estimates one `u_i` from `E` observations, and since predicted consensus is
convex in difficulty, that estimation noise inflates it — masking a real
residual, which is the wrong direction to be wrong in. On simulated null data
the marginal fit lands within 0.001 of the observed rate where the joint fit is
0.020 high.

---

## Two findings from building this

Both constrain how results may be read, and both are recorded in
`prereg/m2_llm_v1.md`.

### The survival-curve plateau test is not sufficient at E = 5

The protocol reads the curve's shape to separate gate A from gate C. On the
simulator control (`scripts/run_discrimination_control.py`), with a genuine
information-limited core versus none at all:

| | real core | no core |
|---|---|---|
| `C_5^clean` | 0.030 | 0.015 |
| trailing slope | 28% | 44% |
| shape verdict | still_falling | still_falling |

Both return gate B. The curve has not flattened in either case, so the shape
test cannot tell them apart, and the endpoint differs by only a factor of two.

What does separate them is comparing the observed curve to the *fitted
hierarchical model's own* curve. That model has one shared difficulty factor
with conditional independence given it, so its `C_k` must fall to zero as `k`
grows — it cannot represent a floor:

| | real core | no core |
|---|---|---|
| observed/predicted, k = 1…5 | 1.00 → 1.42 | 1.00 → 1.06 |
| ratio growth | 0.42 | 0.06 |
| `excess_verdict` | residual exceeds shared difficulty | consistent with shared difficulty |

Every reported residual carries `excess_verdict`. A nonzero `C_5^clean` marked
`consistent_with_shared_difficulty` is not a candidate residual, however
nonzero it looks.

### P5 and P6 are not testable on a binary task

With two labels, "every member is wrong" and "every member agreed" are the same
event. A consensus failure has zero vote entropy, so the disagreement scorer
ranks its own targets *last* — any strategy beats disagreement sampling for
arithmetic reasons rather than epistemic ones.

The race therefore reports `verdict: not_applicable` on binary tasks, and P5/P6
must be evaluated on task families with ≥ 3 labels (§5.2 families B and E;
`configs/datasets/policy_application.yaml` is set up for this). The harmfulness
continuity benchmark tests P1–P4 only.

---

## Phases (§15)

| Phase | Config | Exit criterion |
|---|---|---|
| 0 — infrastructure | `phase0_infra.yaml` | schema, overlap, parsing, memory, dependence metrics all validate |
| 0 — controls | `sim_discrimination.yaml` | the analysis separates a simulated core from simulated shared difficulty |
| 1 — endpoints | `m2_llm_endpoints.yaml` | measure `Δ_div`; decide whether to expand |
| 2 — decomposition | `m2_llm_decomposition.yaml` | separate data-overlap from family effects |
| 3 — multi-task | per-task dataset configs | are clean residuals task-specific or systematic |
| 4 — localization race | only at gate C | compare disagreement, audit, failure memory, oracle |
| 5 — small-model economics | — | error versus total economic cost |

`run_phase0.py` returns a non-zero exit code when an exit criterion fails, so
Phase 0 is a gate and not a suggestion.

## Decision gates (§16)

| Gate | Condition | Next action |
|---|---|---|
| A | `C_DD^clean ≈ 0` | heterogeneous internal ensembles + adjudication for contested items |
| B | `> 0`, still falling | extend the diversity axis; do not claim a floor |
| C | stable residual, curve flattened | test localization and targeted external correction |
| D | residual grows with capability | scaling analysis of the information and harm races |

Gate C is what would license building the anchored failure-memory
architecture, so it is deliberately hard to reach: a nonzero endpoint on a
falling curve is gate B, and gate D needs a capability trend across scales,
never a single-scale run.

## Running on the DGX Spark

```bash
docker build -f docker/Dockerfile -t hg:m2 .
docker compose -f docker/compose.yaml run --rm hg run \
  --config configs/experiments/m2_llm_endpoints.yaml
```

A five-family panel at 8–14B does not fit resident on one Spark, so the runner
loads one family at a time and `unload()` actually frees the device. The default
scoring mode ranks the allowed labels by continuation log-probability instead of
generating, which removes the parser as a source of cross-family disagreement
(§18) and gives every family a real class distribution. vLLM is available for
throughput but does not release GPU memory in-process — use one family per
container invocation with that backend.

## Before the first real run

- [ ] Place the v0.4 harmfulness panel at `data/anchors/harmfulness_v04_panel.jsonl`, with raw annotator judgments
- [ ] Confirm the five checkpoints in `configs/models/` are licensed and Spark-compatible
- [ ] `hg validate` each experiment config and read every warning
- [ ] Run `make phase0` and `scripts/run_discrimination_control.py`; both must pass
- [ ] Match standalone accuracy across families, or plan the hierarchical adjustment (§18)
- [ ] Re-read `prereg/m2_llm_v1.md` and record any deviation *before* looking at the affected result
