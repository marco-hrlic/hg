# HG-Native M0-R1 — Replication + Matched Vanilla Baseline

**Purpose:** Claude implementation/run handoff  
**Status:** next experiment after the first HG-Native M0 pilot  
**Base spec:** `plans/HG_Native_M0_Technical_Spec_v0_1.md`  
**Pilot report:** `HG_Native_M0_Pilot_Results_Report.md`

## 0. Objective

The first pilot established that HG-native can learn the intended structure. The unresolved confirmatory question is:

> **Does HG training produce a measurably more decision-relevant and useful internal representation than matched ordinary next-token pretraining, or would a vanilla transformer learn essentially the same structure anyway?**

This experiment is confirmatory. Do not add new task families, change the architecture, scale the model, or fix unrelated M0 limitations before this comparison is complete.

Primary scientific question:

\[
\boxed{\text{Does HG learn something vanilla does not learn as efficiently or accessibly?}}
\]

## 1. Freeze the pilot configuration

Use the exact pilot architecture/training setup.

```yaml
vocab_size: 16000
d_model: 640
n_layers: 10
n_heads: 10
n_kv_heads: 5
d_ff: 1792
max_seq_len: 256

baseline_params: 56947840
hg_params: 57204862
hg_overhead: 0.45%

steps: 2000
batch_size: 16
ordinary_lm_ratio: 0.80
ordinary_lm_source: TinyStories
tokenizer: trained byte-level BPE, 16000 tokens
device: CUDA / DGX Spark
```

Preserve optimizer, LR schedule, warm-up, HG loss weights, tokenizer, generators, splits, serialized prompt format, growing-context handling for Family C, and deterministic seeding. Prefer loading the exact seed-0 config from the pilot artifact instead of reconstructing it manually.

## 2. Runs required

Seed 0 Baseline and HG already exist.

Run four new jobs:

```text
Baseline seed 1
Baseline seed 2
HG seed 1
HG seed 2
```

Combined confirmatory set:

```text
Baseline seeds: 0, 1, 2
HG seeds:       0, 1, 2
```

Do not tune hyperparameters based on seed 0. If a bug is found that invalidates seed 0, stop and document it before replacing anything.

## 3. Primary hypothesis

Define Family-C quotient selectivity

\[
Q_{\mathrm{sel}}
=
\overline{\mathrm{ProbeAcc}}_{\mathrm{relevant}}
-
\overline{\mathrm{ProbeAcc}}_{\mathrm{irrelevant}}.
\]

Relevant variables:

```text
r = current relevant
f = future relevant
h = reusable hidden
```

Irrelevant variables:

```text
u = persistent irrelevant
n = random irrelevant
```

Primary hypothesis:

\[
\boxed{Q_{\mathrm{sel}}^{HG} > Q_{\mathrm{sel}}^{LM}}
\]

under matched scale, data, compute, and evaluation episodes.

The pilot HG value was approximately `Q_sel = 0.431`; treat this only as context, not a tuning target.

## 4. Vanilla representation controls

This is the most important new evaluation.

### 4.1 Raw vanilla hidden-state probe

At the exact decision-token position used to construct the HG latent, extract

\[
h_t^{LM}\in\mathbb R^{640}.
\]

Freeze the model. Fit the same linear probes for `r,f,h,u,n` using the same probe train/test split as HG.

Report per-variable accuracy and:

\[
Q_{\mathrm{sel}}^{LM,raw}.
\]

### 4.2 Dimension-matched vanilla control

The HG latent is 64-dimensional, so add a fair 64D vanilla control.

Preferred primary control: PCA-64 fitted on the representation-training split only:

\[
h_t^{LM}\rightarrow z_t^{PCA}\in\mathbb R^{64}.
\]

Then fit the same linear probes.

Optional sensitivity control: fixed seeded random/orthogonal projection to 64D.

Report:

\[
Q_{\mathrm{sel}}^{LM,raw},
\quad
Q_{\mathrm{sel}}^{LM,64},
\quad
Q_{\mathrm{sel}}^{HG}.
\]

Do not fit PCA or any learned projection on the evaluation split.

## 5. Probe protocol

For every seed:

1. load identical held-out Family-C episodes for Baseline and HG;
2. extract representations at the same decision position;
3. split probe-fit/probe-test data deterministically;
4. fit linear probes on fit split only;
5. evaluate on held-out probe test split;
6. save item-level outputs.

Required item-level fields:

```json
{
  "episode_id": "...",
  "seed": 0,
  "model_type": "baseline|hg",
  "representation": "raw|pca64|hg_latent",
  "variable": "r|f|h|u|n",
  "target": 0,
  "prediction": 0,
  "probability": 0.83
}
```

Primary analysis uses linear probes only.

## 6. Primary statistics

For each seed compute:

\[
Q_{\mathrm{sel}}^{HG},
\quad
Q_{\mathrm{sel}}^{LM,raw},
\quad
Q_{\mathrm{sel}}^{LM,64}.
\]

Then:

\[
\Delta Q_{\mathrm{sel}}^{raw}
=
Q_{\mathrm{sel}}^{HG}
-
Q_{\mathrm{sel}}^{LM,raw},
\]

\[
\boxed{
\Delta Q_{\mathrm{sel}}^{64}
=
Q_{\mathrm{sel}}^{HG}
-
Q_{\mathrm{sel}}^{LM,64}
}
\]

with the 64D comparison primary.

Report per-seed values, mean, seed-level standard deviation, and paired episode-level bootstrap 95% CI where appropriate. Do not pool all episodes across seeds and treat item count as if it captured training-run uncertainty.

## 7. Pre-registered N4 interpretation

### Outcome A — HG strongly more selective

Example pattern:

```text
HG Q_sel        ~ +0.4
Vanilla Q_sel   ~ 0.0 to +0.1
```

Interpretation: explicit HG auxiliary training materially reorganizes representation toward decision- and transition-relevant state.

### Outcome B — vanilla is rich, HG is more selective

Example:

```text
HG:      r/f/h high; u/n near chance
Vanilla: r/f/h high; u/n also decodable
```

Interpretation: vanilla learns a rich world state, while HG learns a cleaner task-relative decision state:

\[
H_t^{world}\rightarrow S_t^{decision}.
\]

This counts as a successful result.

### Outcome C — vanilla equally selective

If vanilla and HG have comparable `Q_sel`, do not claim N4 as a distinctive HG effect. Shift emphasis to grounding/accessibility/routing and run ablations before scaling.

## 8. N1/N3 matched vanilla controls

The HG model has supervised information-deficit and accessibility heads; vanilla needs a fair proxy baseline.

Do not alter vanilla training.

### 8.1 Confidence feature set

At decision time extract a small feature vector such as:

```text
max action probability
top1-top2 margin
action entropy
chosen-action probability
LM token entropy at decision position
```

If stochastic evaluation is already available, separately test:

```text
4-sample agreement
sample entropy
```

Do not use target labels as features.

### 8.2 Information-deficit proxy

Fit a simple logistic model on the designated fit split:

\[
g_{\mathrm{info}}(x)\rightarrow P(G^{info}=1).
\]

Evaluate on held-out episodes:

```text
AUROC
AUPRC
Brier
ECE
mean positive score
mean negative score
```

Compare with the HG information-deficit head.

Primary difference:

\[
\Delta AUROC_{\mathrm{info}}
=
AUROC_{\mathrm{HG}}
-
AUROC_{\mathrm{LM-conf}}.
\]

### 8.3 Accessibility proxy

Restrict to `G_info = 0` and fit:

\[
g_{\mathrm{access}}(x)\rightarrow P(G^{access}=1).
\]

Evaluate AUROC/AUPRC/Brier/ECE and compare with the HG accessibility head.

## 9. Vanilla routing baseline

Build a confidence-based ACT/THINK/ASK router from the vanilla proxy outputs.

Use the exact same route cost as HG:

\[
C(r)=L(r)+\lambda_B B(r)+\lambda_K K(r).
\]

The vanilla router may use only vanilla-model/proxy quantities at test time. No generator ground truth is permitted.

## 10. Routing price grid

Evaluate:

```text
lambda_B ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}
lambda_K ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10}
```

Use all 64 Cartesian cells if cheap.

For each cell report:

```text
HG expected cost
vanilla confidence-router expected cost
always ACT
always THINK
always ASK
oracle
```

Primary quantity:

\[
\Delta C(\lambda_B,\lambda_K)
=
C_{\mathrm{HG}}-C_{\mathrm{LM-conf}}.
\]

Report fraction of cells HG wins, contiguous winning regions, mean regret to oracle, worst-case regret, and route confusion by structural state.

Do not claim routing success from one isolated price point.

## 11. Structural route evaluation

Use:

```text
S0: information sufficient + accessible    -> ACT
S1: information sufficient + inaccessible  -> THINK
S2: information insufficient               -> ASK
```

For both systems report:

\[
P(ACT|S0),\quad P(THINK|S1),\quad P(ASK|S2).
\]

Also report:

\[
P(ASK|G^{info}=0)
\]

and

\[
P(ACT|G^{info}>0).
\]

## 12. Replication statistics

Required seed-wise metrics:

```text
seed 0
seed 1
seed 2
mean
std
```

Required primary comparisons:

```text
ΔQ_sel_64
ΔQ_sel_raw
ΔAUROC_info
ΔAUROC_access
Δrouting_cost over price grid
Δordinary_LM_NLL
```

Use paired bootstrap by episode wherever both systems evaluate the same item. Use a fixed bootstrap seed and preserve enough outputs to recompute the analysis.

## 13. Capability floor

For each model/seed report:

```text
n
accuracy
balanced accuracy
per-class accuracy
constant-label collapse
```

Do not interpret routing if the relevant model fails N0.

## 14. Ordinary-LM cost

Repeat the matched language-model comparison for all seeds:

\[
\Delta NLL=NLL_{HG}-NLL_{baseline}.
\]

Report it per seed and aggregated. If HG's specialized structure has a language-model cost, treat it as a real tradeoff.

## 15. Preserve current M0 query semantics

Do not fix the multi-bit/single-hop ASK limitation in R1.

The existing `eta_Q > 1` artifact is known and results from single-hop ASK pricing on multi-bit deficits.

For R1:

- keep the action space unchanged;
- do not use `eta_Q` as a headline metric;
- do not cap/redefine it;
- defer genuine multi-turn acquisition to a later milestone.

## 16. Preserve current reuse semantics

M0 reuse works because the disclosed hidden fact remains in the growing token trajectory.

Do not add HG-only persistent memory.

Report repeated-ask behavior, but describe it correctly as reuse from retained context, not memory beyond the context window.

## 17. Provenance fix before new runs

The pilot's Spark deploy excluded `.git`, leaving `git_commit = null`.

Fix provenance without altering training:

1. host deploy script records the source git SHA and dirty state;
2. sync them as plain files, e.g. `DEPLOY_GIT_SHA`, `DEPLOY_GIT_DIRTY`;
3. run manifest reads them when `.git` is absent.

Each new run should record:

```text
source_git_sha
source_git_dirty
config_hash
tokenizer_hash
generator/version hash
seed
device
torch version
cuda version
parameter count
```

Do not rerun seed 0 solely for this.

## 18. Live logging

Add lightweight training progress output:

```text
step
total loss
LM loss
HG component losses
learning rate
grad norm
tokens/sec
elapsed
ETA
```

Make interval configurable. This is engineering hygiene and must not change optimization behavior.

## 19. Required tests

Before launching replication verify:

```text
same decision-token position for HG and vanilla probes
probe split isolation
PCA/projection fitted only on train split
same eval episode IDs across systems
confidence features contain no label leakage
vanilla router receives no generator truth
paired bootstrap preserves episode matching
manifest captures deployed SHA
seed changes training RNG as intended
held-out paired eval set remains matched
```

Run the complete suite on Spark.

## 20. Suggested artifact layout

```text
results/m0_r1/
  seed0/
    baseline/
    hg/
  seed1/
    baseline/
    hg/
  seed2/
    baseline/
    hg/
  probes/
  confidence_controls/
  routing/
  bootstrap/
  summary/
```

Use repository conventions if already established.

## 21. Required final report

Produce:

```text
HG_Native_M0_R1_Replication_Baseline_Report.md
```

It must contain:

- provenance and exact configs;
- training health / NLL / capability floor;
- N4 primary comparison with per-variable probes;
- N1 information-deficit HG-vs-vanilla comparison;
- N3 accessibility HG-vs-vanilla comparison;
- N2 routing price-grid comparison;
- N5 replicated reuse result with context-only caveat;
- paired bootstrap CIs and seed variability;
- explicit choice among Outcome A/B/C;
- scale/no-scale recommendation.

Required N4 summary table:

| seed | HG Q_sel | Vanilla raw Q_sel | Vanilla 64D Q_sel | Δ HG-vs-64D |
|---|---:|---:|---:|---:|

Do not soften a negative result.

## 22. Confirmatory success criteria

Strong M0-R1 success requires:

1. healthy N0 across seeds;
2. replicated positive `ΔQ_sel_64`;
3. effect not explained by dimension/parameter count;
4. HG info/accessibility discrimination beats reasonable vanilla confidence controls;
5. HG routing has lower expected cost over a nontrivial region of the price grid;
6. ordinary LM degradation remains small and quantified.

Partial success is possible if vanilla has comparable quotient structure but HG clearly wins on information/accessibility separation and routing.

A negative result is valid if matched vanilla reproduces HG behavior.

## 23. Scale decision

### GO to ~150M
Only if the replicated experiment supports at least two of:

```text
grounding-awareness advantage
accessibility-separation advantage
dynamic quotient / reuse advantage
```

with matched vanilla controls and no unexplained LM degradation.

### HOLD
If effects appear real but their causal source is unclear, run A0-A4 ablations first.

### REDESIGN
If matched vanilla controls reproduce HG behavior with no meaningful disadvantage, do not scale the current architecture simply because absolute HG metrics are high.

## 24. Scientific discipline

Do not optimize R1 to reproduce the pilot's near-perfect HG metrics.

The question is not:

> Can HG get AUROC 1.0 again?

It is:

> **At matched scale, data and compute, what internal structure or behavior is specifically attributable to HG-native training?**

## 25. Claude kickoff instruction

Please:

1. inspect the existing HG-native implementation, original M0 spec, and pilot artifacts;
2. identify the smallest code changes needed for M0-R1;
3. implement vanilla raw/64D probe controls, confidence proxies, matched router, paired bootstrap, provenance fix, and live logging;
4. add tests;
5. run the complete suite on DGX Spark;
6. run seed-1 and seed-2 Baseline/HG jobs with the frozen pilot config;
7. evaluate all three seeds, reusing seed 0;
8. produce the self-contained report above;
9. do not scale, alter task families, change model architecture, or repair the multi-turn ASK limitation unless an experiment-invalidating issue is discovered;
10. if such an issue is found, stop and document it before modifying the protocol.

The final decision must answer:

\[
\boxed{\text{Does HG learn a decision-relevant representation and routing policy that matched vanilla pretraining does not?}}
\]
