# HG-Native M0 — Detailed Technical Specification

**Version:** 0.1  
**Date:** 2026-09-01  
**Purpose:** implementation specification for the first from-scratch HG-native scaling experiment on DGX Spark.

---

## 1. Objective

Build and compare two matched ~50M-parameter decoder-only transformers:

1. **Baseline-50M** — ordinary causal language model.
2. **HG-50M** — same transformer trunk, tokenizer, data budget, optimizer, schedule, and nominal parameter budget, augmented with explicit HG representation and auxiliary objectives.

The experiment is not intended to produce a competitive general-purpose LM. It is intended to answer five architectural questions:

1. Can an HG-native model learn to distinguish **missing decision-relevant information** from ordinary predictive uncertainty?
2. Can it distinguish **information deficit** from **computational/accessibility deficit**?
3. Can a compact latent state recover the known **decision-relevant quotient** of a controlled environment?
4. Can it route among ACT / THINK / ASK better than a matched vanilla LM?
5. Can information grounded once be **reused across time** rather than reacquired?

The experiment must preserve falsifiability. HG is considered unsuccessful at M0 if it does not materially improve the HG-specific metrics under matched compute and data.

---

## 2. Design principles

### 2.1 Matched comparison

Baseline and HG models must use the same tokenizer, transformer trunk architecture, context length, training token count, optimizer, learning-rate schedule, batch construction, random-data generator families, train/validation/test splits, and evaluation budget.

The HG model may have additional heads, but the total parameter difference should remain small enough that the comparison is not explained by brute-force parameter count. Target:

\[
\frac{N_{\mathrm{HG}}-N_{\mathrm{baseline}}}{N_{\mathrm{baseline}}}<0.05.
\]

If needed, slightly narrow the HG trunk so that total parameter counts are matched within 1–2%.

### 2.2 Structured first, language later

M0 uses compact synthetic/semi-symbolic text and structured actions. Do not require free-form natural-language query generation yet.

Primary actions:

- `ACT_A`
- `ACT_B`
- `THINK`
- `ASK_k` for hidden variable \(k\)

The point of M0 is to test the representation and routing mechanism independently of language-generation quality.

### 2.3 Explicit separation of deficits

The model must expose two separate predicted quantities:

\[
\hat R_t^{\mathrm{info}}
\]

for residual decision-relevant external information deficit, and

\[
\hat R_t^{\mathrm{access}}
\]

for computational/accessibility deficit.

Do not collapse these into a single confidence score.

---

# 3. Model architecture

## 3.1 Transformer trunk

Target size: **~50–70M total parameters**.

Recommended first configuration:

```yaml
model:
  vocab_size: 16000
  n_layers: 12
  d_model: 768
  n_heads: 12
  n_kv_heads: 4
  d_ff: 2048
  max_seq_len: 1024
  norm: rmsnorm
  activation: swiglu
  positional_encoding: rope
  tie_embeddings: true
  dropout: 0.0
```

This should be treated as a starting configuration, not a sacred architecture. Before long runs, compute exact parameter counts programmatically.

The baseline has:

\[
h_t=F_\theta(x_{\le t}),
\]

followed by the usual LM head.

The HG model uses the same trunk and LM head plus a compact HG state.

## 3.2 Decision latent

At selected decision positions:

\[
S_t=f_{\mathrm{HG}}(h_t).
\]

Initial design:

```yaml
hg_latent:
  input_dim: d_model
  hidden_dim: 256
  latent_dim: 64
  layers: 2
  activation: silu
  normalization: layernorm
```

Suggested mapping:

\[
h_t\rightarrow \mathrm{MLP}\rightarrow S_t\in\mathbb R^{64}.
\]

Do not force information-bottleneck regularization in the first run. First establish whether the latent contains the correct variables. Compression regularization is an ablation, not a default.

## 3.3 HG heads

All heads consume \(S_t\) unless explicitly noted.

### Decision head

Predict final task decision:

\[
p_\theta(J_t\mid S_t).
\]

For M0 binary tasks: `A / B`.

Loss:

\[
\mathcal L_{\mathrm{dec}}=-\log p_\theta(J_t\mid S_t).
\]

### Distortion-profile head

Predict the loss associated with each available action:

\[
\hat d_t(a)=g_d(S_t,a).
\]

For discrete M0 actions, output a vector

\[
\hat{\mathbf d}_t=(\hat d_t(a_1),\ldots,\hat d_t(a_m)).
\]

Loss:

\[
\mathcal L_{\mathrm{quot}}=\frac1m\sum_{a\in\mathcal A}\ell(\hat d_t(a),d_t(a)).
\]

### Information-deficit head

Predict whether external information is genuinely required:

\[
p_\theta(G_t^{\mathrm{info}}=1\mid S_t).
\]

Training target is generator-known and must be structural, not derived from the LM's own success/failure.

For exact binary tasks:

\[
G_t^{\mathrm{info}}=\mathbf 1[H(J_t\mid M_t)>0].
\]

For tasks with known minimum external message cost \(B_t^*\), optionally predict both binary need and scalar \(B_t^*\).

### Accessibility head

Predict whether the decision is already determined by the AI-visible information but the current model is likely to fail without extra computation/scaffolding.

Target:

\[
G_t^{\mathrm{access}}=\mathbf 1[H(J_t\mid M_t)=0\land A_t^{\mathrm{base}}\ne J_t].
\]

For training stability, use counterfactual execution statistics rather than one stochastic sample when possible.

A continuous target is

\[
V_t^{\mathrm{scaf}}=L_t(M_t)-L_t(M_t+\mathrm{scaffold}),
\]

restricted to items with zero information deficit.

### Transition head

For sequential tasks:

\[
\hat S_{t+1}=g_T(S_t,A_t).
\]

First implementation should use a stop-gradient target:

\[
\mathcal L_{\mathrm{trans}}=\|g_T(S_t,A_t)-\operatorname{sg}(S_{t+1})\|_2^2.
\]

If latent collapse or instability appears, replace with a contrastive or VICReg-style predictive objective.

### Router head

Input:

\[
[S_t,\hat R_t^{\mathrm{info}},\hat R_t^{\mathrm{access}},c_t]
\]

where \(c_t\) may contain explicitly allowed cost variables.

Output distribution over

\[
\{\mathrm{ACT},\mathrm{THINK},\mathrm{ASK}\}
\]

for M0.

`RETRIEVE` and `DELIBERATE` are postponed.

The router should be trained from counterfactual route values, not from model confidence alone.

Define route cost:

\[
C(r)=L(r)+\lambda_B B(r)+\lambda_K K(r),
\]

where \(L(r)\) is decision loss, \(B(r)\) external information cost, and \(K(r)\) internal compute cost.

Target route:

\[
r^*=\arg\min_r C(r).
\]

Train router by CE against \(r^*\), while separately reporting continuous regret.

---

# 4. Data model

Each training/evaluation episode must expose explicit ground truth.

Recommended canonical schema:

```json
{
  "episode_id": "...",
  "family": "...",
  "visible_state": {},
  "hidden_state": {},
  "full_state": {},
  "available_actions": [],
  "correct_action": "...",
  "distortion_profile": {},
  "needs_grounding": true,
  "min_grounding_bits": 1.0,
  "minimal_query_set": ["h3"],
  "scaffold_variants": [],
  "sequential_state_id": "...",
  "true_quotient_state": "...",
  "metadata": {}
}
```

The saved episode artifact must allow every reported HG metric to be recomputed without rerunning model generation.

---

# 5. M0 task families

## 5.1 Family A — Static Hidden Variable

Purpose: grounding discrimination, structured minimal queries, and cost-sensitive routing.

Generate visible variables \(X=(x_1,\ldots,x_p)\) and hidden variables \(H=(h_1,\ldots,h_q)\). Decision:

\[
J=j(X,H).
\]

Required subclasses:

1. no deficit, \(H(J\mid X)=0\);
2. one-bit deficit;
3. multi-bit deficit;
4. redundant hidden information;
5. irrelevant hidden variables;
6. distractor-visible variables.

The generator must compute \(B^*\) exactly where possible.

Example serialized input:

```text
TASK STATIC-CHOICE
A.cost = 37
B.cost = 29
B.requires = h3
RULE:
choose B iff B is cheaper AND h3 = YES;
otherwise choose A.
KNOWN:
A.cost = 37
B.cost = 29
h3 = UNKNOWN
DECIDE OR REQUEST INFORMATION.
```

Action target: `ASK_h3`.

Contrast:

```text
A.cost = 29
B.cost = 37
B.requires = h3
```

Here the correct action is `ACT_A` and `h3` is irrelevant.

## 5.2 Family B — Accessibility / Scaffolding

Purpose: separate new information from easier representation of existing information.

Start from information-closed problems:

\[
H(J\mid M)=0.
\]

Produce variants:

1. canonical compact form;
2. reordered form;
3. explicit derived comparison;
4. irrelevant constraints block;
5. redundant restatement;
6. chain-decomposed form;
7. minimal explanatory scaffold;
8. misleading but non-informative distractor scaffold.

All variants must preserve the underlying information set.

Primary metric:

\[
S_C=\mathbb E[L(M)-L(M+\text{scaffold})\mid H(J\mid M)=0].
\]

The HG model should predict the accessibility problem, prefer THINK over ASK when information deficit is zero, and show lower unnecessary ASK rate.

For M0, there is no separate external `SCAFFOLD` action. Instead `THINK` exposes a deterministic normalized representation of the same information. This gives a clean zero-bit compute route.

## 5.3 Family C — Sequential Known Quotient

Purpose: test dynamic decision-state learning and grounding reuse.

Construct a controlled Markov process:

\[
Z_{t+1}\sim P(Z_{t+1}\mid Z_t,A_t).
\]

Full state contains `relevant_now`, `relevant_future`, `irrelevant_persistent`, `irrelevant_noise`, and `hidden_reusable` components.

The generator defines the exact quotient:

\[
S_t^*=\phi(Z_t).
\]

Example:

\[
Z_t=(r_t,f_t,u_t,n_t,h),
\]

where \(r_t\) affects immediate loss, \(f_t\) affects action-conditioned future relevant state, \(u_t\) is persistent but irrelevant, \(n_t\) is noise, and \(h\) is a hidden fact needed repeatedly.

The true quotient may be

\[
S_t^*=(r_t,f_t,h).
\]

Evaluation tests whether learned \(S_t\) predicts \(r_t\), predicts \(f_t\), retains grounded \(h\), and does not require \(u_t,n_t\) for optimal decisions.

At \(t=1\), the model may ask for \(h\). If \(h\) remains valid and is retained, subsequent optimal grounding cost should be zero. Measure repeated-ask rate after successful grounding.

## 5.4 Family D — Capability Refinement

Purpose: test whether the learned decision representation changes when the action frontier changes.

Construct states \(z,z'\) such that

\[
z\sim_{C_1}z'
\]

under action set \(\mathcal A_1\), but

\[
z\not\sim_{C_2}z'
\]

when a new action \(a^*\) is introduced.

Train/evaluate under explicit capability tokens `CAPABILITY = C1` or `CAPABILITY = C2`.

Measure representational separability of \(z,z'\) conditioned on capability. Prediction: lower separability under \(C_1\), higher under \(C_2\).

This is exploratory and not required for M0 success.

---

# 6. Training corpus

Suggested initial mixture:

```yaml
data_mix:
  ordinary_lm: 0.80
  static_hidden: 0.08
  accessibility: 0.04
  sequential_quotient: 0.06
  capability_refinement: 0.02
```

This mixture must be ablated later.

Ordinary LM data can initially be a modest open text corpus or synthetic language corpus. M0 does not require encyclopedic knowledge; the purpose is to prevent the model from becoming a pure symbolic classifier.

Do not use evaluation templates verbatim in training. Hold out random seeds, lexical templates, rule compositions, state-transition matrices, and task parameter ranges.

---

# 7. Training objectives

Baseline:

\[
\boxed{\mathcal L_{\mathrm{base}}=\mathcal L_{\mathrm{LM}}.}
\]

HG:

\[
\boxed{
\mathcal L_{\mathrm{HG}}
=
\mathcal L_{\mathrm{LM}}
+\lambda_d\mathcal L_{\mathrm{dec}}
+\lambda_q\mathcal L_{\mathrm{quot}}
+\lambda_t\mathcal L_{\mathrm{trans}}
+\lambda_i\mathcal L_{\mathrm{info}}
+\lambda_a\mathcal L_{\mathrm{access}}
+\lambda_r\mathcal L_{\mathrm{route}}.
}
\]

Initial conservative weights:

```yaml
loss_weights:
  lm: 1.0
  decision: 0.25
  quotient: 0.10
  transition: 0.05
  info_deficit: 0.10
  accessibility: 0.10
  route: 0.10
```

Do not assume these are optimal.

Use warm-up scheduling:

```yaml
hg_loss_schedule:
  0_to_10pct:
    hg_multiplier: 0.0
  10_to_25pct:
    hg_multiplier: linear_0_to_1
  25_to_100pct:
    hg_multiplier: 1.0
```

---

# 8. Training schedule

## 8.1 Smoke stage

Train both models for a tiny token budget sufficient only to test loss decrease, finite gradients, head gradients, checkpointing, and evaluator roundtrip.

## 8.2 Pilot stage

Train Baseline-50M seed 0 and HG-50M seed 0. If HG fails the capability floor or shows pathological LM degradation, stop and debug.

## 8.3 Replication stage

Run at least Baseline seeds 0,1,2 and HG seeds 0,1,2. Only replicated effects count toward gates.

## 8.4 Ablation stage

Run:

- A0 LM only;
- A1 LM + decision;
- A2 LM + decision + quotient;
- A3 A2 + transition;
- A4 full HG.

Prefer one seed for broad screening, then 3 seeds for promising variants.

---

# 9. Optimization defaults

```yaml
optimizer:
  name: adamw
  beta1: 0.9
  beta2: 0.95
  eps: 1e-8
  weight_decay: 0.1

schedule:
  warmup_ratio: 0.02
  type: cosine
  min_lr_ratio: 0.1

precision:
  train: bf16

grad:
  clip_norm: 1.0

checkpointing:
  activation_checkpointing: true
```

Learning-rate candidates:

```text
1e-4
2e-4
3e-4
5e-4
```

Do not tune HG and baseline separately unless the tuning budget is symmetric.

---

# 10. Core metrics

## 10.1 General LM metrics

- validation NLL;
- perplexity;
- token accuracy where useful.

Report

\[
\Delta\mathrm{NLL}=\mathrm{NLL}_{HG}-\mathrm{NLL}_{base}.
\]

HG must not catastrophically damage ordinary LM performance.

## 10.2 Decision metrics

- final accuracy;
- balanced accuracy;
- per-class accuracy;
- expected distortion;
- oracle-grounded accuracy;
- ungrounded accuracy.

Capability floor:

\[
D_C^{\mathrm{blind}},\qquad D_C^{\mathrm{full}}.
\]

When ideal oracle distortion is known:

\[
U_C=\frac{D_C^{\mathrm{blind}}-D_C^{\mathrm{full}}}{D_C^{\mathrm{blind}}-D^{\mathrm{oracle}}}.
\]

## 10.3 Grounding discrimination

Report AUROC, AUPRC, Brier, ECE, discrimination by task subclass, and mean score with/without true deficit.

Do not treat calibration alone as success.

## 10.4 Query efficiency

Report system bits \(B_{\mathrm{system}}\), oracle bits \(B^*\), soundness, minimality, unnecessary ask rate, under-asking rate, and bits per resolved decision.

\[
\eta_Q=\frac{B_{\mathrm{generic}}-B_{\mathrm{system}}}{B_{\mathrm{generic}}-B^*}.
\]

## 10.5 Accessibility / scaffolding

Measure

\[
S_C=\mathbb E[L(M)-L(M+\mathrm{scaffold})\mid G^{\mathrm{info}}=0].
\]

Also report ACT, THINK, and ASK rates on these items.

Desired qualitative behavior:

```text
information missing                  -> ASK
information present but inaccessible -> THINK
information present and accessible   -> ACT
```

## 10.6 Quotient recovery

Use linear probes on \(S_t\) for current relevant, future relevant, grounded reusable, persistent irrelevant, and random irrelevant variables.

Optional selectivity score:

\[
Q_{\mathrm{sel}}=\overline{\mathrm{ProbeAcc}}_{\mathrm{relevant}}-\overline{\mathrm{ProbeAcc}}_{\mathrm{irrelevant}}.
\]

## 10.7 Grounding reuse

After acquiring reusable hidden variable \(h\):

\[
R_{\mathrm{repeat}}=P(\mathrm{ASK}(h)\text{ again}\mid h\text{ already grounded and still valid}).
\]

Also report future external bits

\[
B_{\mathrm{future}}=\sum_{t>t_0}B_t.
\]

---

# 11. M0 gates

## N0 — capability floor

The fully grounded model must materially exceed chance and avoid constant-label collapse. Failure means routing metrics are uninterpretable.

## N1 — information-deficit discrimination

HG must outperform the matched baseline on held-out grounding-need discrimination. Require a replicated AUROC improvement with bootstrap 95% CI for the difference excluding 0, without catastrophic calibration degradation.

## N2 — routing value

Across a preregistered grid of information and compute prices, HG routing must improve expected

\[
C=L+\lambda_BB+\lambda_KK
\]

over always ACT, always ASK, always THINK, and the matched baseline router. Require improvement at more than one isolated price.

## N3 — accessibility separation

On zero-information-deficit items, HG's accessibility score must discriminate actual computation failures; HG should route more failures to THINK than ASK; unnecessary ASK rate must be lower than baseline.

## N4 — quotient recovery

HG latent must recover relevant and future-relevant quotient variables more strongly than irrelevant variables and outperform the baseline's matched representation on quotient selectivity.

## N5 — grounding reuse

HG must materially reduce repeated acquisition of already-known reusable hidden information compared with baseline.

---

# 12. Falsification conditions

### F0
HG improves router metrics only because it has more parameters.

Control: parameter-match within 1–2% or add an equal-capacity baseline auxiliary MLP.

### F1
HG grounding head simply detects surface task templates.

Control: held-out lexical templates and unseen generator parameterizations.

### F2
HG predicts grounding need but final routing has no economic value.

Control: explicit cost curves over \(\lambda_B,\lambda_K\).

### F3
HG latent contains all state variables equally.

Interpretation: no evidence of decision quotient compression.

### F4
HG improves HG metrics but substantially worsens language modeling.

Interpretation: architecture may be a specialist auxiliary system rather than a viable native LM objective.

### F5
Accessibility head merely predicts model confidence.

Control: compare to entropy/logit-margin/self-consistency baselines.

### F6
Reuse improvement comes from simply copying all hidden state into unrestricted memory.

Control: fixed memory budget and matched baseline memory interface.

### F7
Capability-refinement representation split occurs even when action frontier is unchanged.

Interpretation: representation probe may be responding to superficial capability tokens.

---

# 13. Baseline controls

Minimum control set:

1. Vanilla LM trunk.
2. Vanilla + equal-capacity MLP.
3. Vanilla + confidence router.
4. Vanilla + self-consistency router.
5. HG native.

Where compute permits, include an oracle structural router to estimate the recoverable ceiling.

---

# 14. Representation analysis

Save latent vectors for a fixed evaluation subset.

Analyses:

- PCA/UMAP only for visualization, never as primary evidence;
- linear probe for each true state component;
- CKA similarity between baseline and HG trunk layers;
- probe across layers to determine where decision relevance emerges;
- latent intervention tests where feasible;
- quotient-class within/between distance statistics.

Important descriptive experiment:

For states \(z,z'\) with the same true quotient,

\[
\phi(z)=\phi(z'),
\]

measure whether

\[
\|S(z)-S(z')\|
\]

is smaller than for states in different quotient classes.

---

# 15. Repository structure

Suggested additive structure:

```text
src/hg_native/
    model/
        config.py
        transformer.py
        hg_latent.py
        heads.py
        router.py

    data/
        schema.py
        static_hidden.py
        accessibility.py
        sequential_quotient.py
        capability_refinement.py
        lm_mix.py

    training/
        losses.py
        schedules.py
        trainer.py
        checkpoint.py

    eval/
        capability.py
        grounding.py
        routing.py
        query_efficiency.py
        accessibility.py
        quotient_probes.py
        reuse.py

experiments/hg_native/
    configs/
        m0_baseline_50m.yaml
        m0_hg_50m.yaml
        m0_ablation_decision.yaml
        m0_ablation_quotient.yaml
        m0_ablation_transition.yaml
    run_train.py
    run_eval.py
    analyze.py

tests/hg_native/
    test_generators.py
    test_ground_truth.py
    test_model_shapes.py
    test_losses.py
    test_router_counterfactuals.py
    test_checkpoint_roundtrip.py
    test_metric_recompute.py
```

If integration with the existing HG repo is cleaner under existing namespaces, preserve existing conventions rather than forcing this exact path.

---

# 16. Experiment artifacts

Every run directory must contain:

```text
manifest.json
config.yaml
git.json
tokenizer_hash.json
data_manifest.json
train_metrics.jsonl
eval/
    episodes.jsonl
    predictions.jsonl
    counterfactuals.jsonl
    metrics.json
    calibration.json
    query_metrics.json
    quotient_probes.json
    reuse.json
checkpoints/
report.md
```

`manifest.json` must contain git commit, dirty-tree status, hostname/device, CUDA/PyTorch versions, RNG seeds, config hash, tokenizer hash, data-generator version, and model parameter count.

No Python `hash()` may be used for stable cross-process randomization. Use an explicit cryptographic digest or seeded PRNG.

---

# 17. Evaluation protocol

Use strict split isolation.

For every task family hold out at least:

- 20% generator seeds;
- several complete textual templates;
- at least one parameter range;
- unseen combinations of causal rules.

Report:

1. in-distribution;
2. lexical/template OOD;
3. compositional OOD;
4. parameter OOD.

HG claims should rely primarily on held-out structure, not training-template performance.

---

# 18. Statistical protocol

Use episode-level bootstrap confidence intervals.

For model comparisons:

\[
\Delta M=M_{\mathrm{HG}}-M_{\mathrm{base}}.
\]

Report point estimate, 95% bootstrap CI, per-seed values, and pooled value only as secondary.

For multiple routing price points, do not declare success from one isolated crossing.

For theorem-regression synthetic tasks, sample size must be selected from the expected failure probability rather than using one fixed item count everywhere.

---

# 19. Run order

1. Implement model factory, tokenizer, generators, artifact schema, evaluator, deterministic tests.
2. Mechanically verify generator ground truth: correct action, distortion profile, `needs_grounding`, \(B^*\), minimal query set, quotient state, transition.
3. Train a 5–10M parameter micro-model for debugging only.
4. Run 50M smoke baseline and HG.
5. Run one full pilot seed each.
6. Inspect failure modes before replication.
7. Run 3-seed replication.
8. Screen ablations A0–A4.
9. Freeze an M0 report with gate/falsification outcomes and a scale/no-scale recommendation.

Do not move to 150M automatically.

---

# 20. Spark-specific strategy

The DGX Spark should be the default development platform.

Priorities:

- BF16;
- activation checkpointing;
- FlashAttention/SDPA;
- compile only after correctness;
- batch evaluation;
- generate synthetic episodes on CPU in parallel;
- cache tokenized ordinary-LM shards;
- avoid excessive checkpoint frequency.

Profile tokens/sec during the 5–10M micro-model stage and use measured throughput to choose the full M0 token budget.

Do not choose the token budget from generic scaling laws alone. M0 is an architecture experiment, so iteration speed matters.

---

# 21. Token-budget policy

Use three budgets:

```text
micro: enough for end-to-end debug
pilot: enough to establish obvious architectural signal
full:  selected after measured scaling curves
```

During pilot, save evaluations at logarithmically spaced token counts.

Plot HG-vs-baseline difference versus tokens seen. This tells us whether the HG effect appears early and persists, emerges only late, or disappears as ordinary LM competence improves.

---

# 22. Success criterion for scaling to M1

Proceed to ~150M only if the replicated M0 result shows at least two of the following three central effects:

1. **Grounding awareness** — HG materially improves information-deficit discrimination and routing value.
2. **Accessibility separation** — HG better distinguishes THINK-worthy computation failures from ASK-worthy information deficits.
3. **Dynamic quotient/reuse** — HG better recovers the true quotient and/or reduces repeated grounding in sequential tasks.

And all of the following must also hold:

- N0 passes;
- ordinary-LM degradation is acceptable and understood;
- effects survive held-out templates/structures;
- improvements cannot be explained by extra parameter count.

If only query efficiency improves, do not scale yet; the current Qwen experiment already shows query minimality can be solved cheaply by instruction.

---

# 23. M1 preview — ~150M

M1 should add:

- richer natural-language paraphrases;
- more complex compositional rules;
- first templated natural-language query generation;
- cross-family transfer;
- longer sequential horizons;
- more realistic memory pressure.

Do not change the core architecture and scale simultaneously unless M0 reveals a necessary fix.

---

# 24. Main hypotheses

### H-Native-1
Explicit HG training makes residual decision-information need more simply accessible in the latent representation than ordinary LM training.

### H-Native-2
HG-native training reduces dependence on information-theoretically irrelevant external scaffolding, or at minimum enables the model to identify it as accessibility rather than information gain.

### H-Native-3
A latent trained on decision distortion plus action-conditioned transitions approximates the true dynamic decision quotient better than an ordinary LM hidden state.

### H-Native-4
Grounding information acquired at one time step is reused more efficiently by an HG-native model.

### H-Native-5
These effects survive increases in model scale and linguistic complexity.

M0 primarily addresses H-Native-1 through H-Native-4.

---

# 25. Non-goals of M0

M0 does not attempt to establish AGI-scale grounding laws, superiority over modern 3B/7B instruct models, open-domain factual grounding, real human-in-the-loop usability, free-form deliberative dialogue, long-horizon autonomous agency, constructive preference formation, integrity-safe human reference updates, end-to-end learned retrieval, or a competitive foundation model.

---

# 26. Immediate coding checklist

1. Create `HGNativeConfig`.
2. Implement baseline transformer.
3. Implement HG latent and heads.
4. Implement canonical `Episode` schema.
5. Implement Static Hidden Variable generator.
6. Add exhaustive generator ground-truth tests.
7. Implement Accessibility generator.
8. Implement Sequential Known Quotient generator.
9. Implement Capability Refinement generator.
10. Implement training batch serializer.
11. Implement loss registry and HG loss schedule.
12. Implement structured router counterfactual evaluator.
13. Implement capability-floor evaluator.
14. Implement grounding discrimination metrics.
15. Implement query efficiency metrics.
16. Implement accessibility/scaffolding metrics.
17. Implement quotient probe dataset export.
18. Implement grounding-reuse evaluator.
19. Implement run manifest and stable seeding.
20. Train 5–10M parameter debug model.
21. Verify artifact recomputation.
22. Profile Spark throughput.
23. Finalize M0 token budget.
24. Run Baseline-50M seed 0.
25. Run HG-50M seed 0.
26. Review failure modes before replication.

---

# 27. Decision rule after M0

At the end of M0, do not ask only:

> Did HG get higher task accuracy?

The actual decision is:

\[
\boxed{\text{Did explicitly training decision-relevant information structure create measurable, reusable internal structure that a matched vanilla LM did not learn as efficiently?}}
\]

If yes, scale. If no, revise the architecture while experiments are still cheap.
