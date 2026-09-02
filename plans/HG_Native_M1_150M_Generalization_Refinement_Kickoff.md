# HG-Native M1 — 150M Generalization + Capability-Refinement Experiment

**Purpose:** implementation and experiment handoff for Claude  
**Status:** next milestone after successful M0-R1 replication  
**Basis:** `HG_Native_M0_Technical_Spec_v0_1.md`, `HG_Native_M0_R1_Experiment_Kickoff.md`, and `HG_Native_M0_R1_Replication_Baseline_Report.md`

---

# 0. Why M1 exists

M0-R1 established a replicated matched-baseline result at ~57M parameters:

- HG preserves decision- and transition-relevant variables strongly;
- vanilla pretraining also preserves those variables, but retains irrelevant variables almost perfectly;
- HG therefore produces a more selective task-relative decision state;
- HG beats the matched vanilla confidence router across the entire tested routing-cost grid;
- HG information-deficit and accessibility heads outperform vanilla confidence proxies;
- these effects replicate across three seeds with only a small ordinary-LM cost.

The main representational result is **Outcome B**:

\[
H_t^{world}\rightarrow S_t^{decision}.
\]

The vanilla trunk learns a rich state containing both relevant and irrelevant information. HG auxiliary training produces a more selective decision state that preserves what the task needs and suppresses what it does not.

M1 must test whether this effect is more than a property of the exact M0 synthetic generators.

The central question is now:

\[
\boxed{
\text{Does HG's task-relative decision representation generalize, refine, and transfer?}
}
\]

M1 should not merely repeat M0 at a larger scale.

---

# 1. M1 scientific objectives

M1 has three primary objectives.

## M1-A — Structural and lexical OOD generalization

Test whether HG selectivity survives when the surface form, rule composition, and generator parameter ranges differ from training.

The question is:

> Does HG learn the underlying decision-relevant structure, or does it learn the specific templates used to supervise it?

---

## M1-B — Cross-family transfer

Train HG objectives on a subset of task families and evaluate on an unseen family.

The question is:

> Does the HG representation learn a transferable notion of decision relevance, information deficit, and accessibility, or only family-specific classifiers?

---

## M1-C — Capability refinement

Change the available action frontier so that a state variable that was previously irrelevant becomes decision relevant.

The theory predicts that the decision quotient should refine:

\[
\mathcal A_1\rightarrow\mathcal A_2
\]

can induce

\[
z\sim_{C_1}z'
\quad\text{but}\quad
z\not\sim_{C_2}z'.
\]

The key empirical question is:

> Does the learned \(S_t^{decision}\) begin representing the newly relevant variable when the available action set changes?

This is the most theory-specific M1 test.

---

# 2. Keep the two-level architecture

Do not collapse the full model state into the HG bottleneck.

Preserve the conceptual architecture:

\[
H_t^{world}
\rightarrow
S_t^{decision}.
\]

Where:

- \(H_t^{world}\) is the rich transformer hidden state;
- \(S_t^{decision}\) is the compact HG latent used by the decision/grounding/routing machinery.

M0-R1 strongly supports this design: vanilla hidden states retained relevant and irrelevant variables alike, while HG selectively suppressed irrelevant state.

Do not force the trunk itself to discard world information.

---

# 3. Scale target

Target M1 model size:

\[
\boxed{\text{approximately }150\text{M parameters}}
\]

for both matched Baseline and HG variants.

Maintain total parameter matching as closely as practical.

Target:

\[
\frac{N_{HG}-N_{LM}}{N_{LM}}<0.05.
\]

Prefer <2% if easy.

Suggested starting architecture:

```yaml
vocab_size: 16000
n_layers: 16
d_model: 896
n_heads: 14
n_kv_heads: 7
d_ff: 2560
max_seq_len: 512
norm: rmsnorm
activation: swiglu
positional_encoding: rope
tie_embeddings: true
dropout: 0.0
```

Do not assume this lands at exactly 150M.

Use the analytic parameter counter before training and adjust `d_model`, `n_layers`, or `d_ff` to land approximately in the 130–170M range.

Record exact Baseline/HG counts.

---

# 4. HG latent at M1

Keep a compact decision latent.

Initial target:

```yaml
hg_latent_dim: 96
hg_hidden_dim: 384
```

This is a modest increase from M0's 64D latent.

Do not scale the latent proportionally with model size without testing.

Primary comparison should still include:

- raw vanilla hidden state;
- dimension-matched vanilla PCA/projected state;
- HG decision latent.

If 96D is used for HG, use a 96D vanilla control.

---

# 5. Training data philosophy

M1 should remain a mixed LM + HG-pretraining experiment.

Suggested starting mixture:

```yaml
ordinary_lm: 0.85
hg_structured_total: 0.15
```

The exact 85/15 split may be adjusted only before confirmatory runs.

Do not tune it separately for Baseline and HG.

The structured 15% should include:

```text
Static Hidden Variable
Accessibility / Scaffolding
Sequential Known Quotient
Capability Refinement
new OOD-oriented linguistic variants
```

M1 should increase linguistic variability substantially while retaining exact generator-known ground truth.

---

# 6. M1-A — OOD generalization design

Create explicit training and held-out distributions.

## 6.1 Lexical OOD

Hold out entire surface-form families.

Examples:

Training:

```text
Route A costs...
Route B requires...
Choose...
```

Evaluation:

```text
Option red incurs...
Option blue is feasible only if...
Select...
```

The logical structure is identical, but wording and serialization differ.

Do not allow shared sentence templates between train and lexical-OOD test.

---

## 6.2 Structural OOD

Hold out rule compositions.

Example:

Training may contain:

\[
J=(x_1\land h_1)
\]

and

\[
J=(x_2\lor h_2).
\]

Evaluation may contain compositions such as:

\[
J=(x_1\land h_1)\lor(x_2\land \neg h_2).
\]

The component primitives may be seen in training, but the exact composition must be unseen.

---

## 6.3 Parameter OOD

Hold out numerical/logical ranges.

Examples:

```text
training path lengths: 2–6
test path lengths: 7–10

training rule fanout: 1–3
test rule fanout: 4–5
```

Do not make parameter OOD so extreme that N0 fails trivially.

---

## 6.4 Distractor-density OOD

Increase irrelevant visible state beyond training levels.

For example:

```text
training: 2–5 irrelevant variables
OOD test: 10–30 irrelevant variables
```

This is especially important for the decision-quotient hypothesis.

Prediction:

\[
Q_{\mathrm{sel}}^{HG}
\]

should degrade less than vanilla selectivity as distractor density rises.

---

# 7. M1-B — cross-family transfer

Create a leave-one-family-out protocol.

Minimum experiment:

Train HG auxiliary supervision on:

```text
Family A
Family B
Family C
```

while withholding HG-family supervision from:

```text
Family D-transfer
```

The withheld family must still appear in ordinary LM form if needed for basic language competence, but it must not expose the HG targets used for the transfer test.

A stronger version:

```text
Train HG supervision: A, B, C
Test transfer: new Family E
```

Family E should be newly generated for M1.

Suggested Family E:

# Resource Allocation / Scheduling

State includes:

```text
visible jobs
resource capacities
deadlines
hidden preference/cost variable
persistent irrelevant metadata
future-relevant constraint
```

Decision:

```text
choose allocation / schedule A or B
```

The family must expose exact:

\[
J,\quad d(z,a),\quad B^*,\quad G^{info},\quad G^{access},
\]

and where sequential:

\[
S_t^*.
\]

The goal is to test whether HG heads generalize to a new semantic domain rather than simply recognizing route-choice syntax.

---

# 8. M1-C — capability refinement experiment

This is the flagship M1 experiment.

## 8.1 Environment construction

Create state:

\[
Z=(r,f,u,n,v)
\]

where under capability \(C_1\):

- `r` affects current distortion;
- `f` affects future quotient transition;
- `u`,`n` are irrelevant;
- `v` is also irrelevant under the current action set.

Let:

\[
\mathcal A_1=\{a_0,a_1\}.
\]

Construct `v` so:

\[
d_{C_1}(z,a)
\]

and quotient transitions do not depend on `v`.

Thus:

\[
z\sim_{C_1}z'
\]

for states differing only in `v`.

Now introduce:

\[
\mathcal A_2=\mathcal A_1\cup\{a_2\},
\]

where the distortion or transition outcome of \(a_2\) depends on `v`.

Now:

\[
z\not\sim_{C_2}z'.
\]

This is a generator-known quotient refinement.

---

## 8.2 Capability conditioning

Include an explicit capability token/input:

```text
CAPABILITY = C1
```

or

```text
CAPABILITY = C2
```

The same world state may be presented under either capability.

This gives counterfactual pairs:

\[
(z,C_1)
\]

and

\[
(z,C_2).
\]

---

## 8.3 Pre-registered prediction

For the HG latent:

\[
\mathrm{ProbeAcc}(v\mid S,C_1)
\approx \text{chance},
\]

but

\[
\mathrm{ProbeAcc}(v\mid S,C_2)
>
\mathrm{ProbeAcc}(v\mid S,C_1).
\]

Define refinement response:

\[
\Delta_{\mathrm{refine}}
=
\mathrm{ProbeAcc}(v\mid C_2)
-
\mathrm{ProbeAcc}(v\mid C_1).
\]

Primary prediction:

\[
\boxed{
\Delta_{\mathrm{refine}}^{HG}>0.
}
\]

The stronger comparative prediction is:

\[
\boxed{
\Delta_{\mathrm{refine}}^{HG}
>
\Delta_{\mathrm{refine}}^{LM,decision\ control}
}
\]

for dimension-matched decision states.

---

## 8.4 Important control

Ensure the model cannot simply encode `v` because the capability token is correlated with it.

Use identical `v` distributions under \(C_1\) and \(C_2\).

Use paired counterfactual states.

Also include a control variable `u` that remains irrelevant under both capabilities.

Desired pattern:

```text
v under C1 -> suppressed
v under C2 -> represented
u under C1 -> suppressed
u under C2 -> still suppressed
```

That is much stronger evidence than generic probe improvement.

---

# 9. Capability-refinement transfer test

Add an OOD variant where the new action differs from training-time new actions.

For example:

Training refinement:

```text
new action makes v relevant through immediate distortion
```

OOD refinement:

```text
new action makes v relevant only through future transition
```

The theory says both should refine the dynamic quotient.

If HG responds to both, it is evidence that the learned mechanism is tied to decision relevance rather than a memorized action-template pattern.

---

# 10. Preserve M0-R1 baseline comparisons

M1 must keep the matched vanilla controls.

For every major representation result report:

\[
H_t^{LM,raw},
\]

\[
H_t^{LM,kD},
\]

\[
S_t^{HG}.
\]

Where \(k\) equals the HG latent dimension.

Probe:

```text
current relevant variables
future relevant variables
reusable hidden variables
irrelevant persistent variables
irrelevant noise
capability-refined variables
```

Compute:

\[
Q_{\mathrm{sel}}
=
\bar A_{\mathrm{relevant}}
-
\bar A_{\mathrm{irrelevant}}.
\]

Do this separately for:

```text
IID
lexical OOD
structural OOD
parameter OOD
distractor-density OOD
cross-family transfer
capability C1
capability C2
```

---

# 11. New selectivity robustness metric

Define:

\[
\rho_{\mathrm{OOD}}
=
\frac{
Q_{\mathrm{sel}}^{OOD}
}{
Q_{\mathrm{sel}}^{IID}
}
\]

when the IID denominator is positive.

Report for HG and vanilla.

Prediction:

\[
\rho_{\mathrm{OOD}}^{HG}
>
\rho_{\mathrm{OOD}}^{LM}.
\]

Do not use this alone; also report absolute values because ratios become unstable near zero.

---

# 12. Information-deficit and accessibility transfer

Repeat N1/N3 evaluation under OOD and cross-family conditions.

For HG:

\[
\hat R^{info}
\]

and

\[
\hat R^{access}.
\]

For vanilla:

reuse a fair confidence/proxy baseline, but improve the control slightly.

M0-R1 used generic next-token confidence features because vanilla had no action head.

For M1, add a post-hoc **linear decision probe** on the vanilla trunk trained on the same probe-fit split, then derive:

```text
decision probability
decision entropy
top1-top2 decision margin
```

This produces a stronger vanilla confidence control without altering pretraining.

Report both:

```text
generic next-token confidence proxy
decision-probe confidence proxy
```

The stronger decision-probe baseline should be the primary control.

---

# 13. Routing evaluation

Preserve:

\[
\{ACT, THINK, ASK\}
\]

for the core M1 comparison.

Do not add RETRIEVE or free-form query generation yet unless separately isolated.

Use:

\[
C(r)
=
L(r)+\lambda_B B(r)+\lambda_K K(r).
\]

Evaluate the same or denser price grid as M0-R1.

Primary M1 routing questions:

1. Does HG still beat vanilla under OOD?
2. Does HG route newly relevant capability-refinement cases correctly?
3. Does the routing advantage survive cross-family transfer?

Report:

```text
IID grid win rate
lexical OOD grid win rate
structural OOD grid win rate
cross-family grid win rate
capability-refinement grid win rate
```

---

# 14. Multi-turn ASK remains deferred unless required

Do not make M1 primarily about query synthesis.

The M0 single-hop ASK limitation is known.

If capability-refinement or cross-family tasks require multiple hidden bits, either:

1. construct primary confirmatory items so one structured ASK is sufficient; or
2. implement a separate clearly labeled multi-turn acquisition subexperiment.

Do not mix a major action-space redesign into the primary M1 architecture test unless necessary.

---

# 15. Sequential reuse

Keep growing-context reuse as the matched mechanism for core M1.

Add a harder horizon test:

```text
fact grounded at t0
irrelevant turns inserted
fact required again at t0 + k
```

Sweep:

\[
k\in\{1,2,4,8,16\}
\]

within the context window.

Measure repeated-ASK probability and decision accuracy.

Prediction:

HG's explicit information-deficit machinery should remain more stable as irrelevant context accumulates.

This tests reuse robustness without introducing asymmetric persistent memory.

---

# 16. Training schedule

Use staged execution.

## Stage M1-0 — micro validation

Before 150M full runs:

- instantiate model;
- verify exact parameter counts;
- verify all generators;
- verify capability-refinement ground truth exhaustively on small state spaces;
- train a small debug model;
- verify refinement metric behaves correctly.

No scientific claims.

---

## Stage M1-1 — one-seed pilot

Run:

```text
Baseline-150M seed 0
HG-150M seed 0
```

Evaluate all IID/OOD/transfer/refinement suites.

Stop if:

- N0 fails badly;
- HG latent collapses;
- LM degradation becomes large;
- capability-refinement generator leaks capability labels;
- OOD splits are accidentally overlapping.

Do not immediately launch replication if the pilot is unhealthy.

---

## Stage M1-2 — replication

If healthy:

```text
Baseline seeds 0,1,2
HG seeds       0,1,2
```

Use matched evaluation episodes.

---

# 17. Primary M1 hypotheses

## H-M1-1 — selectivity survives OOD

\[
Q_{\mathrm{sel}}^{HG,OOD}
>
Q_{\mathrm{sel}}^{LM,OOD}.
\]

---

## H-M1-2 — HG selectivity degrades less under distractors

As irrelevant state grows:

\[
\frac{dQ_{\mathrm{sel}}^{HG}}{dN_{\mathrm{distractor}}}
\]

should be less negative than vanilla's.

---

## H-M1-3 — cross-family transfer

HG info/accessibility discrimination remains above matched vanilla controls on an unseen task family.

---

## H-M1-4 — capability refinement

\[
\Delta_{\mathrm{refine}}^{HG}>0
\]

for a variable made newly relevant by action-frontier expansion.

---

## H-M1-5 — irrelevant controls remain suppressed

For a variable `u` irrelevant under both \(C_1,C_2\):

\[
\mathrm{ProbeAcc}(u\mid C_2)
-
\mathrm{ProbeAcc}(u\mid C_1)
\approx0.
\]

This distinguishes true refinement from generic representation expansion.

---

## H-M1-6 — ordinary LM cost remains small

HG should preserve the M0 pattern of modest LM degradation.

Report:

\[
\Delta NLL.
\]

Do not set an arbitrary hard threshold in advance, but any material degradation must count against the architecture.

---

# 18. Falsification conditions

## F-M1-0 — OOD collapse

HG wins IID but loses its selectivity advantage under lexical/structural OOD.

Interpretation: M0 may have learned generator-specific supervision patterns.

---

## F-M1-1 — no capability refinement

The newly relevant variable remains suppressed under \(C_2\).

Interpretation: the HG latent may be a fixed bottleneck, not an adaptive decision quotient.

---

## F-M1-2 — indiscriminate refinement

Both newly relevant `v` and permanently irrelevant `u` become highly encoded under \(C_2\).

Interpretation: capability tokens may trigger generic state expansion rather than quotient refinement.

---

## F-M1-3 — vanilla matches refinement

If vanilla dimension-matched controls show the same selective C1→C2 transition, HG refinement is not a distinctive effect.

Still report it, but do not overclaim architecture specificity.

---

## F-M1-4 — cross-family failure

Info/accessibility heads remain near-perfect IID but collapse to chance on unseen families.

Interpretation: supervised heads are family classifiers rather than transferable deficit estimators.

---

## F-M1-5 — large LM degradation

If HG gains selectivity by sacrificing ordinary modeling substantially, the result becomes a specialization tradeoff rather than a generally useful pretraining architecture.

---

# 19. Statistical protocol

Use three seeds for confirmatory results.

For paired model comparisons report:

```text
per-seed metric
mean
seed std
paired item-level bootstrap 95% CI
```

Primary comparisons:

\[
\Delta Q_{\mathrm{sel}}^{IID},
\]

\[
\Delta Q_{\mathrm{sel}}^{OOD},
\]

\[
\Delta_{\mathrm{refine}},
\]

\[
\Delta AUROC_{\mathrm{info}},
\]

\[
\Delta AUROC_{\mathrm{access}},
\]

\[
\Delta C_{\mathrm{routing}}.
\]

Do not treat item bootstrap as a substitute for seed variance.

---

# 20. Required evaluation matrices

Produce a matrix like:

| condition | HG Q_sel | LM raw | LM matched-D | HG−LM | 95% CI |
|---|---:|---:|---:|---:|---|
| IID | | | | | |
| lexical OOD | | | | | |
| structural OOD | | | | | |
| parameter OOD | | | | | |
| distractor OOD | | | | | |
| cross-family | | | | | |
| capability C1 | | | | | |
| capability C2 | | | | | |

Capability refinement table:

| seed | probe(v|C1) | probe(v|C2) | Δrefine | probe(u|C1) | probe(u|C2) |
|---|---:|---:|---:|---:|---:|

---

# 21. Recommended artifact structure

```text
results/hg_native_m1/
  configs/
  seed0/
    baseline/
    hg/
  seed1/
    baseline/
    hg/
  seed2/
    baseline/
    hg/

  eval/
    iid/
    lexical_ood/
    structural_ood/
    parameter_ood/
    distractor_ood/
    cross_family/
    capability_refinement/
    reuse_horizon/

  probes/
    raw/
    matched_dim/
    hg_latent/

  confidence_controls/
    generic/
    decision_probe/

  bootstrap/
  summary/
```

Every evaluation item should preserve:

```text
episode_id
family
split_type
capability
full generator state
relevant-variable labels
irrelevant-variable labels
true quotient id
model prediction
router prediction
latent/probe output references
```

---

# 22. Provenance requirements

Retain the M0-R1 provenance fix.

Every run must record:

```text
git SHA
dirty state
config hash
tokenizer hash
generator version/hash
seed
device
PyTorch/CUDA versions
parameter count
training token count
wall time
```

Do not run confirmatory jobs from an undocumented dirty tree if avoidable.

If dirty, save a patch/diff artifact alongside the manifest.

---

# 23. Required report

Produce:

```text
HG_Native_M1_150M_Generalization_Refinement_Report.md
```

The report must be self-contained.

Required sections:

1. scientific question;
2. exact model/config;
3. training health and ΔNLL;
4. IID replication;
5. lexical OOD;
6. structural OOD;
7. parameter/distractor OOD;
8. cross-family transfer;
9. capability-refinement result;
10. info/accessibility controls;
11. routing;
12. reuse horizon;
13. falsification checks;
14. statistical summary;
15. theory interpretation;
16. next scale/no-scale decision.

Do not frame every positive result as confirmation. Explicitly identify failed hypotheses.

---

# 24. M1 success criteria

M1 should be considered a strong success only if M0's advantage survives beyond IID synthetic structure.

At minimum require:

1. HG retains positive matched-baseline selectivity advantage on at least lexical and structural OOD;
2. cross-family transfer is materially above vanilla control on at least one core HG quantity;
3. capability refinement shows the predicted selective `v` C1→C2 increase while `u` stays suppressed;
4. routing advantage survives at least one OOD/transfer condition;
5. ordinary LM degradation remains quantified and acceptable.

The capability-refinement result is especially important.

If M1 shows only “same M0 effect at 150M IID,” classify it as a weak result.

---

# 25. Interpretation outcomes

## Outcome M1-A — strong generalizing quotient

HG:

- remains selective OOD;
- transfers deficit estimation across families;
- selectively refines when capability changes;
- maintains routing advantage.

Interpretation:

> strong evidence that HG training induces an adaptive task-relative decision representation rather than a generator-specific bottleneck.

This would justify a larger-scale and more natural-domain program.

---

## Outcome M1-B — selective but task-bound

HG remains better than vanilla IID/OOD within known families but cross-family/refinement transfer is weak.

Interpretation:

> HG learns useful task-relative compression, but not yet a general grounding mechanism.

Next step should redesign training diversity/objectives before further scale.

---

## Outcome M1-C — scaling washes out advantage

Vanilla 150M becomes nearly as selective as HG.

Interpretation:

> the M0 advantage may be primarily a small-model inductive-bias effect.

This is scientifically important. Do not hide it.

Then study scaling curves rather than proceeding automatically.

---

## Outcome M1-D — overcompression failure

HG performs well on old capability \(C_1\) but fails to recover `v` under \(C_2\).

Interpretation:

> current HG latent is too aggressively optimized for the existing frontier and does not refine adaptively.

This would validate the theoretical overcompression risk and require architecture changes.

---

# 26. Claude kickoff instruction

Please execute this milestone conservatively.

1. Read the original M0 spec, M0-R1 kickoff, and M0-R1 replication report.
2. Preserve the confirmed M0 architecture and matched-comparison philosophy.
3. Build M1 around generalization and capability refinement, not around simply making the model larger.
4. Implement the OOD split machinery with strict leakage tests.
5. Implement the capability-refinement generator with exact exhaustive ground-truth tests on small state spaces.
6. Add matched-dimension vanilla probes at the M1 latent dimension.
7. Add the stronger vanilla decision-probe confidence control.
8. Run a small debug model first.
9. Run one 150M Baseline/HG pilot pair.
10. Inspect all failure conditions before launching seeds 1/2.
11. If healthy, run three-seed replication.
12. Produce the self-contained M1 report.
13. Do not add free-form query generation, retrieval, RLHF, LoRA, critics, or persistent HG-only memory during the primary experiment.
14. If any requested design choice cannot be implemented fairly for both Baseline and HG, stop and document the issue before substituting a different protocol.

The final M1 decision must answer:

\[
\boxed{
\text{Does HG learn an adaptive decision quotient that survives distribution shift and refines when capability changes?}
}
