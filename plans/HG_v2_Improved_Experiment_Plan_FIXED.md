# HG-v2 Improved Experiment Plan

**Project:** Human Grounding / HG-v2  
**Repository:** `https://github.com/marco-hrlic/hg`  
**Basis:** Results from G0–G2 plus the subsequent v0.8 theoretical refinements.

## 0. Objective

The first HG-v2 experiment established that the synthetic complementarity abstraction behaves as predicted, a learned router can outperform fixed ACT/ASK policies over a broad range of external-information prices, and synthesized targeted queries can reduce external communication cost on a frozen real LM at essentially equal final distortion.

The improved experiment should now ask:

> Can an HG-v2 system jointly decide **whether** external information is needed and **what minimally sufficient query** to issue, while approaching the theoretical query–response frontier and preserving decision quality?

The core architecture under test is

\[
\boxed{
\text{route}
\rightarrow
\text{query}
\rightarrow
\text{external response}
\rightarrow
\text{decision}
}
\]

with routing, query soundness, and query minimality measured separately.

---

## 1. Key lessons from completed G0–G2

### 1.1 Synthetic complementarity is validated

For the index problem,

\[
M_H=(X_1,\ldots,X_n),\qquad M_A=K,\qquad J=X_K,
\]

the previous experiment reproduced

\[
H(J\mid M_A)=1,
\]

\[
B_{\mathrm{generic}}=n,
\]

and

\[
B_{\mathrm{targeted}}=1.
\]

The next experiment should test the full bounded-query frontier rather than only these endpoints.

### 1.2 Routing is independently useful

The learned router successfully selected between ACT and ASK over a meaningful range of external-information prices. Therefore

\[
\text{whether to ask}
\]

and

\[
\text{what to ask}
\]

should remain separate modules.

### 1.3 Query synthesis reduces communication

On Qwen2.5-3B-Instruct,

\[
B_{\mathrm{generic}}=5.00,\qquad
B_{\mathrm{synth}}=2.52,\qquad
B_{\mathrm{oracle}}=1.00,
\]

with nearly equal final decision quality.

Define oracle-gap closure

\[
\eta_Q
=
\frac{
B_{\mathrm{generic}}-B_{\mathrm{system}}
}{
B_{\mathrm{generic}}-B_{\mathrm{oracle}}
}.
\]

For the existing run,

\[
\eta_Q
=
\frac{5.00-2.52}{5.00-1.00}
\approx 0.62.
\]

The remaining gap came primarily from non-minimal queries, not from failure to identify the relevant variable.

---

## 2. Theory updates that must be reflected

### 2.1 Coordination overhead is defined only for sufficient protocols

Do not define

\[
\Omega=C-H(J\mid M_A)
\]

for a protocol that fails to achieve the required distortion.

For distortion tolerance \(D\), define

\[
\mathcal P_D
=
\left\{
P:
\mathbb E[d(J,\widehat J)]\le D
\right\}.
\]

Define

\[
C^*(D)
=
\inf_{P\in\mathcal P_D} C(P),
\]

and only then

\[
\Omega(D)
=
C^*(D)-R_J(D\mid M_A).
\]

If a realized protocol fails the criterion, report:

```text
coordination_overhead = null
protocol_status = insufficient
```

Never assign negative coordination overhead.

### 2.2 Query quality decomposes into soundness and minimality

Measure separately:

**Soundness:** does the query elicit enough information to resolve the required decision?

For exact tasks:

\[
S(Q)
=
\mathbf 1
\left[
\widehat J(Q,Y)=J
\text{ after truthful response}
\right].
\]

**Minimality:** conditional on soundness, how much larger is the response cost than the optimum?

\[
M_{\mathrm{excess}}(Q)
=
B_Q-B^*.
\]

Soundness failure and non-minimality are different failure modes.

### 2.3 Routing quality and estimator quality are separate

Routing regret alone can hide estimator error because thresholded policies may remain unchanged.

Report:

- routing regret;
- Brier score;
- expected calibration error;
- calibration curves;
- mean absolute error against empirical counterfactual value;
- rank correlation between predicted and actual value of asking.

### 2.4 Capability floor is a formal precondition

Before interpreting query compression for a model, verify that it can use perfectly supplied grounding information.

A model passes only if:

1. fully grounded accuracy is materially above chance;
2. accuracy is balanced across correct-action classes;
3. grounding materially improves performance;
4. constant-label behavior is absent.

Models below this floor must not be used for grounding-efficiency or scaling conclusions.

---

## 3. Improved experimental program

Run four linked experiments:

```text
E1 — bounded-query frontier validation
E2 — integrated routing + query synthesis
E3 — query-minimality intervention
E4 — capability/model replication
```

Then:

```text
E5 — second task family
```

Do not begin joint training or LoRA before E1–E5 are understood.

---

## 4. E1 — Full bounded-query index frontier

The v0.8 zero-error theorem gives

\[
\boxed{
b^*(n,r)
=
\left\lceil
\frac{n}{2^r}
\right\rceil
}
\]

for deterministic fixed-length zero-error protocols.

The old experiment tested only the endpoints. Test the entire frontier.

Use

\[
n\in\{8,16,32,64\}
\]

and for each \(n\),

\[
r=0,1,\ldots,\lceil\log_2 n\rceil.
\]

For each pair verify

\[
b_{\mathrm{observed}}
=
\left\lceil\frac{n}{2^r}\right\rceil.
\]

For \(n=64\):

| Query bits \(r\) | Optimal response bits \(b^*\) |
|---:|---:|
| 0 | 64 |
| 1 | 32 |
| 2 | 16 |
| 3 | 8 |
| 4 | 4 |
| 5 | 2 |
| 6 | 1 |

This becomes an automated theorem-regression test.

---

## 5. E1b — Lossy bounded-query frontier

Add

\[
\varepsilon\in
\{0.00,0.05,0.10,0.20,0.30,0.40\}.
\]

For balanced fibers of size

\[
s=\frac{n}{2^r},
\]

the finite converse is

\[
b
\ge
s[1-h_2(\varepsilon)].
\]

In the large-fiber regime,

\[
\boxed{
b^*(n,r,\varepsilon)
\approx
n2^{-r}[1-h_2(\varepsilon)].
}
\]

Keep three claims distinct:

```text
exact single-shot fixed-length zero-error theorem
finite lossy converse
asymptotic lossy rate prediction
```

Do not claim finite single-shot equality for the lossy case.

---

## 6. E2 — Integrated router + query synthesizer

Build:

```text
Decision-state estimator
        |
        v
ACT / ASK router
        |
        +---- ACT
        |
        +---- ASK ----> Query synthesizer
                           |
                           v
                     external response
                           |
                           v
                       final action
```

Do not collapse router and synthesizer into one opaque prompt.

Reuse the corrected `route_choice` task first and use the same Qwen2.5-3B-Instruct model to obtain a paired before/after comparison.

Preserve:

- randomized concession placement;
- symmetric label marginals;
- item-specific instantiated decision rule;
- machine-verifiable correct action;
- known grounding-required vs grounding-not-required items.

---

## 7. E2 arms

Run:

### B0 — no grounding

### B1 — self-consistency / more internal compute

### B2 — generic ask on every item

### B3 — oracle router + oracle targeted query

### B4 — always ask + synthesized query

Replicates existing G2.

### B5 — learned router + synthesized query

Main integrated HG-v2 arm.

### B6 — learned router + oracle targeted query

Isolates routing error.

### B7 — oracle router + synthesized query

Isolates synthesis error.

This factorial design is required to distinguish routing and query-synthesis gaps.

---

## 8. E2 metrics

For every arm report:

### Decision quality

\[
D_{\mathrm{final}}
\]

or accuracy, split by correct action class.

### Query rate

\[
P(\mathrm{ASK}).
\]

Report separately on grounding-needed and grounding-not-needed items.

### Human response cost

\[
B_{H\rightarrow A}.
\]

Report mean and full distribution.

### AI query cost

Where meaningful, report

\[
B_{A\rightarrow H}
\]

or at least query cardinality / number of variables referenced.

Keep this separate from human response cost for now.

### Soundness

Probability that the query supplies enough information to resolve the decision.

### Minimality

Conditional on soundness:

\[
B_Q-B^*.
\]

### Oracle-gap closure

\[
\eta_Q
=
\frac{
B_{\mathrm{generic}}-B_{\mathrm{system}}
}{
B_{\mathrm{generic}}-B_{\mathrm{oracle}}
}.
\]

Only interpret when final distortion is matched.

### Waste rate

Measure bits spent on items already determined by AI-visible information.

This is a central integrated-routing metric.

---

## 9. E3 — Query minimality intervention

The previous LM was nearly perfectly sound but often non-minimal.

Test:

### Q0 — existing synthesizer

### Q1 — explicit minimality instruction

Example:

```text
Ask only for information that can change the final decision.
Do not request information irrelevant once a narrower answer would determine the action.
```

### Q2 — explicit response-bit budget

Example:

```text
You may ask about at most one binary user variable.
```

### Q3 — structured hard cap

Use existing `max_variables` enforcement.

### Q4 — candidate-query reranking

Generate multiple candidate queries and select the one minimizing predicted response cost subject to predicted soundness.

Do not use final-answer correctness for inference-time selection.

---

## 10. E3 analysis

Report the full distribution:

```text
variables requested per query
response alphabet size
raw response bits
exact-gold rate
soundness
minimality excess
```

Because the previous distribution was bimodal, explicitly estimate a mixture:

\[
\pi_Q
=
\alpha\pi_{\mathrm{minimal}}
+
(1-\alpha)\pi_{\mathrm{fallback}}.
\]

Estimate

\[
\alpha.
\]

The primary question is whether minimality interventions increase \(\alpha\) without reducing soundness.

---

## 11. E4 — Capability-floor and model-scale replication

Use a stronger model next, preferably:

```text
Qwen2.5-7B-Instruct
```

or a comparable open-weight model if infrastructure makes that more convenient.

Do not use the 1.5B result as a grounding-efficiency scaling point because it was below the task capability floor.

For each model measure:

```text
ungrounded accuracy
fully grounded accuracy
accuracy by correct class
constant-label tendency
grounding gain
```

Predeclare a grounded-competence threshold before running the next model.

Only models passing the threshold enter coordination-efficiency comparisons.

For each passing model report:

\[
P(\mathrm{grounding\ needed}),
\]

\[
B_{\mathrm{response}}\mid\mathrm{ASK},
\]

\[
P(\mathrm{unnecessary\ ask}),
\]

query soundness,

query minimality,

\[
\eta_Q,
\]

and final distortion.

Do not summarize capability scaling with one external-bit average.

---

## 12. E5 — Second task family

After E2–E4, add one new controlled task with different causal structure.

Good candidates:

- product/selection task;
- scheduling task;
- resource-allocation task.

Requirements:

- exact ground truth;
- controlled hidden information;
- known optimal query;
- known whether grounding is required;
- multiple possible query granularities.

Avoid open-ended preference dialogue at this stage.

---

## 13. Decision-state estimator evaluation

For each item obtain counterfactual ACT and ASK outcomes.

Define empirical value of asking:

\[
V_{\mathrm{ask}}
=
L_{\mathrm{ACT}}
-
L_{\mathrm{ASK}}
-
\lambda B.
\]

Predict

\[
\widehat V_{\mathrm{ask}}.
\]

Report:

- mean absolute error;
- Brier score;
- calibration plot;
- Spearman correlation;
- routing regret over a dense \(\lambda\)-grid.

Use a dense grid because coarse price sweeps can conceal estimator error inside threshold windows.

---

## 14. External-information accounting

Maintain separate quantities:

\[
B_{H\rightarrow A},
\]

\[
B_{A\rightarrow H},
\]

and

\[
D.
\]

Human cognitive cost is not yet rigorously defined.

Log only practical proxies such as:

```text
number of requested variables
query length
number of alternatives presented
response complexity
```

Do not call these Shannon bits.

A future objective may be:

\[
C_{\mathrm{total}}
=
C_{A\rightarrow H}
+
C_{H\rightarrow A}
+
\lambda_H C_{\mathrm{human}},
\]

but do not collapse the terms yet.

---

## 15. Updated episode schema

Store at least:

```python
{
    "item_id": ...,

    "ai_state_features": ...,
    "grounding_required_gold": ...,

    "initial_action": ...,
    "initial_correct": ...,
    "initial_loss": ...,

    "router_score": ...,
    "router_route": ...,
    "oracle_route": ...,

    "query_text": ...,
    "query_variables": [...],
    "query_sound": ...,
    "query_exact_gold": ...,
    "query_response_alphabet_size": ...,

    "external_response": ...,
    "external_response_bits": ...,

    "final_action": ...,
    "final_correct": ...,
    "final_loss": ...,

    "protocol_sufficient": ...,
    "coordination_overhead": ...,

    "metadata": ...
}
```

Undefined quantities must serialize as JSON `null`, never `NaN`.

---

## 16. Scoring-infrastructure audit

Before new pretrained-model comparisons:

1. identify all runs using chat templates plus rank/log-probability scoring;
2. verify label-continuation tokenization;
3. compare bare-label and space-prefixed forms;
4. add regression tests for each supported chat template;
5. rerun affected prior results before citing them.

Do not assume prior M2-LLM rank-scored results are valid until this is checked.

---

## 17. Statistical protocol

For real-LM runs:

- use multiple seeds where practical;
- bootstrap paired differences;
- report confidence intervals for communication savings;
- report final-distortion differences with confidence intervals;
- preserve item-level paired outputs;
- report class-conditional accuracy;
- report full query-cost distributions.

Predefine the tolerance used to call two arms "equal distortion."

---

## 18. Updated decision gates

### H0 — theoretical frontier regression

Verify:

\[
b^*(n,r)
=
\left\lceil\frac{n}{2^r}\right\rceil.
\]

Lossy experiments must respect the finite converse and approach the asymptotic prediction where appropriate.

If not: STOP.

### H1 — integrated routing benefit

Require:

\[
B_{\mathrm{learned\ route+synth}}
<
B_{\mathrm{always\ ask+synth}}
\]

at matched final distortion.

### H2 — minimality improvement

At least one query intervention must reduce response cost or increase \(\eta_Q\) without materially reducing soundness or decision quality.

### H3 — capability-floor pass

The second model must pass the predeclared grounded-competence threshold.

### H4 — cross-model persistence

The routing/query-compression effect must appear in at least two capable models.

### H5 — second-task replication

The effect must survive a second controlled task family.

### H6 — only then custom training

Only after H0–H5 consider:

- LoRA;
- learned critics;
- learned decision-state representations;
- joint router/query training;
- custom HG-1B-style architecture.

---

## 19. Immediate implementation order

For the next coding session:

1. implement the full zero-error bounded-query frontier;
2. add theorem-regression tests;
3. integrate the Phase-1 router with the Phase-2 query synthesizer;
4. add B5/B6/B7 factorial arms;
5. add separate soundness and minimality metrics;
6. add waste-rate measurement;
7. add direct router-calibration metrics;
8. run the integrated Qwen2.5-3B experiment;
9. test minimality interventions Q1–Q3;
10. report back before running the 7B model.

Do not begin custom training yet.

---

## 20. Primary hypotheses

### H-A — routing and synthesis compose

\[
B_{\mathrm{route+synth}}
<
B_{\mathrm{always\ ask+synth}}
\]

at matched distortion.

### H-B — soundness remains high under minimality pressure

A bit-budget or variable-cap constraint should reduce cost without a large decline in query soundness.

### H-C — bimodality is a fallback-policy effect

Minimality interventions should mainly reduce the full-escalation mode.

Operationally,

\[
1-\alpha
\]

should fall.

### H-D — capability affects multiple mechanisms

Report separately:

\[
P(\mathrm{ASK}),
\]

query soundness,

query minimality,

and ability to use grounded information.

---

## 21. Falsification criteria

Treat these seriously.

### F1

Integrated routing fails to reduce external cost because routing does not transfer to the LM task.

### F2

Minimality constraints substantially reduce soundness.

### F3

The previous G2 compression result disappears across seeds or after task corrections.

### F4

A second capable model does not reproduce query savings.

### F5

A second task family shows no advantage over generic escalation.

### F6

Query soundness is driven by template leakage rather than identification of decision-relevant state.

If these occur, update the architecture hypothesis instead of preserving it.

---

## 22. Report format after next run

Return:

1. branch / commit;
2. files modified;
3. theorem-frontier validation table;
4. integrated routing results;
5. factorial decomposition B4–B7;
6. soundness/minimality results;
7. query-cost distribution;
8. waste rate;
9. router calibration;
10. minimality-intervention comparison;
11. paired final-distortion intervals;
12. scorer/tokenization regressions found;
13. recommendation on whether to proceed to 7B;
14. assessment: did the HG-v2 architecture hypothesis become stronger, weaker, or unchanged?

Stop for review before the next model-scale experiment.
