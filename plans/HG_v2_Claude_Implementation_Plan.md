# HG-v2 Prototype Implementation Plan

Repository:
https://github.com/marco-hrlic/hg

## 0. Goal

Extend the existing `hg` research repository to support the first experiments for the
Human Grounding v0.6 architecture.

The immediate research question is NOT:

> Can we build the full HG-1B model?

The immediate question is:

> Can a system identify residual decision-relevant information and acquire it through a
> targeted query with less external information cost than generic human escalation?

The first implementation must be small, controlled, reproducible, and compatible with
the repository's existing experimental infrastructure.

Do not rewrite existing working infrastructure unless necessary.

Reuse:
- existing config conventions
- inference abstractions
- metrics/results infrastructure
- simulator support
- routing infrastructure
- finetuning/PEFT support where useful
- existing run artifact format
- current DGX-friendly execution assumptions

Preserve existing experiments, especially `experiments/m2_llm/`.

The new work should live primarily in a new sibling experiment:

`experiments/hg_v2/`

---

# 1. Theoretical objects we are approximating

The architecture is motivated by the v0.6 Human Grounding theory.

For a decision variable

\[
J
\]

and AI internal information

\[
M_A,
\]

the ideal exact grounding requirement is

\[
H(J \mid M_A).
\]

A human/source may possess complementary information

\[
M_H.
\]

If

\[
H(J \mid M_A,M_H)=0,
\]

AI + human information jointly suffice.

For one-way communication, actual communication may exceed the fundamental decision
deficit:

\[
C_{\rightarrow}
\ge
H(J\mid M_A).
\]

Define coordination overhead

\[
\Omega
=
C_{\rightarrow}
-
H(J\mid M_A).
\]

The architectural hypothesis to test is:

> A system that knows its own residual decision state can synthesize a targeted query
> that reduces \(\Omega\), potentially dramatically.

Eventually the router should approximate something related to conditional
decision-rate distortion,

\[
R_C(D\mid M),
\]

but DO NOT try to numerically estimate Shannon rate-distortion in the neural model in
Phase 1.

Use directly measurable proxies first.

---

# 2. Engineering principles

## 2.1 Do not build the full architecture yet

Do not initially implement:

- complex long-term failure memory
- end-to-end joint model training
- RL routing
- full constructive preference dynamics
- sophisticated integrity classifiers
- distributed multi-GPU serving infrastructure
- large custom critics
- new base-model pretraining

These come only after experimental gates justify them.

## 2.2 Separate expensive inference from cheap analysis

Model calls should produce reusable item-level artifacts.

Analysis scripts must be able to recompute metrics without rerunning models.

This is already a design principle in the repository and should be preserved.

## 2.3 Configuration first

Experiments should be driven by YAML/TOML/config objects, not hard-coded model names,
seeds, task sizes, or policies.

## 2.4 Deterministic simulator tests first

Before using an LLM, demonstrate that the experiment infrastructure reproduces exact
known theoretical results.

---

# 3. Proposed package additions

Do not blindly create all of these files if existing modules already provide equivalent
abstractions. Inspect the repository first and reuse what exists.

Suggested organization:

```text
src/hg/
    grounding/
        __init__.py
        channels.py
        episodes.py
        costs.py
        synthetic.py

    routing/
        # preserve existing modules
        decision_state.py
        grounding_router.py

    queries/
        __init__.py
        base.py
        synthetic.py
        lm_query_synthesizer.py
```

New experiment:

```text
experiments/hg_v2/
    README.md
    configs/
        phase0_index.yaml
        phase1_router.yaml
        phase2_frozen_lm.yaml

    run_phase0.py
    run_phase1.py
    run_phase2.py
    analyze.py
```

If repository conventions suggest different paths, follow the repository conventions
rather than forcing this exact structure.

---

# 4. Core data structures

Create small typed dataclasses/Pydantic models where appropriate.

## 4.1 DecisionState

Conceptual structure:

```python
@dataclass
class DecisionState:
    item_id: str

    candidate_actions: list[str]

    predicted_action: str | None

    # empirical/proxy quantities
    expected_loss_now: float | None
    internal_reducibility: float | None
    external_value: float | None
    human_value: float | None
    contestedness: float | None
    consequence: float | None

    metadata: dict[str, Any]
```

Do not pretend these quantities are exact Shannon quantities.

They are practical estimator outputs.

---

## 4.2 GroundingRoute

```python
class GroundingRoute(Enum):
    ACT = "act"
    THINK = "think"
    RETRIEVE = "retrieve"
    OBSERVE = "observe"
    ASK = "ask"
    DELIBERATE = "deliberate"
```

Phase 0 may only use:

```text
ACT
ASK_GENERIC
ASK_TARGETED
```

It is acceptable to implement a reduced route enum initially if cleaner.

---

## 4.3 GroundingEpisode

Every expensive experiment should save enough information to reconstruct what happened.

Suggested fields:

```python
@dataclass
class GroundingEpisode:
    item_id: str

    ai_private_state: Any
    hidden_reference_state: Any

    initial_action: Any
    initial_loss: float

    decision_state: dict

    route: str

    query: Any | None
    external_response: Any | None

    final_action: Any
    final_loss: float

    external_raw_bits: float | None
    external_cost: float

    metadata: dict
```

Use JSONL or the repository's current preferred item-level artifact representation.

---

# 5. Phase 0 — Exact synthetic complementarity experiment

This phase is mandatory.

No LLM.

The goal is to verify the infrastructure against a theorem whose answer is known
analytically.

## 5.1 Index problem

Generate

\[
M_H=(X_1,\ldots,X_n),
\]

where

\[
X_i \sim \operatorname{Bernoulli}(1/2)
\]

independently.

AI receives

\[
K\in\{1,\ldots,n\}.
\]

The required decision is

\[
J=X_K.
\]

Therefore

\[
H(J\mid K)=1.
\]

The human/source knows all \(n\) bits but does not initially know \(K\).

---

# 6. Phase 0 policies

Implement at least these policies.

## Policy A — ACT

AI guesses without external information.

Expected accuracy should approach:

\[
0.5.
\]

External information cost:

\[
0.
\]

---

## Policy B — GENERIC ASK

Human transmits all of

\[
(X_1,\ldots,X_n).
\]

AI recovers

\[
X_K.
\]

Exact decision accuracy:

\[
1.
\]

Fixed-length external response cost:

\[
n \text{ bits}.
\]

---

## Policy C — TARGETED ASK

AI sends query:

\[
Q=K.
\]

Human returns:

\[
Y=X_K.
\]

Accuracy:

\[
1.
\]

Human response cost:

\[
1 \text{ bit}.
\]

Hence:

\[
\Omega_{\text{generic}}=n-1
\]

and

\[
\Omega_{\text{targeted}}=0.
\]

---

# 7. Phase 0 sweep

Run over something like:

```text
n = 2, 4, 8, 16, 32, 64
```

with enough samples that empirical accuracy/cost closely matches theory.

Save raw item-level results.

Generate plots/tables for:

### Accuracy

versus \(n\).

### Human-response bits

versus \(n\).

Expected:

```text
ACT           0 bits
GENERIC       n bits
TARGETED      1 bit
```

### Coordination overhead

\[
\Omega
=
B_{\text{actual}}
-
1.
\]

Expected:

```text
GENERIC       n - 1
TARGETED      0
```

---

# 8. Phase 0 acceptance criteria

Phase 0 passes only if:

1. ACT accuracy approaches 0.5.
2. GENERIC accuracy is exactly or effectively 1.
3. TARGETED accuracy is exactly or effectively 1.
4. Generic human response cost scales linearly with \(n\).
5. Targeted response cost remains one bit.
6. Results can be regenerated from saved item-level artifacts without rerunning the
   simulator.
7. Automated tests cover the theoretical result.

Add unit tests that verify these properties for small \(n\).

Do not proceed until Phase 0 works cleanly.

---

# 9. Phase 1 — Learned routing on controlled tasks

Now create tasks where sometimes the AI already knows enough and sometimes external
information is required.

Example construction:

With probability \(p\):

\[
J
\]

is directly observable from \(M_A\).

Otherwise it depends on one hidden human/source variable.

The optimal route is therefore known.

Train a small router to choose:

```text
ACT
ASK
```

from available features.

Do NOT use RL initially.

Generate counterfactual outcomes for both routes.

For every sample evaluate:

\[
L_{\text{ACT}}
\]

and

\[
L_{\text{ASK}}.
\]

Include an external-query cost

\[
\lambda B.
\]

Define total route objective:

\[
C_r
=
L_r+\lambda B_r.
\]

Optimal route:

\[
r^*
=
\arg\min_r C_r.
\]

Train the router supervised against \(r^*\).

---

# 10. Phase 1 metrics

Report:

### Route accuracy

Fraction of items where chosen route equals oracle route.

### Decision distortion

Final error/loss.

### External-query rate

Fraction of items using external grounding.

### External information cost

Total/mean response bits.

### Routing regret

\[
\operatorname{Regret}
=
C_{\text{chosen}}
-
C_{\text{oracle}}.
\]

### Information efficiency

For example:

\[
\frac{
D_{\text{before}}-D_{\text{after}}
}{
B_{\text{external}}
}.
\]

Avoid division issues at zero bits.

Report the numerator and denominator separately as well.

---

# 11. Phase 1 acceptance criterion

Learned routing must beat BOTH:

```text
always ACT
always ASK
```

for some nontrivial range of external-information costs.

If it cannot, investigate before moving on.

---

# 12. Phase 2 — Frozen off-the-shelf primary LM

Only after Phase 0 and Phase 1 pass.

Use the existing Hugging Face / Transformers backend if possible.

Start with an off-the-shelf open-weight instruct model.

Preferred initial scale:

\[
1\text{B} - 3\text{B}.
\]

Do not train it initially.

Suggested starting candidate:

```text
google/gemma-3-1b-it
```

but put this in config rather than hard-coding it.

The model must remain replaceable.

---

# 13. Primary-LM role

The frozen LM should initially provide:

1. candidate answer/action
2. optionally candidate alternatives
3. hidden representation if existing infrastructure makes extraction straightforward
4. textual rationale only if experimentally useful

Do not make chain-of-thought a required system dependency.

---

# 14. First LM benchmark design

Do not begin with broad chat.

Construct controlled natural-language tasks where:

- the prompt provides partial information;
- one decision-relevant variable is intentionally hidden;
- that variable is available to a simulated human/source;
- the correct action can be computed exactly;
- the optimal query is known.

Example:

```text
You are planning route A or route B.

Known to AI:
- distance
- toll cost
- travel time

Known only to user:
- whether mountain roads are acceptable

Decision rule:
if mountain roads unacceptable -> B
otherwise choose lower travel-time route
```

The model can:

```text
ACT
ASK GENERIC QUESTION
ASK TARGETED QUESTION
```

Ground truth should remain machine-verifiable.

---

# 15. Query synthesizer v0

Do not immediately train a generative query model.

Start with a structured query representation.

Example:

```python
@dataclass
class QuerySpec:
    variable_name: str
    candidate_values: list[str]
    candidate_actions_affected: list[str]
    natural_language_query: str
```

For synthetic tasks, produce the optimal `QuerySpec` algorithmically.

This gives us a gold standard.

Then compare LM-generated queries to that gold standard.

---

# 16. Query quality metrics

A query should be judged primarily by what it does to the decision problem.

Measure:

### Resolution accuracy

Does the answer allow the correct final action?

### Response cardinality / information cost

How many possible responses must the external source distinguish?

For a response alphabet \(\mathcal Y\),

a simple initial upper-bound cost is:

\[
B_{\text{raw}}
=
\log_2 |\mathcal Y|.
\]

Later use empirical entropy where useful.

### Excess response cost

Compare actual query cost to theoretical/gold minimum.

Approximate:

\[
\widehat\Omega
=
B_{\text{query}}
-
B_{\text{optimal}}.
\]

### Query efficiency

\[
\frac{\Delta D}{B_{\text{query}}}.
\]

Again report raw components as well.

---

# 17. Phase 2 baselines

Always compare against:

### B0 — Frozen LM, no grounding

### B1 — Frozen LM + more sampling / internal self-consistency

### B2 — Frozen LM + generic ask-human escalation

### B3 — Frozen LM + oracle targeted query

### B4 — Frozen LM + HG-v2 synthesized query

The important comparison is:

\[
\text{HG-v2 query}
\]

versus

\[
\text{generic human escalation}.
\]

At equal final distortion, does HG-v2 reduce external response cost?

---

# 18. Phase 3 — Decision-state estimator

Only after controlled LM experiments work.

Add learned predictors for quantities like:

\[
\hat D_t
\]

predicted loss if acting now;

\[
\hat r_t
\]

probability additional internal computation resolves the issue;

\[
\hat v_t
\]

value of external information;

\[
\hat h_t
\]

value specifically of human information;

\[
\hat c_t
\]

probability that the reference itself is contested/unresolved;

\[
\hat\rho_t
\]

consequence of being wrong.

These are proxies.

Do NOT label them as estimates of exact Shannon quantities unless we actually derive
and validate that correspondence.

---

# 19. Phase 4 — Full routing menu

Only after binary ACT/ASK routing works.

Expand routes to:

```text
ACT
THINK
RETRIEVE
OBSERVE
ASK
DELIBERATE
```

The route decision should approximate:

\[
r^*
=
\arg\min_r
\left[
\mathbb E L_r
+
\lambda C_r
\right].
\]

Each route must have measurable cost.

Examples:

```text
THINK       GPU / token cost
RETRIEVE    retrieval latency / call cost
OBSERVE     tool/sensor cost
ASK         human response cost
DELIBERATE  larger human time/cognitive cost
```

---

# 20. Deliberation must stay separate from ordinary ASK

`ASK` assumes relevant information already exists on the human/source side.

`DELIBERATE` means the reference itself may need to be formed or adjudicated.

Do not collapse these routes.

This distinction follows directly from the v0.6 theory.

---

# 21. Integrity / steering experiments

Do NOT attempt a universal learned integrity model initially.

Instead construct controlled synthetic cases with explicitly allowed/disallowed queries.

Example:

Allowed:

```text
"Do you prefer A or B?"
```

Disallowed steering variant:

```text
"A is clearly the better choice. You agree, right?"
```

Measure whether a naive query optimizer prefers the steering query because it reduces
response uncertainty.

Demonstrate experimentally that:

\[
\min H(J\mid M,Q)
\]

without an allowed-policy constraint can select degenerate target-shaping strategies.

This should be a later experiment, not Phase 0.

---

# 22. Memory

Do not make generic failure memory central.

If needed, introduce a structured grounding-memory record:

```python
GroundingMemoryItem(
    task_features=...,
    decision_state=...,
    route=...,
    query=...,
    response=...,
    distortion_before=...,
    distortion_after=...,
    source=...,
)
```

Past grounding becomes part of future internal state.

The research question is whether stored grounding episodes reduce future external
information requirements.

Do not implement this until the basic routing/query result works.

---

# 23. Existing code reuse audit

Before implementing anything, inspect the repository and write:

```text
experiments/hg_v2/REUSE_AUDIT.md
```

For each proposed component state:

```text
existing module:
reusable as-is:
needs extension:
new code required:
reason:
```

Pay special attention to:

- inference
- routing
- metrics
- result serialization
- config handling
- simulator infrastructure
- finetuning
- model loading
- seeds/reproducibility

The goal is to minimize new infrastructure.

---

# 24. Testing requirements

Add unit tests for:

### Synthetic index theorem

For small \(n\):

\[
B_{\text{generic}}=n
\]

and

\[
B_{\text{targeted}}=1.
\]

### Joint sufficiency

If relevant source information is removed, perfect recovery must become impossible.

### Router oracle labels

Verify route-cost calculations manually on tiny fixtures.

### Serialization

A saved episode must round-trip losslessly.

### Analysis reproducibility

Metrics computed from saved artifacts must match metrics from the live run.

---

# 25. Reproducibility

Every experiment result should save:

```text
resolved config
git commit
timestamp
seed
model checkpoint
dataset/task version
environment/package versions where available
item-level results
aggregate metrics
```

Do not overwrite previous runs silently.

---

# 26. DGX considerations

The project runs on NVIDIA DGX Spark.

Prefer:

- sequential loading of larger models
- BF16 where supported
- no requirement that all critics/models remain resident simultaneously
- cached primary outputs where scientifically valid
- batching
- deterministic simulator runs

The first phases should be extremely cheap and should not require serious GPU usage.

---

# 27. Decision gates

## Gate G0 — Mathematical sanity

Synthetic simulator reproduces known theory.

If no:

STOP.

---

## Gate G1 — Routing value

Learned router beats fixed ACT/ASK policies under a meaningful cost tradeoff.

If no:

Do not build complex architecture.

---

## Gate G2 — Query compression

Targeted query synthesis achieves lower external information cost than generic human
escalation at the same decision quality.

This is the first important HG-v2 architectural result.

If no:

Investigate whether the theory-to-architecture mapping is wrong.

---

## Gate G3 — Frozen real LM

The effect survives when the AI state is generated by a frozen off-the-shelf LM.

If no:

Determine whether the failure is representation, routing, or query generation.

---

## Gate G4 — Model scaling

Repeat with at least two primary model sizes/families.

Measure whether external grounding requirement changes with model capability.

---

## Gate G5 — Custom training

Only now consider:

- LoRA on primary model
- learned critics
- joint decision-state training
- learned query synthesizer
- larger HG-1B architecture
- joint/end-to-end optimization

---

# 28. First implementation milestone

For the FIRST coding session, do only:

1. Inspect repository.
2. Produce `REUSE_AUDIT.md`.
3. Create minimal `experiments/hg_v2/`.
4. Implement synthetic index task.
5. Implement:
   - ACT
   - GENERIC ASK
   - TARGETED ASK
6. Save item-level results.
7. Add analysis.
8. Add tests.
9. Run sweep over several \(n\).
10. Produce one table/plot demonstrating:

\[
C_{\rightarrow}^{\text{generic}}=n
\]

versus

\[
C_{\leftrightarrow}^{H\rightarrow A}=1.
\]

Do NOT begin Phase 1 until I review the Phase 0 result.

---

# 29. Expected first result

The experiment should demonstrate:

| Policy | Accuracy | Human response bits |
|---|---:|---:|
| ACT | \(\approx 0.5\) | \(0\) |
| Generic ask | \(1.0\) | \(n\) |
| Targeted ask | \(1.0\) | \(1\) |

with coordination overhead:

\[
\Omega_{\text{generic}}=n-1,
\]

\[
\Omega_{\text{targeted}}=0.
\]

This is intentionally trivial mathematically.

Its purpose is to validate that the repository's new abstraction correctly represents:

- AI private state
- external private state
- decision variable
- external query
- response cost
- final decision
- grounding episode
- analysis pipeline

Once this is correct, the same infrastructure can support the real LM experiments.

---

# 30. Coding style

Prefer:

- simple interfaces
- typed Python
- small modules
- explicit data flow
- unit tests
- existing repository conventions
- minimal dependencies

Avoid premature abstractions.

Do not introduce a framework merely because one may be needed later.

If an existing abstraction works, extend it.

Document all non-obvious theoretical assumptions in code comments/docstrings.

---

# 31. Report back after Phase 0

Return:

1. files added/modified
2. reuse decisions
3. architecture summary
4. commands to reproduce
5. test results
6. Phase 0 numerical results
7. generated plots
8. any deviations from this plan and why
9. concerns discovered in existing infrastructure
10. recommendation: proceed to G1 or not

Stop there and wait for review.
