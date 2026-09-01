# HG-v2 — implementation results, gates G0–G2

**Date:** 2026-09-01
**Repo:** `github.com/marco-hrlic/hg`, branch `claude/hg-v2-phase0-phase1` (7 commits)
**Against:** `plans/HG_v2_Claude_Implementation_Plan.md`
**Audience:** the theory / planning track. This is a report *back* from
implementation, written to be read without the codebase in hand.

---

## 0. One-paragraph summary

Phases 0, 1 and 2 of the plan are implemented and their gates pass. The
headline: on a **frozen off-the-shelf Qwen2.5-3B-Instruct**, at equal final
decision distortion, an LM-synthesized targeted query cost **2.52 bits** against
generic human escalation's **5.00** — a **1.98× reduction** in external
information, with the oracle query at **1.00 bit** showing 3× still unclaimed.
Gate G2 is therefore met, and the first HG-v2 architectural result exists. Four
measurement artifacts had to be fixed to get there, three of which would have
produced flattering nonsense, and five points below are **theoretical issues the
plan under-specified** — those are the parts most likely to be useful to you.

---

## 1. Results by gate

### G0 — mathematical sanity (passed)

The index problem: source holds `M_H = (X_0..X_{n-1})` i.i.d. Bernoulli(½), AI
holds index `K`, decision is `J = X_K`, so `H(J|M_A) = 1` bit exactly. 48,000
episodes, n ∈ {2,4,8,16,32,64}.

| policy | predicted | observed |
|---|---|---|
| ACT accuracy | 0.5 | 0.476–0.519 across n |
| Generic ask bits | n | 2, 4, 8, 16, 32, 64 — exact |
| Targeted ask bits | 1 | 1.00 at every n — exact |
| Ω generic | n−1 | 1, 3, 7, 15, 31, 63 — exact |
| Ω targeted | 0 | 0.00 — exact |
| Both asks' accuracy | 1.0 | 1.0000 — exact |

A fourth arm was added beyond the plan's three: a **joint-sufficiency negative
control** where the source does not hold `X_K`. It pays the same 1 bit and
returns a well-formed but uninformative answer; accuracy falls to ~0.5. This
makes "paid for information" and "acquired information" measurably distinct,
which turns out to matter (see §2.1).

### G1 — routing value (passed)

Task `noisy_index`: the AI additionally holds `m = 9` noisy observations of
`X_K`, noiseless on 40% of items (so `J` is directly observable from `M_A`) and
flipped at rate 0.15/0.30/0.45 otherwise. Optimal route known per item from
counterfactuals. Logistic router on internal-agreement features, fitted on
6,000 items, scored on 6,000 held out. No RL.

Mean total cost `C = L + λB`:

| λ | always_act | always_ask | **learned** | oracle | beats both? |
|---:|---:|---:|---:|---:|:---:|
| 0.01 | 0.0908 | 0.0100 | **0.0056** | 0.0009 | ✓ |
| 0.02 | 0.0908 | 0.0200 | **0.0103** | 0.0018 | ✓ |
| 0.05 | 0.0908 | 0.0500 | **0.0231** | 0.0045 | ✓ |
| 0.10 | 0.0908 | 0.1000 | **0.0380** | 0.0091 | ✓ |
| 0.15 | 0.0908 | 0.1500 | **0.0524** | 0.0136 | ✓ |
| 0.20 | 0.0908 | 0.2000 | **0.0662** | 0.0182 | ✓ |
| 0.30 | 0.0908 | 0.3000 | **0.0806** | 0.0272 | ✓ |
| 0.40 | 0.0908 | 0.4000 | 0.0908 | 0.0363 | ✗ (ties always_act) |

**7 of 8 prices**, each with an item-paired bootstrap CI on the cost advantage
excluding zero. At λ = 0.40 no estimated error probability exceeds the price of a
bit, so the router correctly collapses to always-ACT; a tie is not counted.
Query rate falls monotonically with λ.

### G2 — query compression (passed, on a real frozen LM)

Task `route_choice`: two routes, decision rule **in the prompt** so the correct
action never depends on the model's preferences. One route requires a traveller
concession; whether it is acceptable is withheld and held by a simulated source.
`H(J|M_A)` = 0 bits when the concession route is slower (the rule closes without
it) and 1 bit when it is faster. The source holds an opinion on all five
concessions, so generic escalation costs 5 bits and the gold query 1.

`Qwen2.5-3B-Instruct`, 1000 items, seed 0, NVIDIA GB10:

| arm | accuracy | acc·ref=A | acc·ref=B | bits | resolution | Ω̂ (needs grounding) | exact gold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b0_no_grounding` | 0.3460 | 0.101 | 0.579 | 0.00 | 0.4900 | −1.00 | — |
| `b1_self_consistency` | 0.3570 | 0.099 | 0.602 | 0.00 | 0.4900 | −1.00 | — |
| `b2_generic_ask` | 0.7710 | 0.762 | 0.780 | 5.00 | 1.0000 | +4.00 | 0.00 |
| `b3_oracle_targeted` | 0.7750 | 0.778 | 0.772 | 1.00 | 1.0000 | **+0.00** | 1.00 |
| `b4_synthesized_query` | 0.7720 | 0.772 | 0.772 | **2.52** | 1.0000 | +1.56 | 0.58 |

- **Gate G2:** B4 vs B2 — 2.48 bits saved [2.36, 2.60], **1.98×**, final-loss
  gap −0.0010 (marginally in B4's favour). Equal-distortion condition holds.
- **Mechanism (B3 vs B2):** 5.00 → 1.00 bits, exactly `Ω = 4` vs `Ω = 0`.
- All three ask arms land within 0.004 accuracy of each other, which is what
  makes the equal-distortion comparison legitimate. The ~0.77 ceiling is the
  3B's rule-following limit, a *separate* quantity from query cost.

---

## 2. Theoretical issues the plan under-specified

This is the section written for you. Each item is a place where measurement
forced a correction to the theory-to-architecture mapping, not just to code.

### 2.1 Ω requires a sufficiency precondition

The plan defines `Ω = C→ − H(J|M_A)` unconditionally (§1, §7). Taken literally
this produces two nonsense readings that showed up immediately:

- **ACT** communicates nothing, so `Ω = −H(J|M_A) = −1` — apparently beating the
  source-coding bound. It has compressed nothing; the decision is simply
  ungrounded.
- **The joint-sufficiency control** spends its 1 bit, resolves nothing, and gets
  `Ω = 0` — which reads as *optimal*.

Ω is only interpretable for communication that actually closed the deficit.
Implementation reports it as undefined otherwise, with a status
(`no_communication` / `insufficient_communication`). **Suggested theory change:**
state Ω over sufficient one-way protocols only, or define a companion quantity
for insufficient communication (bits spent per unit of deficit *not* closed).

### 2.2 Query quality decomposes into soundness and minimality — and §16 conflates them

§16 gives one excess-cost quantity, `Ω̂ = B_query − B_optimal`. The real-LM run
shows two independent failure modes that it sums together:

- **Soundness / targeting:** does the query name the decision-relevant variable?
  Measured as resolution accuracy. The 3B scored **1.0000** — the relevant
  variable appeared in *every one* of 1000 queries.
- **Minimality:** does it name *only* that? The 3B scored **0.58** exact-gold and
  averaged 2.52 bits.

So this model is **perfectly sound and badly non-minimal**, and a single Ω̂ of
+1.56 cannot express that. The distinction matters for intervention design:
non-minimality is plausibly fixable by prompt/budget pressure, whereas
unsoundness would indicate a representational failure. **Suggested theory
change:** decompose Ω̂ into a soundness term (probability the answer fails to
resolve, weighted by consequence) and a minimality term (excess alphabet given
resolution).

### 2.3 The query distribution is bimodal, so means mislead

Of 1000 synthesized queries: **578 named exactly one variable** (the optimum) and
**337 named all five** (generic escalation reached by another route); 85 in
between. The 2.52-bit mean describes neither mode. Parse rate was 1.000 and the
fallback never fired, so this is model behaviour, not measurement.

**Implication for theory:** modelling a query synthesizer as "optimal query plus
noise" is wrong for this model. It behaves like a mixture of a competent
targeting policy and a give-up-and-escalate policy. If that generalises, the
right object is a *policy mixture weight*, and the lever is whatever selects
between the modes.

### 2.4 Routing and query synthesis are separable, and the LM has zero routing ability

The synthesizer said "no query needed" (`NONE`) **zero times out of 1000**, and
its targeting quality was identical whether grounding was needed (exact-gold
0.571) or not (0.586). It therefore asked on all **490 items whose prompt
already settled the answer** — paying ~2.5 bits each for nothing.

This is the strongest structural result in the run. Deciding *whether* to ask
and deciding *what* to ask are empirically independent capabilities: Phase 1's
router does the first well and the frozen LM does the second well, and neither
does the other. **Suggested planning consequence:** the architecture needs both
components explicitly; they are not alternative implementations of one
mechanism. Expected effect of combining them on this task is roughly another
halving of external cost on top of G2's 1.98×.

### 2.5 The plan's Phase 2 model scale is at or below the task's capability floor

§12 prefers 1B–3B and suggests `gemma-3-1b-it`. Measured on 120 fully-grounded
items:

| model | ungrounded | grounded | acc·ref=A | acc·ref=B | says A |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 0.500 | 0.492 | 0.000 | 0.983 | 0.008 |
| Qwen2.5-3B-Instruct | 0.292 | **0.633** | 0.700 | 0.567 | 0.567 |

**1.5B cannot do the task at all** — it answers one label on essentially every
item and grounding changes nothing. G2 compares *at equal final distortion*, so
it presupposes the model can use the information; at 1.5B there is no such
point. 3B works but tops out ~0.77 on rule-following.

**Consequence:** a G2/G3 run at 1B would measure inability to apply a
compositional rule, not query compression, and G4's capability-scaling claim
would inherit the confusion. Recommend the plan state a **capability floor
check** as a formal precondition of G2 — grounded accuracy must be materially
above chance and balanced across correct-answer classes before any cost
comparison is read.

### 2.6 Routing regret is insensitive to estimator error within threshold windows

§10 treats routing regret as the measure of router quality. But a route is a
threshold on an estimate, so estimation error only changes a decision when λ
falls strictly between two estimates. In the G1 run the learned router's
per-level estimates were **wrong by up to 0.030** while its *policy* matched the
Bayes-optimal one at every swept price — because those divergence windows cover
only **13.4%** of the λ span and no grid price landed inside one. A test
confirms the mechanism by re-pricing inside a window, where the policies do
diverge.

**Implication:** G1 as specified cannot distinguish a good estimator from a
perfect one, and passing it says less about the decision-state estimator than
§18 assumes. If estimator quality is the object of interest, it needs a
direct calibration measure, not regret.

---

## 3. Measurement artifacts found (relevant to trust in earlier results)

Three of these would have produced results that looked fine.

1. **A scoring bug in existing M2-LLM infrastructure, on its default path.**
   `hg.inference.hf._label_logprob` scored `" " + label`, correct for a raw-text
   prompt ending mid-line but wrong after a chat template, which ends at a token
   boundary — a *different token id*. Measured on Qwen2.5-1.5B: the spaced form
   scored −16.9 where the bare form scored −0.001, i.e. it was reading the far
   tail. Every label then scores uniformly badly and the argmax is decided by
   tail noise, so a model looks non-discriminating. **`scoring_mode: rank` and
   `use_chat_template` are both defaults, so this affects the M2-LLM path too**
   and any prior rank-scored result from a chat-templated model should be
   re-checked.

2. **A label-bias confound in the route_choice task (mine).** With the
   concession always on route A, B was correct on ¾ of items — and 1.5B scored
   **0.767 while reading nothing**. Which label carries the concession is now
   randomized; the marginal is symmetric (0.505/0.495) so constant-answering
   scores exactly chance. Accuracy split by which action is correct is now a
   permanent per-arm metric: the 3B's aggregate 0.708 on the old task ranked
   *below* the always-B model's 0.767, so an aggregate alone ranks them
   backwards.

3. **Abstract rule phrasing was harder than the decision.** Stating the rule over
   "the route with the requirement" put 3B at **0.233 — below chance**, almost
   exactly the 0.25 from always picking the requirement-carrying route. The rule
   is now instantiated with each item's actual labels. Worth noting as a general
   hazard: prompt indirection can dominate the effect being measured.

4. **NaN reaching JSON artifacts.** `json.dump` emits bare `NaN`, which is not
   valid JSON. Undefined metrics are now `None`, with a test asserting the bytes.

---

## 4. Reproduction and cost

```bash
make hg-v2-phase0      # G0, CPU, ~1 min
make hg-v2-phase1      # G1, CPU, ~1 min
make hg-v2-phase2      # G2 pipeline on a simulated backend, CPU, ~1 min

scripts/spark_deploy.sh setup
scripts/spark_deploy.sh run 'experiments/hg_v2/run_phase2.py \
  --config configs/experiments/hg_v2/phase2_qwen3b.yaml'
```

- **352 tests** pass, on x86 and on aarch64.
- The real G2 run took **~18 min** wall-clock on one GB10 for 1000 items × 5
  arms. Dominated by unbatched label scoring (one forward pass per item per
  label surface form); batching would cut this ~4×.
- **Execution path:** ssh + uv with the CUDA-13 aarch64 index pinned, **not**
  `docker/Dockerfile`. That container path is unexercised on the GB10, and
  `pyproject.toml`'s `torch>=2.4` has no aarch64 marker, so a bare
  `[gpu]` install there resolves default PyPI wheels.
- Every metric in §1 was recomputed **on CPU from saved episodes**, no model
  calls. The GPU/analysis split in §2.2 of the plan holds in practice.

---

## 5. Limits of these results

State these alongside any claim built on the report:

- **One task family per gate.** G2 rests on a single construction
  (`route_choice`). G3 as specified wants the effect to survive more than that.
- **One model, one seed** for G2. No scaling curve yet, so G4 is untouched.
- **The 3B's rule-following ceiling is ~0.77**, so "equal final distortion" is
  equality at a mediocre level, not at competence. The compression claim is
  sound at that level but has not been shown at high accuracy.
- **The source is truthful, complete and one-shot.** No reliability, no
  multi-turn, so no communication-complexity subtleties.
- **Phase 0 and 1 are synthetic by design** and are evidence about the
  abstractions, not about any model.
- **`DELIBERATE` is not implemented at all**, deliberately (§20) — nothing here
  speaks to reference formation or contested anchors.

---

## 6. Recommended next steps, in priority order

1. **Route and synthesize in one system.** Highest expected value and directly
   motivated by §2.4: the LM asks a perfectly targeted question 58% of the time
   but asks on *every* item, including the 49% needing none. Cheap to build —
   both components exist.
2. **Close the remaining 3× (minimality).** Oracle 1.00 bit vs LM 2.52, entirely
   attributable to the 34% escalation mode. Since resolution is already 1.0000
   this is not a targeting problem. Cheap tests: a stated bit budget in the
   synthesis prompt; rejecting queries above a `max_variables` cap (already
   implemented, unused).
3. **G4: a second model scale.** 7B is the natural next point (1.5B is below the
   floor, 3B at 0.77). The capability probe exists.
4. **A second task family**, so G3 does not rest on one construction.
5. **Re-check any prior rank-scored M2-LLM result** against §3.1.
6. Only then G5 (LoRA, learned critics, joint training), per §27.

---

## 7. Deviations from the plan, with reasons

| deviation | reason |
|---|---|
| Configs under `configs/experiments/hg_v2/` | repo convention; §3 says prefer it over the plan's tree |
| Reduced route enum (`ACT`, `ASK_GENERIC`, `ASK_TARGETED`) | §4.2 permits; `DELIBERATE` left undefined so §20's distinction cannot be collapsed by accident |
| 4th Phase-0 arm (joint-sufficiency control) | §24 requires the test; made an arm so it appears in the results table |
| Ω reported as undefined in two cases | §2.1 above |
| `Qwen2.5-3B-Instruct`, not `gemma-3-1b-it` | gemma is licence-gated; 1.5B is below the capability floor (§2.5). Model is config-driven, so this is a one-line change |
| G2 gated only when the backend is real | a simulator cannot bear a claim about a frozen LM either way; simulated runs gate on the mechanism check instead |
