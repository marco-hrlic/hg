# HG-v2 — Human Grounding v0.6 prototype

The research question this experiment family exists to answer:

> Can a system identify residual decision-relevant information and acquire it
> through a targeted query with less external information cost than generic
> human escalation?

Not "can we build HG-1B". The question is whether the theory-to-architecture
mapping survives measurement, and it is approached through gates (G0–G5) that
each have to pass before the next is worth building.

Sibling to `experiments/m2_llm/`, which it deliberately does not disturb. What
is reused from that infrastructure and what is new — with reasons — is in
[REUSE_AUDIT.md](REUSE_AUDIT.md).

---

## Status

The gates come from two documents. G0–G2 are the original implementation plan
(`plans/HG_v2_Claude_Implementation_Plan.md`); H0–H6 are the improved plan
(`plans/HG_v2_Improved_Experiment_Plan_FIXED.md`), written after G0–G2 and the
v0.8 theoretical refinements, and they supersede G3–G5 rather than sitting
alongside them.

| Gate | Question | State |
|---|---|---|
| **G0** | Does the simulator reproduce known theory? | **passed** |
| **G1** | Does a learned router beat fixed ACT/ASK policies? | **passed** |
| **G2** | Does targeted query synthesis compress external cost? | **passed** on Qwen2.5-3B-Instruct |
| **H0** | Does the whole bounded-query frontier reproduce `b*(n,r)`? | **passed** (E1) |
| **H1** | Does routing buy anything on top of query synthesis? | see E2 below |
| **H2** | Does a minimality intervention make the query cheaper? | see E3 below |
| H3 | Does a second model pass the capability floor? | not started |
| H4 | Does the effect persist across two capable models? | not started |
| H5 | Does it survive a second task family? | not started |
| H6 | Is custom training justified? | not started |

Phases 0, 1 and 2 are complete and their gates pass. G2 was evaluated on a real
frozen model (Qwen2.5-3B-Instruct, on the DGX Spark): **at equal final
distortion, an LM-synthesized query cost 2.52 bits against generic escalation's
5.00 — a 1.98× reduction**, with the oracle query at 1.00 showing 3× still on
the table. Details below.

Next is G4 (does the grounding requirement change with model capability), which
needs a second model scale, and G3 for a second task family.

---

## Phase 0 — the exact synthetic complementarity experiment

No LLM, no GPU, runs in about a minute on a laptop. Its purpose is not to
produce evidence but to check the abstractions against a theorem before any of
them are pointed at a model.

### The task

The source privately holds `M_H = (X_0, …, X_{n-1})`, i.i.d. Bernoulli(1/2).
The AI privately holds an index `K`. The decision is

    J = X_K

so `H(J | M_A) = 1` bit **exactly** — knowing `K` tells the AI which bit
decides, never its value — and `H(J | M_A, M_H) = 0`: the two states are
jointly sufficient. Both facts hold by construction.

The two ask policies reach identical, perfect decision quality, so accuracy
cannot separate them. The only thing that can is what the source had to
transmit. That is precisely the axis HG-v2 claims to move.

### The arms

| policy | behaviour | accuracy | response bits | Ω |
|---|---|---:|---:|---:|
| `act` | guess from `M_A` alone | ≈ 0.5 | 0 | undefined — nothing was communicated |
| `ask_generic` | source transmits all `n` bits | 1.0 | `n` | `n − 1` |
| `ask_targeted` | AI asks for `X_K` | 1.0 | 1 | 0 |
| `ask_targeted_ablated` | source does **not** hold `X_K` | ≈ 0.5 | 1 | undefined — cost paid, deficit not closed |

The fourth arm is a negative control added beyond the plan's three (plan §24
requires a joint-sufficiency test). It matters because it separates *paying for
information* from *acquiring* it: same query, same 1-bit cost, no resolution.

### Coordination overhead

`Ω = C_→ − H(J | M_A)` is reported **only when the communication actually
closed the deficit**. Two cases would otherwise print numbers that read as
results:

- `act` never communicates, so the subtraction gives `Ω = −1` — apparently
  beating the source-coding bound. It has not compressed anything; the decision
  is simply ungrounded.
- the ablated arm spends its bit and resolves nothing, giving `Ω = 0`, which
  reads as *optimal*.

Both are reported as undefined with a status (`no_communication`,
`insufficient_communication`). A negative Ω from a *sufficient* arm would be a
bug in the bit accounting, not a discovery.

### Run it

```bash
make install-uv       # uv-managed .venv, CPU only
make test-uv          # 220 tests
make hg-v2-phase0     # gate G0; exits non-zero if any criterion fails
```

or directly:

```bash
uv run python experiments/hg_v2/run_phase0.py \
  --config configs/experiments/hg_v2/phase0_index.yaml
```

### Re-analyse without rerunning

```bash
uv run python experiments/hg_v2/analyze.py --run-dir results/<run_id>
```

Every number in `report.md` is recomputed from `episodes/*.jsonl` alone — the
task is never regenerated and no model is called. The runner *itself* takes this
path: it writes the episodes, analyses them by reading them back, and compares
that against the in-memory result. Plan §8's criterion 6 is therefore a
comparison the run performs and fails on, not a claim in a docstring.

### What a run writes

```
results/<timestamp>_hg_v2_phase0_index/
├── config.resolved.yaml   the config that produced this run
├── manifest.json          commit, timestamp, seed, task version, environment
├── episodes/<policy>.jsonl item-level episodes  ← the artifact that matters
├── figures/               3 plots, each with its CSV
├── analysis.json          every computed quantity
└── report.md              the results table and the gate verdict
```

The episode files are large (~40 MB for the default sweep) because each row is
self-describing: both private states, the decision state, the full query, the
response, and loss before *and* after grounding. That is deliberate — a row
that has lost its query cannot be re-analysed — but it means run directories
are gitignored and `n_items` is worth lowering for exploratory runs.

---

---

## Phase 1 — learned routing (gate G1, passed)

Phase 0's task always needed the source, so routing was trivial. Phase 1 needs
a task where the optimal route varies per item **and is known**.

### The task: `noisy_index`

Same index problem, but the AI also holds `m = 9` internal observations of
`X_K`. On 40% of items they are noiseless — `J` is directly observable from
`M_A` and asking is waste (the plan's "with probability p, J is directly
observable"). On the rest each observation is independently flipped with rate
0.15, 0.30 or 0.45, so the AI's majority vote is unreliable and a 1-bit
targeted ask resolves the decision.

`m = 9` rather than 5: the agreement features take only `(m+1)/2` distinct
values, and three levels is coarse enough that *any* correctly-ordered
estimator reproduces the exact Bayes policy, leaving nothing to measure about
estimation quality. Five levels gives the router room to be wrong.

The noise is *graded* rather than binary on purpose. If the AI simply knew
whether it was self-sufficient, the routing feature would predict the label
perfectly and there would be no learning problem, only a lookup. With graded
noise the router sees the **agreement among its own observations** — informative
about whether acting is safe, never conclusive.

### The router

A logistic regression on three internal-agreement features (majority margin,
vote entropy, unanimity), fitted by gradient descent in numpy. No RL (plan §9),
no new dependency. It estimates one thing:

```
P(acting now is wrong | internal agreement)
```

because with `C_r = L_r + λB_r` the route rule collapses to `ACT iff E[L_act] ≤
λB_ask`. Fitting the probability rather than a per-λ classifier hits the same
supervised target — for `0 < λB < 1`, `r*` *is* the "acting would be wrong"
indicator — while letting one fitted model be repriced across the whole λ sweep.

Fitted on half the items, scored on the held-out half. Every policy is
evaluated on the same items in the same order, which is what makes the
comparison item-paired.

### Result

Mean total cost `C = L + λB` on held-out items (6000 items, seed 0):

| λ | always_act | always_ask | **learned** | oracle | wins both? |
|---:|---:|---:|---:|---:|:---:|
| 0.01 | 0.0908 | 0.0100 | **0.0056** | 0.0009 | ✓ |
| 0.02 | 0.0908 | 0.0200 | **0.0103** | 0.0018 | ✓ |
| 0.05 | 0.0908 | 0.0500 | **0.0231** | 0.0045 | ✓ |
| 0.10 | 0.0908 | 0.1000 | **0.0380** | 0.0091 | ✓ |
| 0.15 | 0.0908 | 0.1500 | **0.0524** | 0.0136 | ✓ |
| 0.20 | 0.0908 | 0.2000 | **0.0662** | 0.0182 | ✓ |
| 0.30 | 0.0908 | 0.3000 | **0.0806** | 0.0272 | ✓ |
| 0.40 | 0.0908 | 0.4000 | 0.0908 | 0.0363 | ✗ (ties always_act) |

**Gate G1 passed at 7 of 8 prices**, each with an item-paired bootstrap interval
on the cost advantage that excludes zero. At λ = 0.40 no estimated error
probability exceeds the price of a bit, so the router correctly collapses to
always-ACT and ties it — a tie is not counted as a win.

The query rate falls monotonically as external bits get dearer (0.54 → 0.43 →
… → 0.00), which is the sanity check on the whole cost model.

### Estimation error vs. routing decisions

The learned router's per-level estimates are **not** exact — they differ from
the exact posterior by up to 0.030 — yet its cost matches `bayes_optimal` at
every swept price. That is worth quantifying rather than asserting, and the run
does:

A route is chosen by comparing an estimate against λ, so the two policies differ
only when λ falls strictly between their two estimates at some feature level.
Each level contributes a divergence window of that width, and here the windows
cover **0.052 of a λ span of 0.39 (13.4%)** — none of the eight swept prices
lands inside one.

| majority margin | learned P(wrong) | exact P(wrong) | divergence window |
|---:|---:|---:|---|
| 0.111 | 0.3577 | 0.3878 | width 0.0301 |
| 0.333 | 0.1741 | 0.1903 | width 0.0162 |
| 0.556 | 0.0660 | 0.0654 | width 0.0006 |
| 0.778 | 0.0198 | 0.0154 | width 0.0043 |
| 1.000 | 0.0013 | 0.0003 | width 0.0010 |

So the coincidence is a **robustness property**, not an exact estimator: route
decisions on this task tolerate estimation error of the size the router actually
has. A test verifies the claim by acting on it — re-pricing at a λ *inside* the
widest window makes the two policies diverge, and a λ outside every window
leaves them identical.

One subtlety that cost a false positive in an earlier version of the validity
check: `bayes_optimal` is the ceiling in **expected** cost, while a run measures
**realized** cost. The two policies differ only on items whose true error
probability sits near the price of a bit — precisely where the realized rate can
land on either side by chance — so the learned router can come out nominally
cheaper than its own ceiling with nothing wrong. The check is therefore an
item-paired bootstrap that flags only a *significant* advantage, which is what a
genuine feature leak would produce. Both directions are tested.

### What Phase 1 still does *not* show

**The query is already optimal.** Phase 1 routes between acting and asking a
query that was handed to it. Whether a *synthesized* query compresses external
cost is Phase 2's question — and the answer there turned out to be
complementary: the frozen LM synthesizes a good query but never decides *not* to
ask, which is exactly what this router does.

### Run it

```bash
make hg-v2-phase1     # gate G1; exits non-zero if it does not hold
```

---

## Phase 2 — a frozen LM and query compression (gate G2, passed)

Phase 0 showed a targeted query beats generic escalation when the query is
handed to you. Phase 1 showed a router can decide *whether* to ask. Phase 2 asks
the question the architecture actually rests on: can the system decide *what* to
ask, in natural language, and does that compress external cost?

### The task: `route_choice` (plan §14)

Two routes. The decision rule travels **in the prompt**, so the correct action
never depends on the model's own preferences:

```
- If route A's travel time is longer than route B's, choose B.
- Otherwise, if route A's requirement is acceptable to the traveller, choose A.
- Otherwise choose B.
```

Route A requires one traveller concession — a mountain road, a toll, a ferry,
night driving, an unpaved stretch. **Whether that concession is acceptable is
withheld from the prompt** and held by a simulated traveller. So the residual
deficit varies per item and is exact:

| case | `H(J \| M_A)` | |
|---|---:|---|
| route A slower | 0 bits | the rule closes without the hidden variable |
| route A faster | 1 bit | the withheld bit decides |

The source holds an opinion on **every** concession. Generic escalation
therefore costs one bit per variable (5 bits here) while the gold query costs 1 —
Phase 0's structure again, now in natural language and with the query no longer
given.

### Baselines (plan §17)

| arm | grounding | external bits |
|---|---|---|
| `b0_no_grounding` | none | 0 |
| `b1_self_consistency` | none, 5 internal samples | 0 |
| `b2_generic_ask` | source dumps every preference | 5 |
| `b3_oracle_targeted` | the gold query | 1 |
| `b4_synthesized_query` | a synthesized query | depends on the query |

Every ask arm asks on **every** item, so the query is the only difference
between them. Routing is Phase 1's contribution and G1 already measured it;
excess response cost is reported both overall and over the items that genuinely
need grounding, so the cost of not routing stays visible instead of being folded
into the query result.

### Result on a frozen LM — gate G2 passed

`Qwen2.5-3B-Instruct`, 1000 items, seed 0, on the DGX Spark. `b4` uses the
`lm` synthesizer: the same frozen model is asked which facts it needs, and its
answer is parsed into a `QuerySpec` comparable to the gold one.

| arm | accuracy | acc \| ref=A | acc \| ref=B | bits | resolution | Ω̂ (needs grounding) | exact gold query |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b0_no_grounding` | 0.3460 | 0.101 | 0.579 | 0.00 | 0.4900 | −1.00 | — |
| `b1_self_consistency` | 0.3570 | 0.099 | 0.602 | 0.00 | 0.4900 | −1.00 | — |
| `b2_generic_ask` | 0.7710 | 0.762 | 0.780 | 5.00 | 1.0000 | +4.00 | 0.00 |
| `b3_oracle_targeted` | 0.7750 | 0.778 | 0.772 | 1.00 | 1.0000 | **+0.00** | 1.00 |
| `b4_synthesized_query` | 0.7720 | 0.772 | 0.772 | **2.52** | 1.0000 | +1.56 | 0.58 |

**Gate G2: passed.** B4 against B2 — 2.48 bits saved [2.36, 2.60], a **1.98×
reduction at equal final distortion** (loss gap −0.0010, i.e. marginally
*better*). The mechanism check (B3 vs B2) is 5.00 → 1.00, exactly the
theoretical `Ω = 4` vs `Ω = 0`.

Grounding lifts accuracy from 0.346 to ~0.77 and all three ask arms land within
0.004 of each other — which is what makes "equal final distortion" hold and the
cost comparison legitimate. The ~0.77 ceiling is the 3B's rule-following limit,
a separate matter from query cost. The A/B split is balanced for every grounded
arm, so the label-bias confound is closed.

### What the synthesized queries actually looked like

The mean of 2.52 bits is an average over two modes, not a tendency to hedge:

| variables named | count | what it is |
|---:|---:|---|
| 1 | **578** | the optimal query |
| 3 | 83 | a hedge |
| 4 | 2 | a hedge |
| 5 | **337** | generic escalation, reached by another route |

So the model is *bimodal*: 58% of the time it asks exactly the right single
question, and 34% of the time it asks for everything. Parse rate was 1.000 and
the fallback never fired, so this is the model's behaviour and not the parser's.

Two further observations, both of which matter more than the headline ratio:

- **Targeting is sound but not minimal.** Resolution accuracy is **1.0000** — the
  relevant variable was included in *every* query, all 1000. The model never
  asks the wrong thing; it only over-asks. Excess response cost is +1.56 bits
  against the gold query's 0.00.
- **It has no routing ability at all.** It said `NONE` zero times, and its
  targeting is the same whether grounding was needed (exact-gold 0.571) or not
  (0.586) — so it asked on all 490 items whose prompt already settled the
  answer. That is precisely the decision Phase 1's router makes well, and it is
  the clearest argument in these results that routing and query synthesis are
  complementary rather than alternative mechanisms.

### Result on the simulated backend

Kept for reference: `configs/experiments/hg_v2/phase2_frozen_lm.yaml` runs the
same five arms on CPU with `grounding_simulator` and a controlled stand-in
synthesizer. It does **not** evaluate G2 — the code enforces that — but it does
check that the metrics detect compression when present (the oracle arm goes 5.00
→ 1.00 at identical accuracy), and it is how the pipeline was validated before
any GPU time was spent.

### The LM query synthesizer

`hg.queries` offers three synthesizers, selected by config:

| kind | what produces the query |
|---|---|
| `gold` | the algorithmic optimum — one bit, always the right variable |
| `noisy` | a controlled stand-in with known failure modes, for validating the metrics |
| `lm` | **the primary model itself**, asked what it needs to know |

`kind: lm` is what makes B4 a test of the architecture rather than of a
stand-in. Four design decisions determine what its numbers mean:

- **The output space is variable names, not free text.** The model picks from a
  listed set of available facts, so a synthesized query is directly comparable
  to the gold `QuerySpec` — same representation, same cost function. The
  measured quantity is *targeting*, not phrasing.
- **The instruction says asking costs something** ("name as few as possible —
  each one costs the traveller effort"). A prompt that merely asked "what do you
  need?" invites hedging, and a hedge is precisely the failure mode G2 exists to
  detect.
- **An unparseable answer falls back to generic escalation.** Deliberately
  unflattering: asking nothing would let a model that failed to form a query
  score as *cheap*. The fallback rate is reported so the cost is attributable.
- **`NONE` is honoured.** On an item whose prompt already settles the rule, the
  right query is no query, and recognising that costs zero bits. This is the one
  place B4 can legitimately beat the always-ask oracle B3 — routing and query
  synthesis arriving in the same decision.

Parsing matches canonical names and registered aliases ("tolls", "ferries",
"night driving"), reads only the first line — small instruct models continue
past the answer, and a later "I do not need tolls or ferries" would otherwise
read as asking for them — and returns variables in the task's declared order so
two models naming the same set produce the same query and the same cost.

### Swapping in a real model

The primary LM is replaceable from config (plan §12) — nothing above the backend
knows what produced the candidate action. `configs/experiments/hg_v2/phase2_qwen15b.yaml`
is the real run:

```yaml
backend: transformers
backend_options:
  checkpoint: Qwen/Qwen2.5-1.5B-Instruct   # 1B-3B instruct, per plan §12
  dtype: bfloat16
  scoring_mode: rank      # label log-probs, so no parser in the decision arms
synthesizer:
  kind: lm
```

It drives the repository's existing `Backend` protocol — the same one the
M2-LLM experiment uses — so there is no second model-loading path.

The plan suggests `google/gemma-3-1b-it`, which is licence-gated and needs a
Hub token; Qwen2.5-1.5B-Instruct is ungated and in the same size band. The model
is config-driven precisely so this is a one-line change, and gate G4 will want
at least two sizes or families anyway.

One wrinkle worth knowing: `scoring_mode: rank` is deterministic — it scores each
label's continuation log-probability and takes the argmax — so "five samples"
from it would return the same answer five times and B1 would be B0 with a
different name. B1 therefore switches the backend to sampled generation for its
own passes, restoring the mode afterwards, which keeps the weights loaded once.

### Run it

```bash
make hg-v2-phase2
```

---

## E1 — the full bounded-query frontier (gate H0, passed)

No model, no GPU, about a minute. Improved plan §§4–5 replaces Phase 0's two
endpoints with the whole curve between them.

### The claim

For deterministic, fixed-length, single-shot, **zero-error** protocols where the
AI may spend `r` bits naming what it wants and the source answers with `b` bits,

    b*(n, r) = ceil(n / 2^r)

The `r` query bits partition the `n` candidate indices into at most `2^r`
fibers. Some fiber has size `s ≥ ceil(n / 2^r)`, and conditional on that query
message the answer must determine `X_K` for *every* `K` in the fiber — so the
response map restricted to those coordinates is injective on `{0,1}^s` and
`b ≥ s`. Achievability is the balanced partition: send the fiber verbatim.

Phase 0's two results are the endpoints of this curve: `r = 0` gives `b* = n`
(generic escalation) and `r = log₂ n` gives `b* = 1` (a targeted query).

### Why it is run rather than asserted

An arithmetic identity that no code path can violate is not a regression test.
So each `(n, r)` cell is simulated twice — once at `b*` bits, which must reach
accuracy 1, and once at `b* − 1`, which must not. Either half alone is
compatible with the bound being wrong in the other direction.

The one-bit-short control fails with probability `(s − b) / 2s`, which is
**1/128** at `n = 64, r = 0`. At a fixed item count the check therefore gets
weaker as `n` grows, and a run at 400 items has a 4% chance of seeing no failure
at all and reporting the bound as not tight. Each cell is instead sized to buy
`gate.min_expected_failures` expected failures, so "tight here" means the same
thing at every point of the frontier.

Observed at `n = 64` — the table from plan §4, reproduced by measurement:

| r | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `b*` predicted | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
| `b` observed | 64 | 32 | 16 | 8 | 4 | 2 | 1 |
| accuracy at `b*` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| accuracy at `b*−1` | 0.994 | 0.984 | 0.969 | 0.938 | 0.870 | 0.748 | 0.495 |

### E1b — the lossy frontier

Three claims, kept apart because the plan (§5) is emphatic that merging them is
the error to avoid:

| quantity | status |
|---|---|
| finite converse `b ≥ s(1 − h₂(ε))` | **exact at every blocklength** — the rate-distortion converse applied to one fiber |
| asymptotic rate `b* → n·2^{−r}(1 − h₂(ε))` | a limit; **not** a prediction about any single cell at these sizes |
| achievability | what a constructed code attained — an **upper** bound on `b*`, since the codebook is best-found, not proven optimal |

Codebooks are designed by Lloyd search over all `2^s` source words where `s` is
small enough to enumerate, and by a structured "send `b` coordinates, guess the
rest" scheme where it is not; each cell reports which. Distortion is computed
exactly over the whole source alphabet, not sampled, so the design criterion has
no sampling error in it — the realized error rate is reported beside it as a
finite-sample check.

Gate H0 gates on the zero-error identity, on the truncated control failing, and
on the converse. It does **not** gate on approaching the asymptotic rate: at
`s ≤ 10` that would be gating on a limit the experiment cannot reach.

### Run it

```bash
make hg-v2-e1
```

---

## E2/E3 — routing and query synthesis in one system (gates H1, H2)

The improved plan's central experiment (§§6–10), and the one the Phase 2 result
most obviously demanded: the frozen LM asked a perfectly targeted question 58%
of the time and **never once declined to ask**, spending bits on all 490 items
whose prompt already determined the answer. Phase 1's router makes exactly that
decision. This is the first run where both exist in one system.

    route -> query -> external response -> decision

### The factorial (plan §7)

| arm | route | query |
|---|---|---|
| `b0_no_grounding` | never | — |
| `b1_self_consistency` | never | — (more internal compute) |
| `b2_generic_ask` | always | every preference the source holds |
| `b3_oracle_route_oracle_query` | oracle | the gold one-bit query |
| `b4_always_ask_synth` | always | synthesized — reproduces Phase 2's B4 |
| `b5_learned_route_synth` | learned | synthesized — **the integrated arm** |
| `b6_learned_route_oracle_query` | learned | gold — isolates routing error |
| `b7_oracle_route_synth` | oracle | synthesized — isolates synthesis error |

B6 and B7 are the two single-substitutions of B5, which is what makes the gap
decomposable: `routing_gap = B5 − B7` and `synthesis_gap = B5 − B6`, in mean
external bits at matched quality. Without both, one disappointing number would
be uninterpretable.

### One cache, eight arms

Every arm is a different *policy* over the same three model behaviours: what the
model answers ungrounded, given a generic dump, and given a targeted answer.
Each is measured once per item and the arms are assembled from that cache. Two
consequences, and the second is the important one: GPU time does not scale with
the arm count, and the arms are **exactly item-paired by construction** rather
than by coincidence. A λ sweep is then free — repricing a saved counterfactual
is arithmetic — which is why the price grid is dense (plan §13: a coarse sweep
can hide estimator error inside a threshold window).

### The router on a real model

Phase 1's router ran on a synthetic task where the internal evidence was
constructed. Here it runs on the frozen LM's own signals, and only those:

| feature | from |
|---|---|
| `majority_margin`, `vote_entropy_bits`, `unanimous` | k sampled generations of the ungrounded prompt |
| `label_margin`, `top_label_prob` | the model's own label distribution under rank scoring |

Nothing about the withheld variable, the source, or the reference action enters.
A router that could see those would report its own leak as routing skill — and
note that `needs_grounding` *is* AI-visible on this task (it is whether the
concession route is faster), so a structural router would be near-oracle and
would measure nothing. The question worth asking is whether the model's internal
signal supports the decision, which is falsification F1's habitat.

The estimator is scored directly, not only through the policy it feeds (plan
§2.3): Brier, ECE, MCE, MAE, a calibration curve, and the rank correlation
between the predicted and the empirical value of asking
`V_ask = L_ACT − L_ASK − λB` at every price. Phase 1 established why: its
learned router and the exact posterior produced *identical routes* while holding
different numbers, so routing regret alone cannot see estimator error.

### E3 — minimality interventions (plan §9)

The Phase 2 gap was not a targeting failure. Resolution was 1.0000 and the cost
distribution was bimodal — one variable most of the time, all five the rest — so
what is left is minimality. Four variants, all asking on every item so the query
is the only thing that differs:

| variant | prompt | enforcement |
|---|---|---|
| `q0_baseline` | `query_synth.v1` | none — the comparator |
| `q1_minimality` | `query_synth.v2_minimal` (ask only what can change the decision) | none |
| `q2_budget` | `query_synth.v3_budget` (at most one preference) | none |
| `q3_hard_cap` | `query_synth.v3_budget` | truncate to one variable |

Q2 against Q3 separates *stating* a budget from *imposing* one. Q3's truncation
can break soundness — if the dropped name was the deciding one, the query is now
cheap and useless — which is falsification F2 made measurable rather than
argued. Gate H2 therefore requires a cost reduction **and** soundness within
`gate.max_soundness_loss` **and** decision quality within the distortion
tolerance; any one of the three failing means the intervention traded rather
than improved.

Plan §10 asks for the distribution rather than a summary, so the report prints
the full per-query variable-count distribution and the mixture weights
`alpha_minimal` and `share_full_escalation`.

### Soundness, minimality and overhead are separate (plan §§2.1–2.2)

- **Soundness** is a property of the query, not of the answer the model then
  gave: a sound query the model misapplied is a capability failure, and merging
  the two makes both unreadable.
- **Minimality excess** `B_Q − B*` is reported *conditional on soundness*.
  Averaging an unsound 1-bit query into the excess would make a broken
  synthesizer look economical.
- **Coordination overhead** is `null` with `protocol_sufficient: false` whenever
  the protocol left the deficit open — including an ACT on an item that needed
  grounding, even one that happened to come out right. It is never negative.

### Run it

```bash
make hg-v2-e2                                    # simulated, validates the metrics
scripts/spark_deploy.sh run 'experiments/hg_v2/run_e2.py \
  --config configs/experiments/hg_v2/e2_integrated_qwen3b.yaml -v'
```

The simulated run does **not** evaluate H1 or H2 — those are claims about a
frozen off-the-shelf model. What it establishes is that the factorial, the cost
accounting, the calibration and the minimality metrics detect what they are
meant to detect, which is the precondition for the test rather than the test.

---

## Reading these results honestly

Neither phase involves a model, and both reports say so on their own first page.

Phase 0's index theorem is trivial mathematically. Reproducing it proves exactly
one thing: that the abstractions — AI private state, external private state,
decision variable, query, response cost, final decision, episode artifact,
analysis pipeline — represent the grounding problem correctly. **No Phase 0
number is evidence about any model's behaviour**, and the report says so on its
own first page.

---

## Running a real model: the DGX Spark

Phases 0 and 1 need no GPU, and Phase 2 runs on CPU with the simulated backend.
A real checkpoint needs the Spark — an NVIDIA **GB10, aarch64**, 121 GB, behind
the `Notch` VPN.

**This repository reaches it by ssh + uv, not through `docker/Dockerfile`.**
That Dockerfile's container path has not been exercised on the GB10, and
`pyproject.toml`'s `torch>=2.4` carries no aarch64/cu130 marker — so a bare
`uv pip install -e ".[gpu]"` there resolves against default PyPI wheels. The
CUDA-13 aarch64 index has to be pinned explicitly, which is what the sibling
`research/gated-lm` and `research/snap-picker` projects do:

```bash
scripts/spark_deploy.sh setup          # rsync + create the venv with cu130 wheels
scripts/spark_deploy.sh run 'experiments/hg_v2/run_phase2.py \
  --config configs/experiments/hg_v2/phase2_qwen3b.yaml'
```

Verified on the box: `torch 2.13.0+cu130` with CUDA available, `transformers
5.16.1`, and the full test suite green on aarch64. Two things to know:

- The VPN drops intermittently. `nmcli connection up Notch` if the host stops
  answering; the deploy script checks reachability first and says so.
- `rsync --delete` keeps the remote tree an exact mirror, so a scratch script
  written directly on the Spark disappears on the next sync. Keep throwaway
  files outside the repo tree.

### Choosing a model: measure the capability floor first

Gate G2 is a comparison *at equal final distortion*, so it presupposes the
primary LM can use the information at all. Check that before spending a run.
120 items, fully grounded, on the symmetric task:

| model | says A, ungrounded | says A, grounded | acc \| ref=A | acc \| ref=B |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 0.000 | 0.000 | 0.000 | 1.000 |
| Qwen2.5-3B-Instruct | 0.000 | 0.342 | 0.607 | 0.739 |

**1.5B is unusable here**: it answers the same label on every item and grounding
changes nothing. On the *asymmetric* version of the task it scored 0.767
accuracy while reading nothing — which is what prompted randomizing the labels
(see the Phase 2 task section) and what makes the per-half split a permanent
part of the metrics rather than a one-off check.

3B responds to disclosed constraints (0% → 34% on the minority label) with
imperfect rule-following, which is enough for G2: the comparison only needs the
model to *use* the fact it is given, not to be right about everything.

## Next

Gates G0-G2 have passed and G3's core question — does the effect survive a
frozen off-the-shelf LM — is answered affirmatively by the Phase 2 run. What is
outstanding, in the order the plan wants it:

1. **Route and synthesize together.** The single clearest result to chase from
   Phase 2: the LM asks a perfectly targeted question 58% of the time but asks
   on *every* item, including the 49% that needed no query. Phase 1's router
   supplies exactly that missing decision, and the two have never been run in
   one system. Expected effect is large — roughly halving external cost again,
   on top of G2's 1.98×.
2. **Close the remaining 3×.** The oracle query costs 1.00 bit against the LM's
   2.52, and the gap is entirely the 34% of items where the model asks for
   everything. Since resolution is already 1.0000, this is a *minimality*
   problem, not a targeting one — a promising place for a cheap intervention
   (a stated bit budget, or rejecting queries above `max_variables`).
3. **Gate G4: a second model scale.** Whether the external grounding requirement
   falls with capability. 1.5B is below the task's floor and 3B sits at ~0.77
   rule-following, so 7B is the natural next point; the capability probe in
   `scripts/` is the tool.
4. **A second task family**, so G3 does not rest on one construction.
5. Only then G5 (LoRA, learned critics, joint training) — per plan §27.
