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

| Gate | Question | State |
|---|---|---|
| **G0** | Does the simulator reproduce known theory? | **passed** |
| **G1** | Does a learned router beat fixed ACT/ASK policies? | **passed** |
| G2 | Does targeted query synthesis compress external cost? | **scaffolded, needs a real LM** |
| G3 | Does the effect survive a frozen off-the-shelf LM? | not started |
| G4 | Does the grounding requirement change with model capability? | not started |
| G5 | Is custom training justified? | not started |

Phases 0 and 1 are complete and their gates pass. Phase 2 is built and runs
end to end on a simulated backend, which validates the task, the query
representation, the cost accounting and the query-quality metrics — but **gate
G2 is a claim about a frozen off-the-shelf model, so a simulated run does not
evaluate it**, and the run says so. Testing G2 needs a real checkpoint, which
needs the DGX Spark execution decision below.

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
query that was handed to it. Gate G2 — whether a *synthesized* query compresses
external cost — is untouched.

### Run it

```bash
make hg-v2-phase1     # gate G1; exits non-zero if it does not hold
```

---

## Phase 2 — a frozen LM and query compression (gate G2, scaffolded)

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

### Result on the simulated backend

4000 items, seed 0. `b4` uses a deliberately imperfect stand-in synthesizer
(70% gold query, 20% hedge over three variables, 10% miss):

| arm | accuracy | bits | resolution | Ω̂ (all) | Ω̂ (needs grounding) | gold match |
|---|---:|---:|---:|---:|---:|---:|
| `b0_no_grounding` | 0.7380 | 0.00 | 0.4950 | −0.5050 | −1.0000 | — |
| `b1_self_consistency` | 0.7438 | 0.00 | 0.4950 | −0.5050 | −1.0000 | — |
| `b2_generic_ask` | 1.0000 | 5.00 | 1.0000 | +4.4950 | +4.0000 | 0.00 |
| `b3_oracle_targeted` | 1.0000 | 1.00 | 1.0000 | +0.4950 | **+0.0000** | 1.00 |
| `b4_synthesized_query` | 0.9735 | 1.43 | 0.9497 | +0.9205 | +0.4305 | 0.69 |

**Mechanism check** (`b3` vs `b2`): 5.00 → 1.00 bits, a **5.00×** reduction at
*identical* accuracy (1.0000 both). The cost accounting and query-quality
metrics detect targeted compression when it is present. That is the
precondition for testing G2 — not the test.

**Gate G2** (`b4` vs `b2`, per plan §17) is **not evaluated** here: the backend
is a simulator, and G2 is a claim about a frozen off-the-shelf model. The
comparison is still computed and reported (+3.57 bits saved, but a +0.027
final-loss gap), which is a statement about the configured stand-in synthesizer
rather than about the architecture. The runner's exit code reflects this: the
mechanism check and artifact reproducibility gate the run; G2 gates it only once
`backend` names a real model.

### Two things the run demonstrates on its own

**Internal computation does not substitute for grounding.** `b1` spends five
samples and moves accuracy by **+0.0058** while spending zero external bits.
The withheld variable is not in the prompt, so there is nothing internal to
converge on — the empirical content of `internal_reducibility = 0`, and the
reason "just sample more" is not an answer to a grounding deficit. The simulator
guesses with a *bias* rather than a coin flip precisely so this is visible;
self-consistency cannot correct a bias.

**Resolution accuracy is not decision accuracy.** A query can be perfect while
the model still misapplies the rule. Both are reported, and a test holds them
apart by giving the simulator a 20% error rate on items it has full information
for: resolution stays 1.0000 while accuracy falls.

### Swapping in a real model

The primary LM is replaceable from config (plan §12) — nothing above the
backend knows what produced the candidate action:

```yaml
backend: transformers
backend_options:
  checkpoint: google/gemma-3-1b-it   # 1B–3B instruct, per plan §12
  dtype: bfloat16
```

It drives the repository's existing `Backend` protocol, the same one the
M2-LLM experiment uses, so there is no second model-loading path. What is still
missing for a real run is an LM query synthesizer (`kind: lm` alongside `gold`
and `noisy`), which is the natural first piece of Phase 2 proper.

### Run it

```bash
make hg-v2-phase2
```

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
A real checkpoint does need the Spark, and this repository and the sibling
`research/` projects reach it by two different routes:

- **`hg` today**: an NGC container (`docker/Dockerfile`, base
  `nvcr.io/nvidia/pytorch`), pip-installing `requirements-gpu.txt`, entered
  through the `hg` CLI.
- **`research/gated-lm` and `research/snap-picker`**: direct `ssh notch@10.0.4.10`
  (GB10, behind the `Notch` VPN), work dir `~/marco/`, and **uv** with the
  CUDA-13 aarch64 wheels pinned per-platform:

  ```toml
  [[tool.uv.index]]
  name = "pytorch-cu130"
  url = "https://download.pytorch.org/whl/cu130"
  explicit = true

  [tool.uv.sources]
  torch = [{ index = "pytorch-cu130", marker = "platform_machine == 'aarch64' and sys_platform == 'linux'" }]
  ```

  bf16 autocast on CUDA; `uv run` resolves the right wheels automatically.

The second route is the one currently known to work on the box. **This is the
open decision blocking gate G2**: `hg`'s container path has not been exercised
on the GB10 in this session, and `pyproject.toml`'s `torch>=2.4` carries no
aarch64/cu130 pin, so a bare `uv pip install -e ".[gpu]"` on the Spark would
resolve against default PyPI wheels.

---

---

## Next

1. **An LM query synthesizer.** `hg.queries` has `gold` and `noisy`; a `kind: lm`
   synthesizer that prompts the primary model for a `QuerySpec` is what turns
   B4 into a real test of the architecture rather than of a stand-in.
2. **A frozen checkpoint, to evaluate gate G2.** Needs the Spark execution
   decision above.
3. Gate G3 onward (does the effect survive a real LM), G4 (does the grounding
   requirement change with model capability), and only then G5 (custom
   training) — per plan §27.
