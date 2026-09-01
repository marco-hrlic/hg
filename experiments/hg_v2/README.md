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
| G2 | Does targeted query synthesis compress external cost? | not started |
| G3 | Does the effect survive a frozen off-the-shelf LM? | not started |
| G4 | Does the grounding requirement change with model capability? | not started |
| G5 | Is custom training justified? | not started |

Phases 0 and 1 are complete. **Phase 2 has not been started**: it needs a
frozen LM, and the decision about how this repository reaches the DGX Spark is
still open (see below).

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

## Reading these results honestly

Neither phase involves a model, and both reports say so on their own first page.

Phase 0's index theorem is trivial mathematically. Reproducing it proves exactly
one thing: that the abstractions — AI private state, external private state,
decision variable, query, response cost, final decision, episode artifact,
analysis pipeline — represent the grounding problem correctly. **No Phase 0
number is evidence about any model's behaviour**, and the report says so on its
own first page.

---

## Phase 2 and the DGX Spark

Phase 0 and Phase 1 need no GPU. When Phase 2 introduces a frozen primary LM,
note that this repository and the sibling `research/` projects reach the Spark
by two different routes:

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

The second route is the one that is currently known to work on the box. Before
Phase 2 spends GPU time, decide which one HG-v2 uses — `hg`'s container path
has not been exercised on the GB10 in this session, and `pyproject.toml`'s
`torch>=2.4` carries no aarch64/cu130 pin, so a bare `uv pip install -e ".[gpu]"`
on the Spark would resolve against default PyPI wheels.

---

---

## Next: Phase 2 (not started)

A frozen off-the-shelf instruct model at 1–3B produces the AI's candidate
action, on controlled natural-language tasks where one decision-relevant
variable is withheld and the optimal query is known (plan §12–17). Gate G2 is
the first real HG-v2 architectural result: at equal final distortion, does a
synthesized targeted query cost fewer external bits than generic escalation?

Blocked on nothing in this codebase, but see the Spark section above — decide
the execution path before spending GPU time.
