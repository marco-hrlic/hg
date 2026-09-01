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

Same index problem, but the AI also holds `m = 5` internal observations of
`X_K`. On 40% of items they are noiseless — `J` is directly observable from
`M_A` and asking is waste (the plan's "with probability p, J is directly
observable"). On the rest each observation is independently flipped with rate
0.25 or 0.45, so the AI's majority vote is unreliable and a 1-bit targeted ask
resolves the decision.

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
| 0.02 | 0.1600 | 0.0200 | **0.0170** | 0.0032 | ✓ |
| 0.05 | 0.1600 | 0.0500 | **0.0323** | 0.0080 | ✓ |
| 0.10 | 0.1600 | 0.1000 | **0.0577** | 0.0160 | ✓ |
| 0.15 | 0.1600 | 0.1500 | **0.0831** | 0.0240 | ✓ |
| 0.20 | 0.1600 | 0.2000 | **0.1030** | 0.0320 | ✓ |
| 0.30 | 0.1600 | 0.3000 | **0.1324** | 0.0480 | ✓ |
| 0.50 | 0.1600 | 0.5000 | 0.1600 | 0.0800 | ✗ (ties always_act) |

**Gate G1 passed at 6 of 7 prices**, each with an item-paired bootstrap interval
on the cost advantage that excludes zero. At λ = 0.50 no estimated error
probability exceeds the price of a bit, so the router correctly collapses to
always-ACT and ties it — a tie is not counted as a win.

The query rate falls monotonically as external bits get dearer (1.00 → 0.51 →
0.29 → 0.00), which is the sanity check on the whole cost model.

### What Phase 1 does *not* show

Two limits, both recorded in the run's own report rather than left for a reader
to discover:

1. **The learned router matches `bayes_optimal` exactly at every λ.** That
   reference routes on the exact posterior `P(wrong | observations)`,
   marginalizing the unknown noise level — the ceiling for any router built on
   these features. Matching it is expected here: with `m = 5` the agreement
   features take only **three** distinct values, and both policies are monotone
   thresholds on that ordering, so there are few places for a threshold to
   land. `diagnostics.json` prints the whole feature space with both estimates
   and the realized error rate per level. Consequence: Phase 1 establishes that
   routing has value, but this task is too coarse to discriminate a good
   estimator from a perfect one.
2. **The query is already optimal.** Phase 1 routes between acting and asking a
   query that was handed to it. Gate G2 — whether a *synthesized* query
   compresses external cost — is untouched.

A `validity` block in `analysis.json` checks the one outcome that would
invalidate rather than fail the run: the learned router coming out *cheaper*
than its own ceiling, which would mean information leaked into its features.

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
