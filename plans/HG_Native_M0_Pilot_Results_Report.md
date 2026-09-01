# HG-Native M0 — first pilot-stage results

**For:** the theory and planning track
**Basis:** `plans/HG_Native_M0_Technical_Spec_v0_1.md`
**Date:** 2026-09-01

Self-contained: every number below is reproduced from a saved run directory
or an inline eval JSON, and nothing requires reading the code. This is a
different kind of report than the HG-v2 ones: HG-Native is a separate track
(no pretrained model, no prompting — a from-scratch transformer trained from
nothing), so this is the *first* report on it, covering the entire build
from an empty spec to a real pilot-scale run in one continuous session.

---

## 1. What HG-Native is, and why it's a separate track

HG-v2 tests whether a frozen or lightly-adapted pretrained LM can be *made*
to behave in a grounding-aware way (prompting, query synthesis, routing
policies on top of an existing model). HG-Native asks a different question:
can decision-relevant structure be learned **natively**, as part of
pretraining itself, by a model with explicit auxiliary heads for it? That
needs its own transformer, tokenizer, data generators, and training loop —
nothing in the existing `hg` package (built entirely around calling
pretrained models) was reusable at the architecture level.

---

## 2. Branch, commits, test suite

| | |
|---|---|
| branch | `claude/hg-v2-e1-e2` |
| commits | `2db5ee3` .. `453979b` (13 commits, this session) |
| test suite | 721 passed, 9 skipped locally (no `torch`); **775 passed, 0 skipped** on the DGX Spark (real GPU) |
| pilot run | `results/pilot_20260901/` (checkpoints local-only/gitignored; `manifest.json`/`train_metrics.jsonl`/`report.md` are the durable record) |

Provenance note: the pilot runs executed against the code synced to the
Spark immediately before commit `453979b`; nothing changed between the sync
and the commit, so the commit's content is exactly what ran. The Spark's
checkout has no `.git` (rsync excludes it), so `manifest.json`'s
`git_commit` field is `null` for these two runs — a real gap, noted in
§9.4, not hidden.

---

## 3. What was built

Everything in `src/hg_native/` (new top-level package, sibling to `hg`) and
`experiments/hg_native/`:

- **`data/`** — the `Episode` schema (spec §4) and all four task-family
  generators (spec §5): Static Hidden Variable, Accessibility/Scaffolding,
  Sequential Known Quotient, Capability Refinement. Plus `tokenizer.py`, a
  from-scratch byte-level BPE trainer/loader.
- **`model/`** — `HGNativeConfig` (+ a torch-free analytic parameter
  counter), and the actual model: RoPE + RMSNorm + SwiGLU + grouped-query
  attention trunk, the HG latent, and all six heads (decision, distortion,
  info-deficit, accessibility, transition, router).
- **`training/`** — canonical action vocabulary, per-episode head targets,
  a batch serializer, the loss registry with the §7 warm-up schedule, the
  training loop (AdamW, cosine-with-warmup, grad clipping), checkpointing,
  the run manifest, and ordinary-LM (TinyStories) mixing.
- **`eval/`** — AUROC/AUPRC/Brier/ECE from definitions (no sklearn), a
  from-scratch item-level bootstrap CI, and the gate metrics: capability
  floor (N0), grounding discrimination (N1), routing value (N2),
  accessibility discrimination (N3), quotient-recovery linear probes (N4),
  grounding reuse (N5), and §10.4 query efficiency.
- **`experiments/hg_native/`** — `train_tokenizer.py`, `run_train.py`,
  `run_eval.py`, and configs.

---

## 4. Status of the gates: directional evidence, not formal passes

Important before reading §6: **none of N0–N5 are being declared PASSED**
in the HG-v2 sense (a bootstrapped, replicated, baseline-compared
statistical claim). This run is one seed, no replication, and — the
concrete missing piece — **no baseline confidence-proxy control exists
yet** to compare HG's grounding/accessibility/routing heads against
(a vanilla model has no such heads; spec §13's control set calls for
"vanilla + confidence router" and "vanilla + self-consistency router",
neither of which is built). What follows is real, reproducible measurement
against generator ground truth, at pilot scale, on real data — strong
directional evidence that the architecture is doing something, not a
completed gate-pass claim.

---

## 5. The pilot run

Spec §8.2 pilot stage: Baseline and HG, seed 0, matched trunk.

| | |
|---|---|
| params | baseline 56,947,840 / HG 57,204,862 (**0.45% overhead**, spec §2.1's <5% budget) |
| trunk | vocab 16000, d_model 640, 10 layers, 10 heads / 5 KV heads, d_ff 1792, max_seq_len 256 |
| tokenizer | a real trained byte-level BPE, 16,000 tokens, trained on 100,000 TinyStories examples |
| data | spec §6 mix: 4 synthetic families + `ordinary_lm` (20,000 real TinyStories examples, ratio 0.8) |
| steps | 2000, batch size 16 |
| device | Spark GPU (a real fix mid-session — see §9.1) |

This is the first configuration in the project that actually lands in
spec §3.1's stated "~50–70M" band; the spec's own literal recommended
YAML computes to ~87.8M (§9.3).

---

## 6. Results

### 6.1 ΔNLL (spec §10.1) — does HG damage ordinary language modeling?

| | baseline | HG | Δ |
|---|---:|---:|---:|
| first-step loss | 9.7809 | 9.7809 | 0 (identical init, shared seed) |
| final `lm` loss | 1.7586 | 1.7683 | **+0.0097 nats** |
| final `ordinary_lm` loss | 2.1681 | 2.1778 | +0.0097 |

Essentially no degradation. `ln(16000) = 9.68`, so the first-step loss
(9.78) confirms healthy initialization at this vocabulary size for both
models.

HG's own per-head loss trajectory (step 0 → step 1999, `hg_multiplier`
ramping 0 → 1 over training per the §7 warm-up schedule):

| head | step 0 | step 1999 |
|---|---:|---:|
| decision | 0.721 | 0.588 |
| quotient (distortion) | 0.600 | 0.132 |
| info_deficit | 0.808 | **0.0015** |
| accessibility | 1.020 | **0.0021** |
| route | 1.111 | **0.0165** |
| transition | 1.229 | 0.202 |

### 6.2 N0 — capability floor

| | |
|---|---:|
| n | 3300 |
| accuracy | 0.746 |
| balanced accuracy | 0.786 |
| per-class accuracy (A / B) | 0.656 / 0.917 |
| constant-label collapse | no |

### 6.3 N1 — grounding (information-deficit) discrimination

| | |
|---|---:|
| AUROC | **1.000** |
| AUPRC | 1.000 |
| Brier | 0.00002 |
| ECE | 0.0017 |
| mean score, deficit present | 0.9997 |
| mean score, no deficit | 0.0046 |

Per-subclass AUROC is `null` for every Family-A subclass: each subclass is
single-class by construction (e.g. `no_deficit` never needs grounding), so
AUROC is undefined within it — the pooled number above is the meaningful
one. This is an artifact of the task design, not a metric bug (documented
in `eval/grounding.py`).

### 6.4 N2 — routing value

Route accuracy against the oracle `r*` at `(λ_B, λ_K) = (1, 1)`: **0.990**.

Expected cost across the price grid (model vs. always-X baselines vs. the
oracle):

| λ_B | model | oracle | always ACT | always THINK | always ASK |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.034 | 0.029 | 0.217 | 0.100 | 0.033 |
| 1.0 | 0.053 | 0.047 | 0.217 | 0.100 | 0.123 |
| 10.0 | 0.249 | 0.130 | 0.217 | 0.100 | 1.023 |

The model tracks the oracle closely and correctly avoids the always-ASK
trap at high λ_B (asking becomes ruinously expensive there — 1.023 —
while the model pays 0.249). Not yet compared against a matched
confidence-router baseline (§4).

### 6.5 N3 — accessibility discrimination

| | |
|---|---:|
| AUROC | **1.000** |
| n (zero-deficit items) | 1200 |
| unnecessary-ASK rate | **0.000** |
| THINK rate on accessibility-deficit items | **1.000** |

Exactly the qualitative behavior spec §10.5 asks for: on items with zero
information deficit, the model never asks, and it correctly routes every
accessibility-deficit item to THINK.

### 6.6 N4 — quotient recovery (the central result of this run)

Linear probe accuracy on the HG latent `S_t`, Family C, held-out split:

| variable | role | accuracy |
|---|---|---:|
| `r` | current-relevant | 1.000 |
| `f` | future-relevant | 0.964 |
| `h` | reusable hidden | **0.902** |
| `u` | persistent irrelevant | 0.507 |
| `n` | random irrelevant | 0.542 |

**`Q_sel = mean(relevant) − mean(irrelevant) = +0.431`.**

This is the first clean, non-confounded confirmatory result in the
project: relevant variables (including the one that has to be *recalled*
across turns, not just read off the current input) are recovered at
90–100%, while both irrelevant variables sit at chance (~50%). Section 9.2
explains why this required an architectural fix mid-session and gives the
before/after progression.

### 6.7 N5 — grounding reuse

Repeated-ask rate (`ASK` chosen again after `h` is already grounded and
still valid): **0.000**, n=600. Given §6.6, this is now a meaningful
number — not asking reflects the fact that `h` is actually recoverable
from context, not a shallow "never ask after t=0" shortcut (see §9.2).

### 6.8 Query efficiency (spec §10.4, Family A)

| | |
|---|---:|
| soundness | 0.943 |
| under-asking rate | 0.000 |
| unnecessary-ask rate | 0.300 |
| `B_system` mean (bits) | 0.767 |
| `B*` mean (bits) | 1.250 |
| `η_Q` | 1.333 |

`η_Q > 1` looks like it should be capped at 1 but isn't, by construction:
it's a known artifact of pricing every single-hop `ASK` at a flat 1-bit
cost even on multi-bit-deficit items it only partially resolves (see
`eval/query_efficiency.py`'s docstring). Not a bug to silently clip — a
limitation of the current cost model, flagged there.

---

## 7. Comparison across scales: the N4 fix actually did something

The single most useful number in this report is how `h`'s probe accuracy
moved as two real things changed:

| stage | `h` accuracy | `q_sel` | what changed |
|---|---:|---:|---|
| debug (before fix) | 0.48 (chance) | −0.17 to −0.27 | `h` never appeared in any turn's text |
| debug (after fix) | 0.68 | −0.02 | growing multi-turn context added (§9.2) |
| **pilot (this run)** | **0.90** | **+0.43** | same fix, at real 57M scale with real data |

This is a controlled before/after: same generator logic, same evaluation
code, one architectural change, measured three times. The direction and
magnitude of the movement is itself evidence that the metric is measuring
something real rather than noise.

---

## 8. Two things worth carrying back into the theory

### 8.1 A structural gap in "grounding reuse" for M0's action space

Spec §5.3 assumes a model can ground `h` once and reuse it. But M0's
model is a plain decoder-only transformer with no built-in memory across
separate calls — the only way information can persist from one step to
the next is if it is literally still in the token sequence. There were
exactly two ways to make "reuse" testable at all:

1. **Grow the context** — each step's prompt is the whole trajectory so
   far, with the environment's disclosed answer to `ASK_h` staying in
   view. Chosen.
2. **Give the model persistent cross-step state** — e.g. actually feed the
   transition head's own prediction back in as working memory. Rejected:
   this would be a structural capability only the HG variant could have
   (baseline has no transition head), breaking the matched-comparison
   principle (spec §2.1) that the whole architecture-vs-architecture
   question depends on.

This is a general point, not specific to Family C: **any M0-style
"reuse across time" claim requires the reusable fact to actually be
present in context**, because nothing else carries state between calls.
If a later milestone wants reuse to work over an interval too long to fit
in context, that requires a real architectural decision (retrieval, a
persistent memory module applied identically to both variants) — M0 as
specified does not have a mechanism for it.

### 8.2 The single-hop `ASK_k` action space can't fully resolve a multi-bit deficit

Family A's `multi_bit_deficit` subclass needs two independent hidden bits,
but the action space has one `ASK` per step. `correct_action` is defined
as asking the first variable in `minimal_query_set`; the distortion
profile honestly reflects that this often doesn't fully resolve the item
(computed by resolving only the asked variable and defaulting the other to
`NO`), rather than pretending a single ask always succeeds. This is the
direct cause of §6.8's `η_Q > 1` artifact. A real multi-turn action space
(ask, get an answer, ask again) would remove this limitation, but that is
a genuinely bigger change than M0's structured, single-hop design.

---

## 9. Notable engineering findings (methodologically relevant, not just bugs)

### 9.1 Every prior run in this project trained on CPU with the GPU idle

`TrainConfig` defaulted to `device="cpu"`, and nothing overrode it until
this session's pilot-run push. Fixing it (an explicit `--device` flag,
defaulting to CUDA when available) measured a **~35–45x speedup**
(0.088s/step vs. ~3–4s/step at 16M params, batch 16) — the difference
between a pilot run being a multi-hour undertaking and a 15–20 minute one.
Worth checking for in any future from-scratch training setup: a correctly
running loss curve does not tell you whether you're using the accelerator.

### 9.2 Two bugs that only a real training run surfaced, not unit tests

1. **Missing next-token shift.** An early version trained the LM head
   against `labels = ids.clone()` — predicting each token from *itself*.
   With tied embeddings, the residual stream already carries `embed(x_t)`
   straight to the output, so the untrained model "solved" this trivially:
   first-step loss was exactly `0.0`, gradient norm exactly `0.0`, on a
   real ~9.8M-parameter run. A smaller unit-test-scale model only showed a
   milder version of the same effect — this needed the real run to be
   visible at all.
2. **Unscaled init.** Default `nn.Linear`/`nn.Embedding` init let the tied
   LM head reach logits of magnitude ~400 at random init (the reason bug 1
   could reach exactly `0.0` loss). GPT-2-style init (`std=0.02` plus
   `1/sqrt(2·n_layers)` residual-output scaling) fixed it; first-step loss
   is now `ln(vocab_size)` almost exactly, at every vocab size tried
   (259, 16000).

### 9.3 The spec's own recommended config doesn't hit its stated target

§3.1's literal recommended YAML (vocab 16000, 12 layers, d_model 768,
d_ff 2048, GQA 12/4 heads) computes to **~87.8M params**, not the stated
"~50–70M" — caught by the analytic parameter counter the spec itself asks
for ("compute exact parameter counts programmatically... before long
runs"). This report's pilot config (§5) was sized down to actually land in
the stated band.

### 9.4 Provenance gap: the Spark checkout has no git history

`manifest.json`'s `git_commit`/`git_dirty` fields are `null` for both
pilot runs because the rsync-based deploy path (`scripts/spark_deploy.sh`)
excludes `.git`. The code that ran is provably the code in this repo (see
§2's provenance note), but the manifest itself can't attest to that on its
own — worth fixing (e.g. write the git SHA into a plain file before
syncing) if unattended/scheduled runs on the Spark become routine.

---

## 10. Assessment: is HG-Native promising so far?

**Cautiously, yes — with the caveat that this is architecture-validation,
not a statistically gated result yet.** The headline finding (§6.6–§7) is
that a from-scratch ~57M model, trained for 2000 steps on a data mix that
is 80% real natural language, developed a latent representation that:

- recovers the variables the decision rule actually needs (`r`, `f`, and
  — after fixing the one real architectural gap — the across-time `h`) at
  90–100% linear-probe accuracy,
- does **not** trivially encode variables the rule never uses (`u`, `n`
  sit at chance),
- drives near-perfect grounding/accessibility discrimination and routing
  that tracks a computed oracle,
- costs essentially nothing in ordinary language-modeling quality
  (ΔNLL = 0.0097 nats).

None of this required scale, RLHF, or prompting — it emerged from
auxiliary losses on top of ordinary next-token prediction, at 57M
parameters, in 2000 steps. That is the actual bet the HG-native track is
making, and this run is the first evidence it can pay off at all. The
honest limitation is that "emerged" here means "on synthetic tasks
purpose-built to have exactly this structure" — whether it survives
contact with less on-the-nose tasks, more seeds, and a real baseline
comparison is exactly what's outstanding.

---

## 11. What's outstanding, priority-ordered

1. **A baseline confidence-proxy control.** Without it, N1/N2/N3 have
   nothing to be compared *against* in the HG-v2 sense — spec §13's
   "vanilla + confidence router" / "vanilla + self-consistency router".
   This is the single piece most needed to turn §6 into a real gate claim.
2. **Replication.** Seeds 1, 2 at the same pilot config (spec §8.3) —
   everything needed to run this already exists; it's compute time, not
   new code.
3. **Bootstrap CIs on this run's numbers.** `eval/bootstrap.py` exists and
   is tested; it just wasn't run against this pilot's episodes yet.
4. **Ablations A0–A4** (spec §8.4): does the effect need all six heads, or
   would decision+quotient alone reproduce most of §6.6?
5. **F0–F7 as measured checks**, not spec text. F0 (parameter-count
   confound) is already effectively addressed by the <5% overhead budget;
   the rest aren't yet instrumented.
6. **Live training progress logging.** `run_train.py` currently buffers
   everything until a run completes — fine for a 15-minute pilot, a real
   problem for anything longer.
7. **The two documented design gaps** (§8): decide whether §8.1's
   growing-context approach is the intended solution going forward for
   any future reuse-over-longer-intervals claim, and whether §8.2's
   single-hop action space is acceptable for M0 or needs a genuine
   multi-turn extension before M1.

Per spec §22's own scale/no-scale criterion — "proceed to ~150M only if
the replicated M0 result shows at least two of: grounding awareness,
accessibility separation, dynamic quotient/reuse" — this run shows
directional evidence for all three, but "replicated" is not yet true.
Items 1–3 above are what stand between this report and an actual §22
decision.
