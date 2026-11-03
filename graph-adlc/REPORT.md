# Graph-Driven ADLC v0.1: Implementation Report

**Date:** 2026-11-03
**Branch:** `ccr-26a87342-teewk8` (repo `marco-hrlic/hg`, folder `graph-adlc/`)
**Source handoff:** `graph_adlc_v0_1_handoff.zip`, kept in [`docs/`](docs/)
**Status:** v0.1 done. All 14 release-blocking acceptance tests pass, the vertical slice replays from the CLI, and every graph change has a readable diff and an explanation of its effects.

---

## 1. Executive summary

The handoff asked for a local proof of concept testing one thesis: *a persistent, incrementally refined specification graph can drive implementation readiness, scope-change impact, and spec/code/evidence reconciliation.*

This prototype implements that thesis end to end. It uses only the Python standard library plus SQLite, with pytest for tests. There is no graph database, LLM, web server or frontend. Impact results are produced by a fixed rule table over registered edges, with no hidden reasoning.

The critical scenario from the spec runs from a single command and produces an auditable trace:

> partial discovery → local readiness → implement → new provider constraint → explainable influence → stale evidence → reconciliation

| Metric | Value |
|---|---|
| Acceptance tests (T01–T14) | **14 / 14 pass** |
| Total tests | **44 pass** (0.8 s) |
| Package source | ~2,900 lines across 14 modules |
| Test source | ~660 lines across 7 files |
| Runtime dependencies | none (stdlib only) |
| CLI commands | `init, status, frontier, context, impact, propose, commit, next, export, log, work {create,start,submit,show,list}, evidence {add,list}, reconcile` |

**Main findings**

1. **Local readiness works as intended.** A blocked requirement (`REQ-RETRY`) never blocks unrelated ready work (`REQ-CHARGE`, `REQ-REPORT`), and there is no global project phase.
2. **Directional, aspect-sensitive influence gives explainable, bounded impact.** The provider contract change reaches 6 obligations on 5 nodes, each with an explanation path. The unrelated `REQ-REPORT`, which shares the same goal root, is untouched.
3. **Readiness and freshness are separate dimensions.** After the contract change, `REQ-RETRY` is still `ready` in spec terms, but its implementation freshness is `needs_revalidation`.
4. **Stale work can't be silently accepted.** Re-submitting it is refused, and verifying it requires an explicit rebase plus fresh test *and* contract evidence.
5. **The main cost is noise.** The rules are conservative, so a plain spec change to a requirement provisionally flags consumers of its implementing component. That's intended by the spec, and measuring it is the stated next experiment (§10).

---

## 2. Scope delivered vs. requested

| Handoff requirement | Delivered | Where |
|---|---|---|
| Python 3.12+, `sqlite3`, pytest; no graph DB / LLM / server / frontend | ✅ | `pyproject.toml` (no dependencies) |
| Typed graph validation | ✅ closed enums, per-kind payload schemas, endpoint rules, single-parent acyclic refinement forest | `models.py`, `validation.py` |
| Atomic, versioned, revision-guarded mutations | ✅ strict base-revision check + per-entity `expected_rev`, one `BEGIN IMMEDIATE` transaction | `delta.py`, `core.commit` |
| `propose` read-only, `commit` atomic and auditable | ✅ verified by tests (database and event log unchanged by `propose`) | `core.propose`, `changes` and `events` tables |
| Readiness (`blocked` / `conditional` / `ready`) with structured reasons | ✅ | `readiness.py` |
| Influence: directional rules, old∪new edge union, paths, terminals, cycles | ✅ | `influence.py` |
| Influence = candidate obligations, no silent suppression of hypothesized edges | ✅ hypothesized edges add `investigate_unknown` and never prune; every report carries a completeness disclaimer | `influence.py` |
| Bounded context packets | ✅ with an inclusion rule per node | `context.py` |
| Work units, baselines, evidence freshness, reconciliation | ✅ | `lifecycle.py` |
| Deterministic next-action policy | ✅ tiered, configurable weights | `scheduler.py` |
| CLI per SPEC §5 | ✅ plus `work`, `evidence`, `reconcile`, `log` | `cli.py` |
| 14 acceptance tests | ✅ plus 30 supporting tests | `tests/` |
| Single-script replay with machine-readable trace | ✅ `python -m graph_adlc.demo` → `trace.json` and `snapshot.json`; `scripts/demo.sh` | `demo.py` |
| README with copy-paste trace sequence | ✅ | `README.md` |

**Added beyond the handoff:** two scenario fixtures (`idempotency_window_change.json` for step D and `retry_window_reconciliation.json` for step F), an optional approval policy, an `events` audit log, `gadlc log`, and canonical export/import round-tripping.

---

## 3. Architecture

```text
            ┌──────────── cli.py (argparse) ─────────────┐   demo.py (replay + trace.json)
            │                                            │
            ▼                                            ▼
                        core.py  — programmatic API (GraphADLC)
   ┌───────────┬─────────────┬──────────────┬─────────────┬──────────────┬──────────────┐
   ▼           ▼             ▼              ▼             ▼              ▼              ▼
delta.py  validation.py  readiness.py  influence.py   context.py   lifecycle.py   scheduler.py
(staged   (invariants,   (leaf state,  (rule table,  (bounded     (work units,   (ranked
 ops)      aspects)       reasons)      BFS, paths,   packets)     freshness,     actions)
                                        pruning)                   reconcile)
   └───────────────────────────────┬──────────────────────────────────────────────────────┘
                                   ▼
                     models.py (records, enums)   store.py (SQLite schema, transactions)
```

**Design principle:** every operation loads an in-memory `Graph` (nodes and edges, including tombstones), derives results from it, and writes back only in a single transaction. All derived state (readiness, freshness, impact, context, schedule) is recomputed, never stored as a status, except for the explicit, auditable `needs_revalidation` / `invalidated_by` markers that SPEC §2 requires.

### 3.1 Module inventory

| Module | Lines | Responsibility |
|---|---:|---|
| `influence.py` | 418 | Rule table, `UnionGraph` (old∪new edges), BFS worklist, explanation paths, evidence-gated pruning |
| `cli.py` | 406 | `gadlc` command set, human and `--json` renderers |
| `lifecycle.py` | 360 | Work-unit state machine, baselines, required checks, evidence staleness, commit-time invalidation, reconcile/rebase |
| `models.py` | 287 | Enums (`NodeKind`, `Relation`, `Aspect`, `Obligation`, `Readiness`, `WorkStatus`, `Freshness`) and dataclasses |
| `core.py` | 260 | Public API, `propose` / `commit` pipeline, status |
| `validation.py` | 254 | Node payload schemas, endpoint-kind rules, forest/acyclicity, aspect classification |
| `delta.py` | 253 | Applies `add/update/remove_node`, `add/remove_edge` to an isolated graph copy; seeds; field-level diffs |
| `store.py` | 249 | SQLite schema, `BEGIN IMMEDIATE` transactions, change log, events, work units, evidence |
| `demo.py` | 122 | 22-step vertical-slice replay and machine-readable trace |
| `readiness.py` | 105 | Leaf readiness with structured blockers and conditions, frontier, readiness diff |
| `scheduler.py` | 95 | Tiered, transparent action ranking |
| `context.py` | 87 | Bounded context packet and text view |

### 3.2 Storage schema

The tables of SPEC §5, plus these additions:

| Table | Notes |
|---|---|
| `nodes`, `edges` | Soft-delete (`deleted`); IDs never reused; edge foreign keys to nodes |
| `changes` | `CHG-<graph_rev>`; base revision, actor, rationale, operations, **impact**, **diff** (added) |
| `work_units` | `WORK-<n>`; targets, baseline (revisions, assumptions, required checks, `evidence_after_seq`), context IDs, status, code revision, **history** |
| `evidence` | `EV-<n>`; spec revisions recorded at capture time, code revision, **`invalidated_by`** (change ID), **`seq`** for ordering |
| `events` | Append-only audit log: seeded, committed, work-unit transitions, evidence, reconciliations |
| `meta` | `graph_revision`, ID counters, optional `approval_required_aspects`, `scheduler_weights` |

---

## 4. Semantics implemented

### 4.1 Readiness (SPEC §1.3)

Only **leaf requirements** are candidates.

| State | Triggered by |
|---|---|
| `blocked` | open blocking question via `blocks`; empty acceptance criteria; required contract `incompatible` / `unresolved` / `rejected`; assumption `proposed`, `rejected` or `invalidated`; non-leaf |
| `conditional` | accepted tracked assumption; hypothesized `constrained_by` / `depends_on` edge; required contract `draft` / `proposed` / `deprecated` |
| `ready` | none of the above |

Results carry structured `blockers` and `conditions`, each with a code, a message and the node and edge IDs involved. A separate **freshness** dimension (`unimplemented`, `in_progress`, `current`, `needs_revalidation`) comes from work units and evidence and never feeds back into readiness.

### 4.2 Change aspects (SPEC §2)

A declared aspect is validated against the node kind and the patched fields. If it doesn't fit, it is **upgraded conservatively** and the upgrade is reported in `aspect_overrides`:

| Node kind | Accepted as declared | Upgraded |
|---|---|---|
| contract | `contract` | `semantic` → `{contract, semantic}`; anything else → `contract` |
| component, artifact | `semantic`, `implementation` | others → `semantic` |
| test, artifact | `evidence` **only** for bookkeeping keys (`last_result`, `last_run_at`, `evidence_ref`, `notes`) | otherwise → `semantic` / `implementation` |
| goal, capability, requirement, question | `semantic` | others → `semantic` |
| any `update_node` | — | `structure` → `semantic` (structure is reserved for adding or removing nodes and edges) |

This makes the spec's "do not accept an arbitrary `evidence` classification to suppress impact" enforceable, and a test covers it.

### 4.3 Influence rule table (SPEC §3.2)

The engine runs a BFS over the **union of old and proposed edges**. Visited keys are `(node, aspect, obligation, terminal)`. It never bounces back along the edge it arrived on. For each `(target, obligation)` it keeps the first (shortest) path.

Only *origin* events (real changes) fan out across relation types. Derived `review_spec` continues downward only.

| Event at N | Edge (from N's perspective) | Obligation on other endpoint | Terminal |
|---|---|---|---|
| semantic | `refines` N → child | `review_spec` (recursive) | no |
| semantic (origin) | `refines` parent → N | `review_spec` on parent | **yes** (no sibling fan-out) |
| semantic (origin) | `blocks` question N → target | `review_spec` | yes |
| semantic (origin) | `implements` component → N | `review_implementation` | no |
| semantic (origin) | `verifies` test → N | `rerun_test` | yes |
| semantic (origin) | `depends_on` consumer → N | `check_contract` | yes |
| contract (origin) | `constrained_by` subject → N | `check_contract` (one more hop) | no |
| contract (origin) | `verifies` test → N | `rerun_test` | yes |
| contract (subject) | `depends_on` consumer → subject | `check_contract` | yes |
| contract (subject requirement) | `verifies` / `implements` | `rerun_test` / `review_implementation` | yes / no |
| implementation | `realized_by` component → artifact | `review_implementation` | no |
| implementation | `implements` component → requirement | `check_contract` | yes |
| implementation | `depends_on` consumer → component | `check_contract` **provisional** unless pruned | yes |
| structure (edge add/remove) | both endpoints, per relation | e.g. `depends_on`: consumer `check_contract`, provider `review_spec` | yes |
| any | path crosses a **hypothesized** edge | + `investigate_unknown`, confidence `low`, never pruned | yes |

Each `ImpactRecord` has the shape `{target_id, obligation, origin_id, path[edge ids], reason, confidence, change_ref, provisional}`. After a commit, `change_ref` becomes `CHG-n/op[i]`.

### 4.4 Pruning policy (SPEC §3.2)

An implementation-only change stops at a provider boundary **only if** the proposal's `evidence_refs` point to a `contract_preservation` evidence record that:

- is about that provider and has result `passed`;
- has not been invalidated;
- records contract revisions equal to the current ones;
- covers contracts that are not changed in the same delta.

A bare `"contract_preserved": true` claim is ignored and noted in the report. Hypothesized edges are never pruned. A pruned boundary is reported (`pruned[]`, with the evidence ID), not dropped silently.

### 4.5 Transaction pipeline (SPEC §2)

1. `BEGIN IMMEDIATE`, then re-read the graph inside the transaction.
2. Strict `base_graph_revision` check and per-op `expected_rev` check. Either failing raises `RevisionConflict`.
3. Apply to an isolated copy, then validate schemas, references, endpoint kinds, single parent and acyclicity. A failure raises `ValidationError`.
4. Optional approval gate.
5. Influence over old∪new edges.
6. Write the changed nodes and edges, then `graph_revision += 1`.
7. Mark stale work and evidence (§4.6).
8. Insert a `changes` row with diff and impact, and an `events` row. Commit.

Any failure rolls back everything. `propose` runs steps 2–5 and a dry run of step 7 without writing.

### 4.6 Lifecycle and evidence freshness (SPEC §4, R6)

- **Work-unit baseline:** `base_graph_revision`, the revisions of the target, its contracts, its providers and their contracts, and its verifying tests. It also records assumptions, each with its originating node, node revision and graph revision, plus acceptance criteria, required checks, and `evidence_after_seq`.
- **State machine:** `proposed → active → submitted → verified`, plus `blocked`, `needs_revalidation` and `abandoned`. Disallowed transitions raise an error.
- **Verification:** a unit becomes `verified` only if every baseline revision is current *and* every required check has passing, current evidence recorded after the latest baseline.
- **Invalidation on commit:** evidence is invalidated, with `invalidated_by = CHG-n`, if any of its recorded revisions changed or its node received `check_contract`, `rerun_test` or `investigate_unknown`. A work unit becomes `needs_revalidation` on the same conditions applied to its baseline or targets.
- **Reconcile:** reports baseline mismatches, stale evidence and the obligations raised since the baseline. `--rebase` takes the current revisions as the new baseline, adds a `contract_check` for each contract that changed, and resets `evidence_after_seq` so old evidence can't satisfy the new checks.
- **Submitting stale work** (`needs_revalidation`) is refused, with exit code 5 from the CLI.

### 4.7 Scheduler (SPEC §4)

The scheduler returns a ranked, transparent set of actions. The score components are printed with each one.

| Tier | Action | Score basis |
|---|---|---|
| 1 | `reconcile` | 4000 + 10 × (baseline mismatches + stale evidence) |
| 2 | `resolve_question` | 3000 + 10 × leaf requirements unlocked |
| 3 | `implement` | 2000 − criteria count − 5 × uncertain dependencies − 50 if conditional |
| 4 | `refine` | requirements missing criteria; capabilities or goals with no refinements |

Weights can be overridden through the meta key `scheduler_weights`.

### 4.8 Context packets (SPEC §4)

`context(REQ)` contains the target, its refinement ancestors, constraining contracts, providers and their contracts, implementing components (and those components' providers), verifying tests, and blocking questions. Each node records `included_because`. Only edges between included nodes are listed. For `REQ-RETRY` the packet holds 7 nodes and excludes `REQ-REPORT`, `CMP-REPORTING`, `CAP-REPORTING`, `REQ-CHARGE` and `ART-PAYMENT`.

---

## 5. Vertical-slice demonstration (SPEC §6)

Run with `python -m graph_adlc.demo --workdir out`, which executes 22 CLI steps. Below is the trace, condensed from actual output.

### A. Discovery: graph revision 1

```text
Frontier @ graph rev 1
  REQ-CHARGE@r1      READY
  REQ-REPORT@r1      READY
  REQ-RETRY@r1       BLOCKED       open_blocking_question(Q-RETRY-BOUNDS, E15); missing_acceptance_criteria(REQ-RETRY)

Suggested actions:
   1. [tier 2] resolve_question  Q-RETRY-BOUNDS,REQ-RETRY  score=3010.0  currently blocks 1 leaf requirement(s)
   2. [tier 3] implement         REQ-REPORT                score=1999.0
   3. [tier 3] implement         REQ-CHARGE                score=1998.0
```

### B. Refinement: propose (read-only), then commit → revision 2

```text
Impact: 6 candidate verification obligation(s)
  CAP-PAYMENTS   review_spec            origin=REQ-RETRY      path=E04       (parent consistency, terminal)
  CMP-PAYMENT    review_implementation  origin=REQ-RETRY      path=E09
  CMP-REPORTING  check_contract         origin=REQ-RETRY      path=E09>E13   [provisional]
  REQ-CHARGE     check_contract         origin=REQ-RETRY      path=E09>E08
  REQ-RETRY      review_spec            origin=Q-RETRY-BOUNDS path=E15
  TEST-RETRY     rerun_test             origin=REQ-RETRY      path=E10
Readiness changes: REQ-RETRY: blocked -> ready
Committed CHG-2 -> graph rev 2
```

Only `REQ-RETRY` and `Q-RETRY-BOUNDS` changed revision. `REQ-REPORT` and `REQ-CHARGE` stay at r1.

### C. Implementation

```text
Work unit WORK-1 [implement] targets=['REQ-RETRY'] status=proposed
  baseline graph rev 2; node revisions CON-IDEMPOTENCY@r1, REQ-RETRY@r2, TEST-RETRY@r1
  required checks: test_result:TEST-RETRY
... submit --code-rev git:a1b2c3 --evidence TEST-RETRY:test_result:passed
Work unit WORK-1 status=verified   (satisfied test_result:TEST-RETRY by EV-1)
```

### D/E. Contradiction and influence → revision 3

The provider only honours idempotency keys for 24h, so `CON-IDEMPOTENCY` changes from r1 to r2 with aspect `contract`.

```text
Impact: 6 candidate verification obligation(s)
  CMP-PAYMENT    check_contract         path=E16
  CMP-PAYMENT    review_implementation  path=E06>E08
  CMP-REPORTING  check_contract         path=E16>E13     (provider-dependent consumer)
  REQ-CHARGE     check_contract         path=E06
  REQ-RETRY      check_contract         path=E07
  TEST-RETRY     rerun_test             path=E11
Readiness changes: (none)
Invalidated:
  work WORK-1 (verified) -> needs_revalidation: CON-IDEMPOTENCY r1 -> r2; REQ-RETRY received a verification obligation
  evidence EV-1 on TEST-RETRY stale: CON-IDEMPOTENCY recorded at r1, now r2; TEST-RETRY received a verification obligation

Suggested actions @ graph rev 3:
   1. [tier 1] reconcile  WORK-1,REQ-RETRY  score=4020.0
   2. [tier 3] implement  REQ-REPORT
   3. [tier 3] implement  REQ-CHARGE

$ gadlc work submit WORK-1 ... --evidence TEST-RETRY
REFUSED: WORK-1 needs revalidation against the current graph; run reconcile (and rebase) first
```

`REQ-REPORT` gets no obligation, even though it shares the root `GOAL-CHECKOUT`.

### F. Reconciliation → revision 4

```text
Reconcile WORK-1: needs_revalidation -> needs_revalidation (inspect)
  baseline mismatch CON-IDEMPOTENCY: r1 -> r2
  next: rebase; record contract_check for CON-IDEMPOTENCY at r2; record fresh test_result for TEST-RETRY
Committed CHG-4 -> graph rev 4  (REQ-RETRY r2 -> r3: adds "No retry after the 24h window")
Reconcile WORK-1 --rebase: needs_revalidation -> active
  new baseline rev 4: CON-IDEMPOTENCY@r2, REQ-RETRY@r3, TEST-RETRY@r1
... submit --code-rev git:d4e5f6 --evidence TEST-RETRY:test_result:passed --evidence CON-IDEMPOTENCY:contract_check:passed
Work unit WORK-1 status=verified  (EV-2, EV-3)

Leaf requirements (readiness | freshness):
  REQ-CHARGE@r1  ready | unimplemented
  REQ-REPORT@r1  ready | unimplemented
  REQ-RETRY@r3   ready | current
```

This matches all 8 steps of the expected demonstration trace in `ACCEPTANCE.md`. `test_demo.py` asserts it automatically.

---

## 6. Acceptance results

| ID | Requirement | Test | Result |
|---|---|---|---|
| T01 | Partial seed imports; no global-completeness assumption | `test_T01_partial_seed_imports_all_nodes_and_edges` | ✅ |
| T02 | REQ-CHARGE and REQ-REPORT ready; REQ-RETRY blocked | `test_T02_local_readiness` | ✅ |
| T03 | Accepted assumption gives conditional; assumptions copied into the work unit with origin revisions | `test_T03_conditional_with_tracked_assumption_copied_to_work_unit`, `test_T03b_hypothesized_dependency_is_conditional_not_ready` | ✅ |
| T04 | Refinement makes REQ-RETRY ready; REQ-REPORT revision unchanged | `test_T04_refinement_unblocks_retry_without_touching_unrelated_revisions` | ✅ |
| T05 | Semantic change to a consumer doesn't invalidate its provider | `test_T05_semantic_change_to_consumer_does_not_invalidate_provider` | ✅ |
| T06 | Contract change reaches consumers and tests, with paths | `test_T06_contract_change_reaches_consumers_and_tests_with_paths` | ✅ |
| T07 | Pruning with evidence; no pruning without it (or with a stale or failed result, or a bare claim) | `test_T07_*`, `test_T07b_*`, `test_T07c_*` | ✅ |
| T08 | Removed `depends_on` edge surfaces both endpoints | `test_T08_removed_edge_surfaces_both_endpoints` | ✅ |
| T09 | Reverse dependency cycle: terminates, unique obligations | `test_T09_cycle_terminates_with_unique_obligations` | ✅ |
| T10 | Second proposal from the same base rejected, no partial changes | `test_T10_second_proposal_from_same_base_rejected_atomically` | ✅ |
| T11 | Evidence at contract rev N becomes needs_revalidation at N+1 | `test_T11_evidence_freshness_after_contract_revision` | ✅ |
| T12 | Every obligation has origin, reason, path and change reference | `test_T12_every_obligation_has_origin_reason_and_path` | ✅ |
| T13 | Retry context includes ancestors, contracts, component and test; excludes reporting | `test_T13_context_is_bounded` | ✅ |
| T14 | Stale work can't reach verified without reconciliation | `test_T14_stale_work_cannot_be_verified_without_reconciliation` | ✅ |

**Supporting tests (30):**
- export/import canonical round-trip;
- invalid seed endpoints;
- six invalid-delta classes (second parent, refinement cycle, dangling reference, wrong edge direction, immutable field, payload schema);
- tombstoning with cascade and no ID reuse;
- the `evidence` aspect can't suppress impact;
- aspect classification table;
- unaccepted assumption blocks;
- contract status drives readiness locally;
- downward recursion and upward terminal propagation;
- `propose` is read-only;
- stale `expected_rev` rejected;
- atomic rollback on a mid-delta invariant failure;
- commit audit trail;
- approval policy;
- blocked requirement can't get a work unit;
- baseline contents;
- required checks gate verification;
- propose previews invalidation without applying it;
- old evidence doesn't count after rebase;
- scheduler ordering (reconcile first; unblock before implement; smaller scope first);
- full demo trace.

---

## 7. Traceability to SPEC design requirements

| Req | Statement | How it's met |
|---|---|---|
| R1 | Seed with incomplete and unresolved regions | Payload schemas permit empty criteria and open questions; T01 |
| R2 | Addressable entities for all node kinds | 8 closed node kinds, stable string IDs, tombstones |
| R3 | Local implementability, no global phase | `readiness.py`; T02 and T04; `status` has no phase field |
| R4 | Atomic, optimistic-concurrency deltas with reason and provenance | `core.commit`; actor and rationale in `changes`; T10 |
| R5 | Explainable directional influence, including deleted edges | `UnionGraph` and rule table; T05, T06, T08, T12 |
| R6 | Never silently treat superseded work or evidence as current | `mark_stale_after_change`, derived freshness check; T11 and T14 |
| R7 | Work packets from ready leaves with scoped context | `create_work_unit` and `context_packet`; T03 and T13 |
| R8 | At least one full evolution loop | `demo.py` and `test_demo.py` |

**Milestones:** M1 (graph kernel), M2 (graph intelligence), M3 (lifecycle loop) and M4 (demonstration and trace) are all complete.

---

## 8. Design decisions and interpretations

Where the spec was silent or ambiguous, this is what the implementation chose.

1. **Base-revision check is strict.** Any graph change after a proposal's base rejects it, even if the entities it touches are untouched. This is the simplest reading of T10. A future version could rebase automatically when the entities don't overlap.
2. **`remove_node` cascades.** Incident edges are tombstoned in the same transaction and analysed as removed edges, rather than rejecting the removal.
3. **Edge-structure changes are terminal at both endpoints**, with a fixed obligation for each relation. Each endpoint's obligation path is the changed edge itself, and its origin is that edge's ID.
4. **Obligations on tombstoned targets aren't emitted.** Those targets stay visible in the diff instead.
5. **Assumption status matters.** `accepted` makes a requirement conditional. `proposed` blocks it ("no unaccepted assumptions"). `rejected` or `invalidated` blocks it.
6. **`investigate_unknown` invalidates evidence.** Adding a hypothesized dependency on a provider therefore invalidates earlier evidence about that provider. This is conservative, and arguably over-conservative for preservation evidence. It's covered by a test and documented.
7. **The contract-check requirement after a rebase is derived from contract revisions that changed** since the old baseline, plus any unresolved `check_contract` on the target.
8. **A requirement's baseline includes its verifying tests**, so a change to a test definition (its command) also stales the work, which is a correct conservative choice. Ancestors are deliberately *excluded*: a parent change creates a `review_spec` obligation but doesn't stale implementation evidence.
9. **Deterministic output.** Edges are iterated in sorted ID order, records are sorted by `(target, obligation, origin)`, and `GADLC_FIXED_TIME` pins timestamps for reproducible traces.

---

## 9. Limitations and risks

| Area | Limitation | Severity / note |
|---|---|---|
| Completeness | Impact is complete only relative to registered edges and rules. A missing edge isn't proof of independence. | Inherent; stated in every report |
| Precision | Conservative rules produce provisional obligations. For example, a spec change to `REQ-RETRY` flags `CMP-REPORTING` and `REQ-CHARGE` through the shared component. | Medium. This is the main usability risk. Contract-preservation evidence is the intended way to cut it. |
| Evidence semantics | Contract checks and preservation results are recorded claims (kind, result, revisions), not formal proofs or executed test runs | Inherent for v0.1 |
| Code linkage | Code revisions are opaque strings. There is no Git integration and no automatic artifact or edge discovery. | Out of scope for v0.1 |
| Concurrency | One SQLite writer at a time. Concurrent proposers are handled only by optimistic rejection. | Out of scope for v0.1 |
| Edge ontology | Seven relations with fixed endpoint kinds. There is no edge update op (remove and re-add instead). | Acceptable for v0.1 |
| Work units | One open work unit per requirement; single-target units only | Simplification |
| Scale | The whole graph is loaded for each operation. That's fine for hundreds or low thousands of nodes; recursive SQL CTEs would be needed beyond that. | Low for a proof of concept |
| Obligation tracking | Obligations are recorded per change, but there is no "resolve obligation" workflow yet; only reconciliation consumes them | Gap worth closing in v0.2 |

---

## 10. Recommendations and next steps

1. **Run the comparison experiment from SPEC §9 before adding features.** Replay a corpus of seeded changes through (A) document-oriented context selection, (B) graph impact, and (C) graph impact with validated contract boundaries. Measure recall and precision against an independently curated oracle, missed impacts, verification effort, and the overhead of maintaining the graph. The `--json` output of `propose` and `impact` is already in the shape needed for scoring.
2. **Add an obligation-resolution workflow.** Add an `obligations` table with states `open`, `discharged` (with evidence reference) and `waived` (with reason). The scheduler could then rank open `check_contract` and `review_spec` items, and reconciliation would close them explicitly.
3. **Make precision tunable without weakening safety.** Allow per-edge `scope` annotations (for example, which contracts a `depends_on` actually relies on), so a provider-contract change only reaches consumers that use that contract.
4. **Integrate with Git.** Validate `code_rev` against the repository, link `artifact` nodes to file hashes, and emit `implementation` changes automatically when a realized artifact changes.
5. **Agent integration (v0.2).** Expose `context`, `propose`, `next`, `submit` and `reconcile` as explicit tool calls. Agents propose deltas, and the engine stays the deterministic arbiter.
6. **Relax the strict base-revision check.** Allow automatic rebase when a proposal's touched entities and impact region don't overlap the intervening commits, while keeping rejection on overlap.

---

## 11. How to reproduce

```bash
cd graph-adlc
python -m pip install -e '.[test]'
pytest -q                                   # 44 passed
python -m graph_adlc.demo --workdir out     # human trace + out/trace.json + out/snapshot.json
./scripts/demo.sh                           # same sequence with the gadlc CLI
```

**Key files:** [`README.md`](README.md) (usage), [`docs/SPEC.md`](docs/SPEC.md) (normative spec), [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) (release gate), [`fixtures/`](fixtures/) (scenario data), [`tests/`](tests/) (acceptance suite).
