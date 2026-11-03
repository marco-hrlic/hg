# graph-adlc — Graph-driven Agentic Development Lifecycle, v0.1

A runnable local proof of concept of [`docs/SPEC.md`](docs/SPEC.md): a persistent,
incrementally refined specification graph that drives **local implementation
readiness**, **explainable change impact**, and **spec/code/evidence reconciliation**.

It uses only Python 3.12+, `sqlite3` and `argparse`, with `pytest` for tests. There is no graph
database, LLM, web server or frontend, and no hidden reasoning: every impact
result comes from a fixed rule table over registered edges.

The lifecycle demonstrated end to end: **partial discovery → local readiness → implement →
new provider constraint → explainable influence → stale evidence → reconciliation**.

## Quick start

```bash
cd graph-adlc
python -m pip install -e '.[test]'     # provides the `gadlc` command (or use `python -m graph_adlc`)
pytest -q                              # 14 acceptance tests (T01–T14) + supporting tests
python -m graph_adlc.demo --workdir out   # full replay; writes out/trace.json and out/snapshot.json
```

### Copy-paste CLI walkthrough

The same sequence is in [`scripts/demo.sh`](scripts/demo.sh). Every output line includes
node IDs and revisions (`REQ-RETRY@r2`), graph revisions and change IDs (`CHG-3`), so the trace
can be checked independently.

```bash
DB=demo.sqlite; rm -f $DB
# A. Discovery: REQ-RETRY blocked; REQ-CHARGE and REQ-REPORT ready
gadlc init     --db $DB --seed fixtures/payment_seed.json
gadlc next     --db $DB
# B. Refinement: read-only preview, then atomic, revision-guarded commit -> graph rev 2
gadlc propose  --db $DB --delta fixtures/retry_refinement.json
gadlc commit   --db $DB --delta fixtures/retry_refinement.json
gadlc frontier --db $DB
# C. Implementation: bounded context, work unit with baseline revisions, evidence -> verified
gadlc context  --db $DB REQ-RETRY
gadlc work create REQ-RETRY --db $DB
gadlc work start  WORK-1    --db $DB
gadlc work submit WORK-1    --db $DB --code-rev git:a1b2c3 --evidence TEST-RETRY:test_result:passed
# D/E. Contradiction: provider only honours idempotency keys for 24h -> graph rev 3
gadlc propose  --db $DB --delta fixtures/idempotency_window_change.json
gadlc commit   --db $DB --delta fixtures/idempotency_window_change.json   # EV-1 stale, WORK-1 needs_revalidation
gadlc evidence list --db $DB
gadlc next     --db $DB                                                   # reconcile ranked first; REQ-REPORT still schedulable
gadlc work submit WORK-1 --db $DB --code-rev git:a1b2c3 --evidence TEST-RETRY   # REFUSED (exit 5)
# F. Reconciliation: update spec (graph rev 4), rebase, fresh test + contract evidence -> verified
gadlc reconcile WORK-1 --db $DB
gadlc commit   --db $DB --delta fixtures/retry_window_reconciliation.json
gadlc reconcile WORK-1 --db $DB --rebase
gadlc work submit WORK-1 --db $DB --code-rev git:d4e5f6 \
  --evidence TEST-RETRY:test_result:passed --evidence CON-IDEMPOTENCY:contract_check:passed
gadlc status   --db $DB
gadlc log      --db $DB                   # every change with diff, impact and invalidations
gadlc export   --db $DB --out snapshot.json
```

Add `--json` to any command for machine-readable output. Other commands:
`gadlc impact --db $DB NODE --aspect {semantic,contract,implementation,structure}` shows a hypothetical
impact (pass `--evidence EV-n` to let contract-preservation evidence prune a boundary), and
`gadlc evidence add NODE --kind contract_preservation` records a validation result.

## Layout

```text
graph_adlc/
  models.py       typed records + closed enums (node kinds, relations, aspects, obligations, statuses)
  validation.py   node payload schemas, endpoint rules, refinement forest/acyclicity, aspect classification
  delta.py        staged application of add/update/remove node/edge ops to an isolated graph copy
  store.py        SQLite schema, single write transactions, change log, events
  readiness.py    leaf readiness (blocked / conditional / ready) with structured reasons
  influence.py    rule table, old∪new edge union, BFS worklist with explanation paths, pruning policy
  context.py      bounded work-context packets (+ text view)
  lifecycle.py    work units, baselines, evidence freshness, invalidation on commit, reconciliation
  scheduler.py    transparent ranked next actions
  core.py         programmatic API (seed_graph, propose, commit, impact, context, create/submit/reconcile…)
  cli.py          gadlc CLI
  demo.py         end-to-end replay producing trace.json
fixtures/         payment_seed.json, retry_refinement.json (normative) + the two scenario deltas for D and F
tests/            test_validation / readiness / influence / transactions / lifecycle / demo
docs/             the original handoff: SPEC.md, ACCEPTANCE.md, IMPLEMENTATION_PROMPT.md
```

## Acceptance test map

| ID | Test |
|---|---|
| T01 | `test_validation.py::test_T01_partial_seed_imports_all_nodes_and_edges` |
| T02 | `test_readiness.py::test_T02_local_readiness` |
| T03 | `test_readiness.py::test_T03_conditional_with_tracked_assumption_copied_to_work_unit` (+ `T03b` hypothesized edge) |
| T04 | `test_readiness.py::test_T04_refinement_unblocks_retry_without_touching_unrelated_revisions` |
| T05 | `test_influence.py::test_T05_semantic_change_to_consumer_does_not_invalidate_provider` |
| T06 | `test_influence.py::test_T06_contract_change_reaches_consumers_and_tests_with_paths` |
| T07 | `test_influence.py::test_T07_pruning_requires_validation_evidence` (+ `T07b` stale evidence, `T07c` hypothesized edge) |
| T08 | `test_influence.py::test_T08_removed_edge_surfaces_both_endpoints` |
| T09 | `test_influence.py::test_T09_cycle_terminates_with_unique_obligations` |
| T10 | `test_transactions.py::test_T10_second_proposal_from_same_base_rejected_atomically` |
| T11 | `test_lifecycle.py::test_T11_evidence_freshness_after_contract_revision` |
| T12 | `test_influence.py::test_T12_every_obligation_has_origin_reason_and_path` |
| T13 | `test_lifecycle.py::test_T13_context_is_bounded` |
| T14 | `test_lifecycle.py::test_T14_stale_work_cannot_be_verified_without_reconciliation` |
| trace | `test_demo.py::test_vertical_slice_trace` (8-step demonstration shape from ACCEPTANCE.md) |

## Semantics and design decisions

Where the spec left a choice open, this is what the implementation does.

**Readiness** (`readiness.py`). Only leaf requirements are candidates.
- `blocked`: an open blocking question linked by `blocks`, empty acceptance criteria, a required
  contract that is `incompatible`, `unresolved` or `rejected`, or an assumption that is `proposed` (not accepted),
  `rejected` or `invalidated`.
- `conditional`: one of the following holds:
  - an accepted, tracked assumption;
  - a hypothesized `constrained_by`/`depends_on` edge;
  - a required contract in `draft`, `proposed` or `deprecated`.
- `ready`: none of the above.

Readiness never changes because evidence went stale. Freshness is a separate dimension
(`unimplemented | in_progress | current | needs_revalidation`) shown next to readiness in `gadlc status`.

**Aspects** (`validation.classify_update`). A declared aspect is checked against the node kind and the patched
fields. If it doesn't fit, it is **upgraded conservatively** and reported in `aspect_overrides`.
`evidence` is honoured only for bookkeeping keys (`last_result`, `last_run_at`, `evidence_ref`, `notes`) on tests
and artifacts, so it cannot be used to suppress impact. A `semantic` change to a contract is treated as both
`semantic` and `contract`.

**Influence** (`influence.py`, rule table in the module docstring). The engine does the following:
- It runs a BFS over the union of old and proposed edges.
- Visited keys are `(node, aspect, obligation, terminal)`.
- It never walks straight back along the edge it arrived on.
- For each `(target, obligation)` it keeps the first path found, which is a shortest path.

Only real changes ("origin" events) fan out across relation types. Derived `review_spec` obligations continue
**downward only**. A child change gives the parent a terminal `review_spec`, with no sibling fan-out.

`check_contract` on a consumer is terminal "until a violation is established". This is why `REQ-REPORT`
is not invalidated by the idempotency change, even though `CMP-REPORTING` gets a check.

Hypothesized edges are always traversed. They add `investigate_unknown`, lower confidence to `low`, and
never allow pruning.

**Pruning** (SPEC §3.2). An implementation-only change can stop at a provider boundary only if the proposal
carries `evidence_refs` to a `contract_preservation` evidence record that meets all of these conditions:
- it is about that provider;
- its result is `passed`;
- it has not been invalidated;
- its recorded contract revisions equal the current ones;
- those contracts are not changed in the same delta.

A bare `"contract_preserved": true` claim is ignored and noted. Otherwise the consumer gets a **provisional**
`check_contract`.

**Commit** (`core.commit`). The steps are:
1. One `BEGIN IMMEDIATE` transaction covers all of the following steps.
2. Re-read the graph, then check `base_graph_revision` (strict) and each op's `expected_rev`.
3. Validate the result, compute impact, and write the nodes and edges.
4. Mark stale work and evidence, and record the change (`CHG-<rev>`) with its diff and impact.

Any failure rolls back the whole transaction. `propose` runs the same pipeline without writing.

Optional approval gating: set the meta key `approval_required_aspects`, for example `contract`.

**Staleness on commit** (`lifecycle.mark_stale_after_change`). Evidence is invalidated, and recorded with the
change ID, in either of these cases:
- one of its recorded spec revisions changed;
- its node received `check_contract`, `rerun_test` or `investigate_unknown`.

A work unit becomes `needs_revalidation` if its baseline revisions changed or its target received one of those
obligations. A work unit is never left `verified` against superseded revisions.

**Work units** (`lifecycle.py`). The baseline records the following:
- `base_graph_revision`;
- revisions of the target, its contracts, its providers and their contracts, and its verifying tests;
- assumptions, each with its originating node, node revision and graph revision;
- acceptance criteria and required checks.

Required checks start as one `test_result` for each verifying test.

`submit` moves the unit to `verified` only when all of these hold:
- every baseline revision is still current;
- every required check has a passing, current piece of evidence recorded after the latest baseline.

Stale work is refused at submit. `reconcile` reports baseline mismatches, stale evidence and the obligations
raised since the baseline. `reconcile --rebase` takes the current revisions as the new baseline and adds a
`contract_check` for each contract that changed. Evidence recorded before the rebase no longer counts.

**Scheduler** (`scheduler.py`). Ranking is by tier, then a transparent score:
1. reconcile invalidated work, weighted by its count of mismatches and stale evidence;
2. resolve blocking questions, weighted by how many leaf requirements they unlock;
3. implement ready, then conditional, requirements, preferring fewer criteria and fewer uncertain dependencies;
4. refine.

Weights can be overridden as JSON in the meta key `scheduler_weights`.

**Storage**. The tables are those of SPEC §5, plus three additions:
- `changes.diff_json`;
- `evidence.invalidated_by` and `evidence.seq`;
- an `events` audit log.

Nodes and edges are soft-deleted (tombstoned), and IDs are never reused. Export is canonical JSON with
sorted keys. Timestamps are UTC; set `GADLC_FIXED_TIME` to make them reproducible.

## Known limitations (by design for v0.1)

- An impact result is a set of *candidate verification obligations*. It is complete only relative to the
  registered edges and rules. A missing edge is not proof of independence, and every report says so.
- The rules are deliberately conservative, so a semantic change to a requirement provisionally flags consumers
  of the component that implements it. Measuring how noisy this is was left for the next experiment (SPEC §9).
- Contract-preservation and contract checks are recorded evidence, not formal proofs.
- There is one writer process at a time (SQLite), no multi-agent concurrency, no GUI, and no automatic
  discovery of edges from code.

## v0.2 evaluation

[`evaluation/`](evaluation/) holds the v0.2 influence evaluation (handoff in [`docs/v0_2/`](docs/v0_2/)). It is a
24-scenario corpus with a frozen oracle, comparing a document baseline, this engine, and this engine with
evidence-gated pruning. Reproduce it with `evaluation/run_all.sh`; findings are in
[`evaluation/RESULTS.md`](evaluation/RESULTS.md).

## v0.3: LLM-assisted graph maintenance (opt-in)

[`graph_adlc/llm_graph/`](graph_adlc/llm_graph/) adds a bounded maintenance loop on top of the unchanged engine.
Independently written specs and read-only Git code go to a model, which returns a cited proposal. The proposal is
validated for evidence and provenance, reviewed op by op by a human, and committed through the normal
`propose`/`commit` path. The loop also produces a spec-only Markdown export and a missing-edge challenge. Design:
[`docs/V03_DESIGN.md`](docs/V03_DESIGN.md). Handoff: [`docs/v0_3/`](docs/v0_3/).

```bash
gadlc-llm --db g.sqlite propose --bundle bundle.json --event "requirements draft 2" --provider anthropic --model <id>
gadlc-llm --db g.sqlite show    --bundle bundle.json P-0001
gadlc-llm --db g.sqlite decide  --bundle bundle.json P-0001 --reviewer alice --accept op-1,op-2 --defer op-3
gadlc-llm --db g.sqlite challenge --bundle bundle.json --node CON-X --aspect contract
gadlc-llm --db g.sqlite export-spec --out spec.md
```

The longitudinal evaluation is in [`evaluation/llm_graph/`](evaluation/llm_graph/): two projects, 5 events each, one
persistent graph per arm.
* `evaluation/llm_graph/run_all.sh` runs the deterministic scripted replay, scoring, the report and 26 tests.
  These include the v0.1 and v0.2 suites.
* `run_replay.py --mode live` is the separate, explicit live trial. Set `GADLC_LLM_MODEL` and either
  `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`; `GADLC_LLM_PROVIDER` picks one if both are set.

**No live model has been evaluated yet.** See [`evaluation/llm_graph/RESULTS.md`](evaluation/llm_graph/RESULTS.md).
