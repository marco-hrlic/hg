# Coding-agent kickoff prompt

Implement a runnable local Python proof of concept from `SPEC.md`. Use `fixtures/payment_seed.json`, `fixtures/retry_refinement.json` and `tests/ACCEPTANCE.md` as the normative initial scenario and release gate.

## Strict scope

- Python 3.12+, SQLite (`sqlite3`), pytest; no external graph database, LLM API, web server or frontend.
- Implement typed graph validation, atomic/versioned mutations, readiness, influence explanations, bounded context, work units, evidence freshness and a deterministic next-action policy.
- Respect edge directions defined in `SPEC.md`; in particular, `refines` is parent-to-child and `depends_on` is consumer-to-provider.
- Treat influence as *candidate verification obligations*, not a proof of semantic impact completeness. Never silently suppress hypothetical or unknown dependencies.
- `propose` is read-only. `commit` must be atomic, revision guarded and auditable.
- On changes to an already verified requirement's relevant spec/contract revisions, make affected evidence stale until a fresh check is recorded.

## Suggested repository layout

```text
graph_adlc/
  __init__.py
  models.py            # typed records and enums
  validation.py        # graph invariants and payload validation
  store.py             # SQLite schema, transactions and export
  readiness.py         # leaf requirement readiness, structured blockers
  influence.py         # directional, aspect-sensitive propagation + paths
  context.py           # scoped work context packets
  lifecycle.py         # work units, evidence freshness, reconciliation
  scheduler.py         # transparent action scoring / prioritization
  cli.py               # init, status, frontier, context, impact, propose, commit, next, export
fixtures/
  payment_seed.json
  retry_refinement.json
tests/
  test_validation.py
  test_readiness.py
  test_influence.py
  test_transactions.py
  test_lifecycle.py
```

## Implementation sequence

1. Initialize repository, package, CLI and SQLite schema. Add import/export fixtures and graph invariant validation.
2. Implement `propose` and `commit` using staged graph mutations and optimistic concurrency.
3. Implement readiness and frontier, including `blocked`, `conditional` and `ready`.
4. Implement influence rule table, old/new graph union, explanation paths, terminal obligations and cycles.
5. Implement work-unit baseline revisions, test evidence, stale detection, reconciliation and suggested actions.
6. Implement the 14 deterministic acceptance tests; replay the entire payment-retry demo from a single script.

## Definition of finished

`pytest -q` passes; README has a copy-paste command sequence that produces a readable trace: initial frontier, refined frontier, work unit creation, contract change with impact, evidence invalidation, reconciliation, and unaffected work still available. Trace output must include IDs and revision numbers so that it can be independently inspected. No hidden LLM reasoning is used to produce purportedly deterministic impact results.

## Questions for later, not blockers

- Richer edge ontologies and learned semantic influence rules.
- Formal contracts, proof reuse and stronger pruning.
- Automatic edge discovery from source code and runtime traces.
- More sophisticated planning policies (value of information; expected rework cost).
- Multi-agent concurrency and a GUI.
