# Graph-driven ADLC v0.1 — implementation handoff

This is an **implementation-ready specification and acceptance fixture pack**, not executable prototype code.

- `SPEC.md` — architecture, data model, semantics, CLI/API contract, milestones.
- `fixtures/payment_seed.json` — initial incomplete example graph.
- `fixtures/retry_refinement.json` — sample proposed graph transaction.
- `tests/ACCEPTANCE.md` — 14 deterministic tests to implement.

Start with the storage kernel and graph validation, then readiness + influence, then work units + reconciliation. Do not start with LLM integration or a graph database: the proof point is lifecycle behavior, not tool integration.

The critical scenario is: **partial discovery → local readiness → implement → new provider constraint → explainable influence → stale evidence → reconciliation**.
- `IMPLEMENTATION_PROMPT.md` — ready-to-paste coding-agent task with scope and build sequence.
