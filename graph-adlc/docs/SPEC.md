# Graph-Driven Agentic Development Lifecycle — v0.1 Prototype Specification

**Status:** implementation-ready design proposal  
**Goal:** test whether a persistent, incrementally refined specification graph can drive implementation readiness, scope change impact, and coordinate specification/code reconciliation.

## 0. Thesis and boundary

Discovery seeds a **partial specification graph**. Refinement and implementation proceed concurrently, but implementation is allowed only for sufficiently specified leaves (possibly under tracked assumptions). Each modification is a **versioned graph transaction**. A typed influence engine computes *candidate* affected nodes and the validation actions required before the change can be accepted. Code and tests are linked back to the graph; observations from implementation may in turn propose graph changes.

This is **not** a complete autonomous developer, graph ontology, requirements proof system, or orchestration framework. Specifically, a graph-derived influence domain is only complete relative to registered edges/rules; lack of an edge is *not* proof of no dependency.

### Design requirements

- **R1**: Seed a graph with incomplete and unresolved regions.
- **R2**: Represent requirements, contracts, components, tests, code artifacts, and unanswered questions as addressable entities.
- **R3**: Derive implementability locally from requirements, blocking questions, contracts and assumptions. Do not use a single global project phase.
- **R4**: Allow atomic, optimistic-concurrency graph deltas; retain reason and provenance.
- **R5**: Compute explainable candidate influence under edge-specific directional rules, including deleted edges.
- **R6**: Track whether work/evidence was produced against superseded spec revisions; never silently consider it current.
- **R7**: Materialize implementation work packets from ready leaf nodes, with scoped context.
- **R8**: Demonstrate at least one discovery → refinement → implementation → contradiction → impact → reconciliation loop.

## 1. State model

At lifecycle step `t`:

`X_t = (G_t, C_t, W_t, E_t)`

- `G_t = (V_t, E_graph,t)`: versioned specification graph.
- `C_t`: repository state, represented for the prototype by Git revision and code artifact paths.
- `W_t`: work units created for implementation/refinement/verification.
- `E_t`: evidence (test results, assertions, reviews, traces), keyed to spec and code revisions.

Agents/controllers produce **proposals**, not uncontrolled mutations:

`proposal = (base_graph_revision, operations[], rationale, actor, evidence_refs[])`

The system computes impact and gate results; an accepted proposal atomically creates a new graph revision. An implementation step may change code before or after a graph proposal, but claiming completion requires reconciling the two revisions.

### 1.1 Node types (closed enum for v0.1)

| Kind | Meaning | Additional data |
|---|---|---|
| `goal` | top-level business outcome | optional success criteria |
| `capability` | bounded system capability | optional description |
| `requirement` | implementable behavior or rule | acceptance criteria, open questions, assumptions |
| `contract` | interface/invariant/guarantee | statement, status, optional checks |
| `component` | logical implementation unit | role, optional contract refs |
| `artifact` | real repository file/symbol | path, optional symbol |
| `test` | verification artifact | command, expected outcome |
| `question` | unresolved discovery/design question | blocking flag, state |

Every node: `{id, kind, title, body, data, rev, created_at, updated_at}`. Stable IDs across revisions. `data` is JSON validated by node-kind schema. Soft-delete nodes (tombstones) so history and affected dependencies remain inspectable.

### 1.2 Edge types and **fixed directions**

| Edge | Direction | Meaning |
|---|---|---|
| `refines` | parent → child | child refines parent; forms the hierarchical backbone |
| `depends_on` | consumer → provider | consumer relies on provider |
| `constrained_by` | subject → contract | subject must satisfy that contract |
| `implements` | component → requirement | component realizes requirement |
| `verifies` | test → requirement/contract | test provides evidence toward a claim |
| `realized_by` | component → artifact | component is implemented in artifact |
| `blocks` | question → target | unresolved question blocks or qualifies target |

Edges carry `{id, src, dst, relation, rev, status, evidence_ref?}`; `status ∈ {confirmed,hypothesized}`. Referential integrity and allowed endpoint kinds are validated. All `refines` edges form an acyclic forest; a node has at most one refinement parent for v0.1 (other links use cross-edges). Refinement does **not** imply that all descendants are implementable.

### 1.3 Readiness is derived, never set as a global status

Only **leaf requirements** can become implementation work candidates in v0.1. A leaf requirement `r` is:

- `blocked` if any directly linked blocking, unresolved `question` exists; if acceptance criteria are empty; or if a required contract is explicitly incompatible/unresolved.
- `conditional` if the behavioral criteria are present and blocking questions are resolved, but at least one dependency/contract is relied on under an explicit, tracked assumption or hypothesized edge.
- `ready` if criteria are present, no blocking questions exist, and known required contracts are accepted with no unaccepted assumptions.

`conditional` requirements can be implemented only by a work unit recording the assumptions and their originating revisions. Unknown dependencies outside the graph remain a documented limitation, not a valid reason to infer safe implementation. An implementability check returns structured reasons, not only a boolean.

**Important:** An already implemented requirement can become `stale` in a separate *evidence freshness* dimension. Specification readiness and implementation freshness must not be conflated.

## 2. Graph transactions

Supported operations:

```json
{"op":"add_node","node":{...}}
{"op":"update_node","id":"REQ-RETRY","expected_rev":2,"patch":{...}}
{"op":"remove_node","id":"...","expected_rev":1}
{"op":"add_edge","edge":{...}}
{"op":"remove_edge","id":"EDGE-7","expected_rev":1}
```

`update_node.patch` is restricted to known mutable fields (`title`, `body`, `data`); v0.1 does not allow changing node ID or kind. Each change declares an **aspect**: `semantic`, `contract`, `implementation`, `structure`, or `evidence`. A caller may specify aspect, but the engine must validate it against the changed fields and affected node kinds. For uncertain classification, use the conservative `semantic` or `structure` category; do not accept an arbitrary `evidence` classification to suppress impact.

Transaction pipeline:

1. Check `base_graph_revision` and per-entity expected revisions.
2. Apply delta to an isolated proposed graph (no durable mutation yet).
3. Validate schema, references, endpoint types and refinement acyclicity.
4. Compute influence using **the union of old and proposed edges**, so removals are not invisible.
5. Recompute readiness for changed regions and all affected dependents.
6. Derive action requirements (`review_spec`, `review_implementation`, `rerun_test`, `check_contract`, `investigate_unknown`).
7. Produce a preview with `impact`, explanations, blocking issues, stale work and newly ready work.
8. Commit atomically if hard invariants and configured approval requirements are satisfied. Record event and evidence.

`propose` is read-only. `commit` must reject obsolete revisions and mutations failing hard invariants. Domain reviews may be pending after commit: the commit must leave affected work/evidence explicitly marked `needs_revalidation`, not incorrectly `verified`.

## 3. Influence algorithm

### 3.1 Important distinction

The v0.1 engine computes **candidate verification obligations**, not a mathematically proven complete semantic impact set. It must report *why* each target is included and what check to perform. It must not silently stop on an unknown/hypothesized relationship.

An influence record is:

`{target_id, obligation, origin_id, path: [edge_ids], reason, confidence}`

Obligations are a set: `review_spec`, `review_implementation`, `rerun_test`, `check_contract`, `investigate_unknown`.

### 3.2 Initial deterministic propagation policy

- `semantic` change to a parent → recursively `review_spec` on refining children.
- `semantic` change to a child → `review_spec` on its parent, **terminal** (do not automatically fan out to siblings).
- `semantic` change to a requirement → `review_implementation` on components with `implements → requirement`; `rerun_test` on tests with `verifies → requirement`.
- `contract` change → `check_contract` for subjects with `constrained_by → contract`; `rerun_test` for tests verifying that contract.
- Relevant provider `contract` or externally observable behavior change → `check_contract` for consumers with `depends_on → provider`. Pure implementation change cannot cross this boundary *unless* contract preservation is unverified or unknown, in which case add `check_contract` and provisionally flag the consumer.
- `implementation` change to artifact → `review_implementation` on component with `realized_by → artifact`, then `check_contract` for requirements implemented by that component (terminal until a contract/behavior violation is established).
- `structure` changes start from both incident endpoints and evaluate relevant rules across the old/proposed graph union.
- A hypothesized edge on any candidate path requires `investigate_unknown` and cannot justify pruning.

Use a queue/worklist and visited keys `(node_id, aspect, obligation)`; terminal obligations must not propagate. Preserve at least one shortest explanation path per target/obligation. Cycles in cross-dependencies are permitted, so de-duplication is mandatory.

**Pruning policy for v0.1:** an unchanged contract is sufficient to *suggest* a boundary, but the engine may declare a consumer **unaffected** only if an explicit validation result records the relevant contract as preserved under the same assumptions/spec version. Otherwise the consumer remains subject to `check_contract`. This is conservative by design.

### 3.3 Pseudocode

```python
def analyze_impact(before, after, delta):
    graph = union_of_relevant_edges(before, after)  # includes removed links
    queue = seed_change_events(delta, before, after)
    visited, obligations = set(), {}
    while queue:
        event = queue.popleft()
        key = (event.node, event.aspect, event.obligation)
        if key in visited:
            continue
        visited.add(key)
        record_obligation_with_path(obligations, event)
        if event.terminal:
            continue
        for edge in graph.incident(event.node):
            for propagated in rule_table(event, edge, graph):
                queue.append(propagated)
    return sorted(explain_and_group(obligations))
```

## 4. Work unit creation and lifecycle controller

Work unit schema:

```json
{
  "id": "WORK-19",
  "kind": "implement",
  "target_ids": ["REQ-RETRY"],
  "base_graph_revision": 12,
  "node_revisions": {"REQ-RETRY": 3, "CON-IDEMPOTENCY": 2},
  "assumptions": [],
  "context_node_ids": ["REQ-RETRY", "CON-IDEMPOTENCY", "CMP-PAYMENT"],
  "acceptance_criteria": ["Retries respect maximum attempts", "Duplicate requests do not double charge"],
  "status": "proposed"
}
```

State transitions: `proposed → active → submitted → verified`; alternative transitions: `→ blocked`, `→ needs_revalidation`, `→ abandoned`. Work unit completion must reference code revision and verification evidence, and confirm that tracked graph revisions are still current (or explicitly rebase/resolve mismatches).

### Minimal deterministic scheduler

Return a **ranked set** of suggested actions, not a fictitious globally optimal plan:

1. Reconcile invalidated work/evidence where risk is high.
2. Resolve blocking questions that unlock the greatest count of leaf requirements (simple graph count, not a claimed precise economic value).
3. Implement ready requirements, favoring smaller scope and fewer uncertain dependencies.
4. Refine outlined requirements when the implementation frontier is empty or capacity remains.

For v0.1 this policy is transparent and configurable, and no LLM is required. Agent integration later consumes these suggestions through explicit tool calls.

### Context materialization

`context(requirement_id)` returns: target + refinement ancestors + linked contracts + provider nodes/contracts + components implementing target + tests verifying target + known blocking questions. Include node IDs and revisions. Do **not** silently include every descendant or the whole graph. Preserve provenance and any open assumptions. A generated text prompt is a *view* of this graph, not its source of truth.

## 5. Storage / API / CLI

### Storage

Python 3.12+; `sqlite3` standard library; one database; no external graph server.

Tables:
- `nodes(id PK, kind, title, body, data_json, rev, deleted, created_at, updated_at)`
- `edges(id PK, src FK, dst FK, relation, status, rev, deleted)`
- `changes(id PK, graph_rev, base_graph_rev, actor, rationale, operations_json, impact_json, created_at)`
- `work_units(id PK, kind, targets_json, baseline_json, context_json, status, code_rev, created_at, updated_at)`
- `evidence(id PK, work_unit_id, node_id, kind, result, spec_revs_json, code_rev, details_json, created_at)`
- `meta(key PK, value)` with `graph_revision`.

Use a single SQLite write transaction for application of graph delta + change event + stale markers. Keep JSON snapshot export/import for reproducible fixtures and future Git review. Database history is authoritative in v0.1; Git tracks source code, fixtures and exported snapshots.

### Programmatic core (no HTTP server needed)

```python
seed_graph(snapshot) -> GraphRevision
get_node(id) -> Node
get_subgraph(root_id, depth=None) -> GraphView
readiness(requirement_id) -> ReadinessResult
frontier() -> list[ReadinessResult]
propose(delta) -> ProposalReport
commit(delta, approvals=None) -> GraphRevision
impact(node_id, aspect) -> list[ImpactRecord]
context(requirement_id) -> ContextPacket
suggest_actions() -> list[SuggestedAction]
create_work_unit(requirement_id) -> WorkUnit
submit_work_unit(work_id, code_revision, evidence) -> WorkUnit
reconcile(work_id) -> ReconciliationReport
```

### CLI targets

```bash
gadlc init --db demo.sqlite --seed fixtures/payment_seed.json
gadlc status --db demo.sqlite
gadlc frontier --db demo.sqlite
gadlc context --db demo.sqlite REQ-RETRY
gadlc impact --db demo.sqlite REQ-RETRY --aspect semantic
gadlc propose --db demo.sqlite --delta fixtures/retry_refinement.json
gadlc commit --db demo.sqlite --delta fixtures/retry_refinement.json
gadlc next --db demo.sqlite
gadlc export --db demo.sqlite --out snapshot.json
```

Command syntax is the proposed interface, **not a claim that these commands are already implemented**.

## 6. Demonstration: payment retry vertical slice

Seed the graph from `fixtures/payment_seed.json`:

- Goal: reliable checkout.
- Payment capability; reporting capability.
- A well-specified payment requirement, and a retry requirement with a blocking unanswered question.
- An idempotency contract, payment component, retry test, and payment code artifact.
- Reporting component depends on the payment component.
- A ready unrelated reporting requirement to test that impact is not global.

Expected run:

**A. Discovery**: ingest seed; `REQ-RETRY` is blocked, `REQ-CHARGE` is ready, unrelated `REQ-REPORT` is ready.

**B. Refinement**: answer retry question, add acceptance criteria and explicit assumptions; retry becomes ready or conditional, without blocking unrelated work.

**C. Implementation**: create work unit for retry; save a spec-revision baseline; submit code and verification evidence.

**D. Contradiction**: discover external provider does not guarantee idempotency after its retention window; propose change to `CON-IDEMPOTENCY`.

**E. Influence**: retry requirement gets `check_contract`, its verification needs re-evaluation, payment component and dependent reporting integration receive review obligations unless a validated boundary allows pruning; unrelated leaf `REQ-REPORT` must not automatically become invalidated solely because it shares the same root.

**F. Reconciliation**: mark evidence/work against old contract stale, update spec and implementation, rerun required checks, establish fresh evidence. No silent acceptance.

## 7. Acceptance tests (must be automated)

See `tests/ACCEPTANCE.md`; numbered requirements below are release-blocking.

1. **Partial seed**: initialization permits incomplete requirements and open questions.
2. **Local readiness**: blocked retry does not block ready charge or reporting requirements.
3. **Conditional work**: assumptions are explicit and recorded in work unit baseline.
4. **Refinement**: resolving a question changes retry readiness without modifying unrelated node revisions.
5. **Directional impact**: changing a dependent doesn't automatically invalidate its provider.
6. **Contract propagation**: changing a provider's contract identifies its direct consumers and relevant tests.
7. **Pruned boundary**: verified contract preservation can stop propagation; an unverified claim cannot.
8. **Removed edge**: deleting an edge still surfaces both endpoints in the proposed change's impact.
9. **Cycle safety**: two cross-dependent components don't create an infinite worklist.
10. **Revision conflict**: stale expected revisions reject commits atomically.
11. **Evidence freshness**: a passed test against rev 4 is flagged stale after a relevant rev 5 change.
12. **Provenance**: every impact obligation includes an originating change and an explanation path.
13. **Context bounds**: requesting retry context excludes unrelated reporting details unless an explicit relevant edge connects them.
14. **Reconciliation**: work cannot become `verified` if a required current check is still missing.

## 8. Implementation order / definition of done

**Milestone 1 — graph kernel:** models, validation, SQLite store, seed/export, versioned deltas, invariant tests.

**Milestone 2 — graph intelligence:** readiness, context packets, deterministic influence rules, explanation paths, no silent pruning, acceptance tests 1–9 and 12–13.

**Milestone 3 — lifecycle loop:** work units, deterministic controller, evidence freshness, conflict handling, reconciliation, tests 10–11 and 14.

**Milestone 4 — demonstration:** replay the vertical slice end-to-end, with a machine-readable trace of graph revisions, frontiers, changes, impact, actions and evidence.

**v0.1 done when:** all 14 tests pass; a reviewer can reproduce one evolution loop with the CLI; each graph mutation has a readable diff and explanation of effects; stale evidence is not treated as current; and unrelated ready requirements remain schedulable.

## 9. Limits, falsifiability and the next experiment

- The graph cannot guarantee all true semantic dependencies are represented.
- Contract-preservation checks are evidence-conditioned; no formal soundness guarantee in natural-language specs.
- A graph may increase overhead rather than reduce total work. Measure this explicitly.
- For an empirical comparison, replay the same set of seeded changes through: (A) document-oriented context selection, (B) graph-based impact selection, (C) graph + validated contract boundaries.
- Measure impact recall and precision against an independently curated oracle; missed impacts and escaped failures; time/tokens for implementation and verification; graph maintenance cost; and amount of blocking/rework.

Only after demonstrating value should v0.2 introduce agent-based extraction of nodes/edges, repository dependency discovery, richer policy selection, and distributed execution.

## 10. Implementation defaults

- Use a small library and `argparse`/`Typer` CLI; `pytest` for tests. No web server or graph visualization required.
- Graph revision and node revision are independent monotonic integers.
- All IDs are stable strings, not auto-numbered relative positions.
- Modifying a child does **not** automatically rewrite its parent; it creates a parent-consistency obligation.
- In a transaction containing changed and deleted edges, evaluate old and proposed links; report conservative scope if uncertain.
- Avoid inventing semantic facts from missing graph edges; record open questions or ambiguous dependencies instead.
- Use canonical JSON for deterministic fixture diffing; represent timestamps in UTC.
- Do not treat LLM confidence as a substitute for test or contract evidence.

### Relevant technical precedent

- SQLite supports recursive graph/tree queries: https://www.sqlite.org/lang_with.html
- Bazel Skyframe illustrates dependency-based incremental invalidation and change pruning: https://bazel.googlesource.com/bazel/+/3b9ed6e9d3570a0c67e0d59e65b3785bbc1fad99/site/en/reference/skyframe.md
- JSON Schema Draft 2020-12 for validating exported node/edge payloads: https://json-schema.org/draft/2020-12/
