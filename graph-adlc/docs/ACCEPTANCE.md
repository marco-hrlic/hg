# Acceptance tests and expected assertions

Use deterministic fixtures. These tests describe required behavior; no test runner or implementation is included in this handoff.

| ID | Setup / action | Expected assertion |
|---|---|---|
| T01 | Import payment seed | All nodes and edges exist; no assumption of global completeness |
| T02 | Compute frontier | REQ-CHARGE and REQ-REPORT ready; REQ-RETRY blocked |
| T03 | Add an accepted temporary provider assumption to otherwise specified retry | Readiness = conditional and assumptions copied into work unit |
| T04 | Apply retry refinement delta | REQ-RETRY becomes ready; REQ-REPORT revision unchanged |
| T05 | Semantic change to consumer of payment component | Provider not automatically invalidated |
| T06 | Semantic/contract change to CON-IDEMPOTENCY | Consumers & verifying tests receive obligations with paths |
| T07 | Validate unchanged provider contract for implementation-only modification | Can prune beyond contract, with evidence; without evidence cannot prune |
| T08 | Remove a depends_on edge | Old consumer and provider endpoints considered in impact |
| T09 | Add reverse depends_on edge to form cycle | Impact terminates and produces unique obligations |
| T10 | Two proposals built from same base rev; commit first then second | Second rejected, no partial changes |
| T11 | Pass tests against contract rev N, then change contract to N+1 | Old evidence marked needs_revalidation, not current |
| T12 | Inspect impact output | Every obligation includes origin, reason and path |
| T13 | Build scoped context for REQ-RETRY | Includes ancestors, contracts and relevant component/test; unrelated report spec excluded |
| T14 | Submit stale work unit after contract change | Cannot transition to verified without reconciliation |

## Demonstration trace expected shape

1. `init` -> revision 1, retry blocked, charge/report ready.
2. `propose retry_refinement` -> preview shows newly ready retry and consistency obligation on parent.
3. `commit retry_refinement` -> revision 2.
4. `create_work_unit REQ-RETRY` -> baseline captures requirement and contract revisions.
5. `submit_work_unit` with test result -> verified only after required checks.
6. `propose CON-IDEMPOTENCY change` -> impact includes retry and tests, plus provider-dependent consumers as applicable.
7. `commit` -> evidence stale; controller prioritizes reconciliation.
8. `reconcile` with updated code/test/contract evidence -> current, unaffected work remains available.
