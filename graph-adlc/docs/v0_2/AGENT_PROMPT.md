# Coding-agent kickoff — v0.2 evaluation milestone

You are working in `marco-hrlic/hg`, directory `graph-adlc/`, according to the supplied v0.1 implementation report. **First inspect the current repository and run its existing tests/demo**; do not assume the report describes the exact current checkout.

Read `HANDOFF.md` and `EVALUATION_PROTOCOL.md` completely. They specify the next **experiment**, not a rewrite of the graph engine.

## Main task

Create a deterministic, reproducible comparison between:

- A: document-only selection from a complete readable spec;
- B: current v0.1 graph influence without pruning;
- C: current graph influence with admissible contract-preservation evidence pruning.

Build a 24-scenario versioned corpus with independently adjudicated checks and adversarial tests, including missing edges, multihop propagation, true negatives, and valid/invalid evidence boundaries. Implement a small evaluation harness, normalization, scoring, and a Markdown results report. Keep oracle data out of the prediction path.

## Non-negotiable constraints

- Keep the existing CLI, graph schema, and 14 acceptance tests functioning; preserve all 44 existing tests unless the repository differs, and document any differences.
- Reuse `GraphADLC`, `propose`/`impact`, existing schemas, and evidence-validation logic instead of reimplementing the graph algorithm.
- Do not invent ground truth from the engine output; record independent adjudication and unresolved cases.
- Do not add a new graph DB, LLM service, UI, weighted edges, Git sync, knowledge fabric or autonomous agent orchestration.
- Avoid hidden algorithm tuning on evaluation cases. A negative result is acceptable.
- Every prediction must be reproducible, explainable, and linked to a versioned scenario.

## Work sequence

1. Inspect current repo/API, run baseline tests/demo, summarize deviations.
2. Implement the dataset schema/loader, three method adapters, and no-mutation isolation.
3. Add curated scenario fixtures and oracle, independently of algorithm output.
4. Implement strict obligation/target scoring, pruning audits, and cost proxies.
5. Add deterministic regression tests and one-command evaluation.
6. Produce `RESULTS.md`: measured statistics, notable false positives and false negatives, limitations, and a recommendation about which follow-on hypothesis merits testing.

**Completion:** do not stop at the harness; run it and include real generated results. If independent ground-truth review or actual code evidence is unavailable, disclose it, label unadjudicated cases unknown, and produce a transparent partial evaluation rather than fabricated precision/recall.
