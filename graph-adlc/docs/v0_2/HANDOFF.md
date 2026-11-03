# Graph-Driven ADLC — v0.2 Evaluation Handoff

**Purpose:** evaluate the practical value and failure modes of the implemented v0.1 influence engine before expanding the architecture.

**Repository target:** `marco-hrlic/hg`, directory `graph-adlc/`, as identified by the 2026-11-03 implementation report. Inspect the actual repository and its current branch before making changes; this handoff is based on the report, not a live source-code audit.

**Starting point (reported, not independently verified):** v0.1 has 44 passing tests (including 14 acceptance tests), a deterministic Python/SQLite engine, versioned graph transactions, readiness, influence explanations, stale-evidence handling, reconciliation, scoped context, `gadlc` CLI, and a replayable payment-retry demo. The report notes that influence can be noisy, registered edges may be incomplete, contract-preservation evidence is a recorded claim rather than a proof, and Git integration is not implemented.

## 1. Goal and falsifiable questions

Build a **reproducible comparison harness**, not a new ADLC engine. Using the same change scenarios, compare:

- **A — Document baseline:** an explicitly defined document-only impact-selection procedure using a human-readable project specification and references, without traversing the graph.
- **B — Graph:** the existing v0.1 directional, aspect-sensitive impact algorithm without contract-boundary pruning.
- **C — Graph + verified boundary:** the same graph algorithm, with its existing evidence-gated contract-preservation pruning enabled when admissible.

Answer:

1. Does B achieve better *recall of genuinely required reviews/checks* than A? At what precision and maintenance cost?
2. Does C reduce unnecessary verification obligations compared with B **without increasing missed required checks** on evaluated scenarios?
3. Where do graph rules miss multihop impacts, propagate too widely, or fail because a dependency is not registered?
4. What fraction of the lifecycle benefit is due to the representation itself, as opposed to extra manually curated knowledge or evidence?

**A negative result is a successful experiment if measured and reported honestly.** Do not tune the algorithms on the held-out cases and then present those cases as independent validation.

## 2. Non-goals

Do not implement the knowledge fabric, learned/weighted edges, LLM graph extraction, autonomous agents, a web UI, an ontology redesign, a graph database, production Git synchronization, or a new obligation scheduler. Those remain candidate future topics, not prerequisites. Avoid changing core v0.1 semantics except to repair demonstrated bugs; test any repairs separately and disclose them.

## 3. Evaluation unit and dataset

A **scenario** contains a versioned graph state before the change, a proposed delta, relevant human-readable specs, optional available evidence, and an independently adjudicated impact oracle. Include the graph after the proposed delta only via the ordinary `propose` pipeline; do not bypass validation or manufacture predictions from the oracle.

Target an initial **24 scenarios**, drawn from the v0.1 payment example and at least one independent, small application fixture. A defensible distribution is:

| Change family | Target cases | Important variants |
|---|---:|---|
| Semantic / requirement | 6 | leaf, parent, acceptance-criteria change, harmless wording |
| Contract / interface | 6 | consumer break, preserved contract, failed/stale evidence |
| Implementation | 5 | internal refactor, observable behavior, shared component |
| Structure | 4 | edge add/remove, deleted dependency, cycle |
| Adversarial / incomplete graph | 3 | missing cross-link, unexpected multihop, unknown boundary |

Include at least three **true negative** scenarios with no review obligation outside the changed entity, at least three **contract-preservation** cases (current pass, failed, stale), and at least three **missing-edge** cases. Overlap with the families above is allowed; record tags for each. If the case count is reduced, state the actual count and the coverage lost rather than padding with duplicates.

For each scenario, store:

```text
id, description, tags
seed / starting snapshot identifier
setup deltas (if any), proposed delta, change aspect
human-readable spec snapshot (document-baseline input)
available evidence identifiers and validity at the proposed revision
independently adjudicated review/check oracle
independently adjudicated material-impact oracle (if knowable)
reviewer(s), rationale, disputed/unknown items
```

**Prevent leakage:** do not derive the oracle from v0.1 propagation rules. Base it on independent reading of stakeholder intent, specified contracts, code/test inspection when available, and reasoned change consequences. Prefer two independent reviewers (e.g. architect and engineer) with disagreements adjudicated and recorded. The reviewer must not see the method predictions before creating the first oracle draft. Keep `oracle/` separate from prediction inputs.

If actual code behavior cannot be established, label an item `unknown` rather than treating a plausible prediction as ground truth. Distinguish **materially changed artifacts** from **artifacts requiring review/test**; an unaffected component may legitimately require a contract check.

## 4. Fair, reproducible methods

### A. Document baseline

Create a *readable* specification snapshot from the same scenario facts used by B/C. The document must include relevant requirements, contracts, architectural relationships and stable IDs to the extent they are known, without leaking ground-truth answers. Avoid a deliberately impoverished or broken-text baseline.

Define and freeze one repeatable selection procedure before looking at oracle scores. Minimum implementation: section/ID/reference-aware retrieval over the rendered spec, including explicit cross-references and a fixed retrieval budget; output a list of candidate review/check targets with explanations. A stronger human/LLM-assisted document baseline is optional **as a separately reported condition**; do not mix it with deterministic A results. Log the exact rule/configuration and context consumed.

### B. Existing graph

Execute `propose`/`impact` on the v0.1 engine, preserve `target_id`, `obligation`, `origin_id`, path, and provisional/confidence flags. Do not attach evidence references in a way that invokes boundary pruning; record what evidence was available even when B does not use it for pruning. Never exclude hypothesized-edge obligations silently.

### C. Graph with evidence-gated pruning

Use the same graph, scenario, rule table and evidence availability as B. Enable only v0.1's existing admissible `contract_preservation` check: passing, not invalidated, correct contract revisions, and no concurrent change to the purportedly preserved contract. Confirm failed/stale evidence and bare claims do **not** prune. Log each boundary pruned and the evidence that justified it.

**Fairness rule:** A/B/C must have access to the same underlying facts. C is specifically allowed to *apply* the shared contract-preservation record as a pruning rule; any extra effort to produce that evidence counts toward C's cost. Do not give C a freshly created favorable result unavailable to the other methods without accounting for its generation cost.

Each method must output the same normalized schema (see `EVALUATION_PROTOCOL.md`). Do not manually post-edit predictions.

## 5. Metrics and interpretation

Primary unit: a predicted **(target, check obligation)** pair compared with an independently specified set of necessary checks. Normalize equivalent obligation names only through a pre-registered mapping. Also score target-only impact separately to isolate mismatches in obligation types.

Report per-scenario and aggregate:

- **Recall:** `TP / (TP + FN)`, with explicit missed checks and their severity.
- **Precision:** `TP / (TP + FP)`, with explicit unnecessary checks.
- **F1:** descriptive, not the only objective.
- **Critical misses:** missed mandatory/security/business-critical checks counted separately and shown by scenario.
- **Negative-case false alarms:** predicted checks when the oracle says none; do not hide these inside averages.
- **Effort proxies:** candidate review/check count, context size, algorithm time, evidence-production actions, manual graph/document maintenance actions.
- **Pruning safety:** for B vs C, checks removed by C, and whether any removed check was oracle-required.

Micro-average over all adjudicated check pairs; also report macro metrics for nonempty-gold cases and results by change family. For empty denominators, emit `null`/`N/A`, never silently substitute 1. Do not represent wall time, estimated review cost or context tokens as actual human hours or escaped production defects. If no real implementation replay exists, **do not report escaped defects**—report only missed impact obligations as a proxy.

Add a sensitivity table for unknown labels: best case / worst case, or report counts explicitly excluded. Confidence intervals are optional for this small exploratory corpus, but do not imply population-level significance from 24 curated cases.

## 6. Core adversarial tests

Explicitly test:

1. A change in one requirement sharing a component with unrelated requirements: does provisional fan-out become excessive?
2. A real multihop influence that crosses more than one semantic edge type: can the **origin-only fan-out** rule miss it?
3. A removed edge: old∪new traversal must still surface relevant checks.
4. A cycle: termination and deduplication must hold.
5. A missing edge: the graph must not be credited with omniscient completeness; record the miss and whether `investigate_unknown` was raised.
6. A preserved boundary with fresh, valid evidence: C may prune, and the oracle must independently assess whether this was safe.
7. Stale/failed preservation evidence: C must not prune.
8. A change that affects readiness but not code, and one that affects code but not requirement readiness.
9. A true negative: distinguish a useful narrow answer from a missing-dependency failure.

Preserve explanation paths. A false positive with an explainable path is still a false positive; explanation is not correctness.

## 7. Implementation plan

### M0 — Verify the baseline

- Inspect repository, `REPORT.md`, `docs/SPEC.md`, `docs/ACCEPTANCE.md`, and the current source/API signatures.
- Run `pytest -q` and the demo on the actual checkout; record results and any differences from the report.
- Identify the precise JSON signatures of `propose`, `impact`, export/import and evidence records. **Do not invent interface fields** from this handoff.

### M1 — Harness and fixtures

- Create a separate package/directory, e.g. `evaluation/`, that imports or invokes the existing engine.
- Build isolated temporary databases per scenario and method; no run may mutate the shared baseline.
- Add a dataset loader, schema validation, normalized prediction adapter, deterministic document renderer, document baseline, and prediction recorder.
- Record input revision/hash, method version, configuration, RNG seed if used, and execution time.

### M2 — Curated oracle and corpus

- Construct the scenario set, including adversarial and no-impact cases.
- Freeze the oracle before comparing algorithms (or isolate a tuning set and held-out set).
- Support `unknown` and reviewer disagreement; report cases excluded from a metric.

### M3 — Scoring and reporting

- Implement pair-level and target-level metrics; aggregate by family and overall.
- Produce machine-readable JSON/CSV and a concise Markdown report with confusion counts, examples of misses and noise, and costs/limitations.
- Compare B and C pruning with a dedicated diff identifying every obligation removed.

### M4 — Reproducibility and regression gate

- Add tests for normalization, oracle masking, null denominators, deterministic runs, old∪new behavior, cycles, evidence validity, and no mutations from evaluation.
- Ensure the original v0.1 tests still pass.
- Provide a single documented command to reproduce all metrics and outputs.

Suggested interface (adapt to actual repository layout):

```bash
python -m graph_adlc_eval.run --dataset evaluation/scenarios --out evaluation/out
python -m graph_adlc_eval.report --predictions evaluation/out/predictions.jsonl --oracle evaluation/oracle --out evaluation/out
```

These are *proposed* commands, not claims about existing code.

## 8. Artifacts expected from the coding agent

```text
evaluation/
  README.md                   # how to reproduce and interpret
  scenarios/                 # fixtures, deltas, readable spec snapshots
  oracle/                    # adjudicated truth and rationale; never an algorithm input
  graph_adlc_eval/           # harness, baseline, adapters, metrics, reporting
  tests/                     # evaluation regression tests
  out/                       # reproducible JSON/CSV/Markdown results (may be generated)
  RESULTS.md                 # observed numbers, bad cases, honest conclusion
```

`RESULTS.md` must distinguish what is measured from assumptions, proxies and limitations. Include 3–5 concrete false positives/false negatives with their source evidence, or fewer if that many are not observed. Report the effect of any algorithm modification both before and after the change; do not silently improve the baseline while evaluating it.

## 9. Definition of done

The evaluation milestone is **complete** when:

- All v0.1 regression tests pass, or deviations are explained with failing test details.
- The scenario set and oracle are inspectable, versioned, and reproducible.
- A, B and C produce deterministic normalized outputs for the same scenarios.
- Both obligation-level and target-level confusion statistics are generated without answer leakage.
- Missing-edge, multihop, and pruning-safety cases are visibly represented.
- The report accounts for graph curation, evidence creation, and document-baseline costs rather than asserting an efficiency gain from fewer predictions alone.
- The final assessment explicitly states whether results support, fail to support, or are insufficient to evaluate the original thesis.

**Do not make better metrics a release gate.** Make *measurement quality, safety disclosure and reproducibility* the release gate.

## 10. What to decide only after the results

Possible follow-on experiments: (i) contract-scoped `depends_on` edges to improve precision, (ii) additional propagation rules for demonstrated multihop misses, (iii) first-class obligation resolution, (iv) Git-derived code/graph links, and only then (v) agents, learned edge weights, roles/HITL, or a knowledge fabric. Preserve these as research options; none is approved for this milestone.

---

**Input reference:** `REPORT.md`, 2026-11-03, especially §§4, 6, 9–10; original `docs/SPEC.md` §9. The v0.1 implementation report is a self-reported record of delivered behavior, not an independent audit of source code.
