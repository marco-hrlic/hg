# ADLC Influence Experiment — Evaluation Protocol

This is the measurement contract for `HANDOFF.md`. Freeze it before looking at oracle results. Keep any deviations visible in `RESULTS.md`.

## A. Scenario and prediction record

Use JSONL with a stable scenario ID. The following is a **conceptual schema**; the coding agent must adapt fixture paths to the actual repository APIs and existing delta formats.

```json
{
  "id": "SC-001",
  "family": "contract",
  "tags": ["contract_preserved", "shared_component"],
  "seed_ref": "path/to/base-snapshot.json",
  "setup_refs": ["path/to/setup-delta.json"],
  "delta_ref": "path/to/proposed-delta.json",
  "document_ref": "path/to/rendered-spec.md",
  "evidence_refs": ["EV-1"],
  "description": "Illustrative scenario; not a populated fixture"
}
```

Normalize each method's output to **one record per predicted obligation**:

```json
{
  "scenario_id": "SC-001",
  "method": "B",
  "target_id": "CMP-PAYMENT",
  "obligation": "check_contract",
  "explanation": "path or document section reference",
  "path_edge_ids": ["E12", "E15"],
  "pruned": false,
  "provisional": true
}
```

The `method` is one of `A`, `B`, `C`. Preserve raw engine output separately. For A, `path_edge_ids` may be empty but `explanation` must cite a document section or ID. Normalize and deduplicate by `(scenario_id, method, target_id, obligation)`. Report dropped/unmappable records as errors, not silently.

## B. Oracle

An oracle record distinguishes **necessary check** from **material impact**:

```json
{
  "scenario_id": "SC-001",
  "necessary_checks": [
    {"target_id": "CMP-PAYMENT", "obligation": "check_contract", "severity": "major", "reason": "independent rationale"}
  ],
  "material_impacts": [
    {"target_id": "CMP-PAYMENT", "reason": "behavior actually changes"}
  ],
  "unknown_checks": [],
  "reviewers": ["reviewer-role-1", "reviewer-role-2"],
  "adjudication_notes": "Recorded before method comparison"
}
```

An explicitly adjudicated *no-check* oracle uses `necessary_checks: []` and `unknown_checks: []`. `unknown` labels are **excluded from the judged universe** and must be counted separately. Never construct oracle checks by copying graph predictions.

## C. Equivalence / scoring

Exact obligation matches by default. Define any equivalence map (e.g. a document baseline phrase mapped to `check_contract`) *before* scoring. Report checks whose labels could not be mapped.

Let `O_s` be adjudicated required check pairs in scenario `s`, `P_{m,s}` predictions for method `m`, and `U_s` unknown pairs. Define `O'_s = O_s - U_s` and `P'_{m,s} = P_{m,s} - U_s`; unknown-label predictions are counted separately as excluded. Then:

- `TP = |P' ∩ O'|`; `FP = |P' - O'|`; `FN = |O' - P'|`.
- Micro precision = `ΣTP / (ΣTP + ΣFP)`; micro recall = `ΣTP / (ΣTP + ΣFN)`.
- When a denominator is zero, report N/A, not an invented perfect score.
- Macro recall is computed only over cases with nonempty `O'`; also report total negative cases and false alarms there.
- Target-only metrics use projections of `P'` and `O'` to target IDs, deduplicated within each scenario.
- Severity is an independently assigned oracle property and is used to enumerate misses, not to hide them in weighted averages.

**Unknown-set caveat:** a partially adjudicated universe cannot make all other predicted checks confidently false positives. Require reviewers to explicitly assess every predicted check after the blind initial oracle is frozen; retain an audit trail of any added ground-truth labels and rerun the other methods against the revised oracle. This is a measurement quality gate, not a license to tailor the oracle to favored methods.

## D. Pruning comparison

Compare B and C on the **same scenario and evidence snapshot**. Compute `removed = P_B - P_C`. For each removed pair, show:

- target and obligation;
- relevant contract, evidence record and revision;
- reason that C pruned;
- oracle label: required / not required / unknown.

Any removed oracle-required check is a **pruning miss**. Treat missing/failed/stale evidence and hypothesized edges as mandatory no-prune conditions under v0.1 rules.

## E. Cost reporting

Report machine-reproducible quantities separately from estimated human costs:

- predicted checks, graph/context nodes and document tokens/characters;
- elapsed CPU/wall time (with environment and multiple runs if compared);
- number of curated graph edges/nodes and their edits;
- number of evidence records/tests needed for pruning;
- reviewer actions or time only if actually measured (otherwise `not measured`).

A reduced obligation count is **not** proof of reduced cost if it comes with missed critical checks or expensive evidence generation.

## F. Validity constraints

- Deterministic methods use frozen versioned configuration.
- Oracle data is unavailable to prediction modules.
- Scenario fixtures may not be altered after oracle unblinding without clearly marking a new dataset version.
- Synthetic scenarios provide failure-mode coverage; they do not establish real-world efficiency or defect reduction.
- Report both per-family results and examples of model failure.
