# Graph-Driven ADLC — v0.3 Coding-Agent Kickoff

**Milestone:** LLM-assisted specification-graph construction, maintenance, and change-impact evaluation  
**Target repository:** `marco-hrlic/hg`, `graph-adlc/` (verify the actual checkout/branch before changing anything)  
**Inputs:** `REPORT.md` (v0.1 implementation report), `RESULTS (1).md` (v0.2 evaluation report), existing repository/docs/tests  
**Status:** Implementation instructions, not a claim that v0.3 exists

## 0. Read this first — what we're really building

The overall goal is **not** to outperform a document-search baseline with a hand-curated graph and a deterministic traversal algorithm. The original ADLC idea is a **persistent, incrementally refined specification graph** that humans and AI agents collaboratively build and maintain while software is implemented in ordinary Git repositories.

The graph begins during human-to-human discovery, evolves as specifications are refined, and incorporates facts learned from implementation and verification. Different regions become implementable at different times. A change can invalidate dependent specifications, code work, or verification evidence. Humans retain authority over business intent and meaningful decisions. Business-readable specs and ordinary source code remain accessible independently of the graph.

**This milestone tests the missing mechanism:** can an LLM, acting through controlled and auditable graph proposals, use natural-language specs and code evidence to create, discover, and maintain enough of the graph that its influence analysis becomes useful over successive changes?

The deterministic graph engine remains infrastructure and the authority for structural validity and transaction semantics. **The LLM must not be treated as the source of truth.**

### Context from completed experiments

- **v0.1** reportedly implemented Python/SQLite typed graph storage, readiness, influence propagation, explainable deltas, work/evidence freshness, reconciliation, a CLI and demo. Its report says 44 tests passed, including 14 release-blocking acceptance tests. Confirm this against the current repository.
- **v0.2** reportedly added an evaluation harness and 18 tests without modifying the core engine. On 24 small synthetic, hand-curated scenarios, a document baseline reached **0.593 recall**, deterministic graph impact **0.574**, and graph plus pruning **0.556** under primary oracle v1.0. These results did **not** support the preregistered influence hypotheses; they do **not** evaluate an LLM-maintained graph or a full lifecycle.
- The document baseline was rendered from the same curated graph, so v0.2 largely compared **traversal procedures given the same knowledge**, not independent document-centric and graph-centric workflows.
- **SC-14 is a non-negotiable safety case:** a current, passing contract-preservation record lacked coverage of a relevant input domain; pruning removed a necessary critical check. Never interpret `passed && current` as proof of sufficient evidence.
- Noted v0.2 failure classes: missing dependency edges, premature terminal propagation, implementation changes missing tests, lack of editorial/semantic distinction, and conservative false alarms.

Do not erase, overstate, or silently “fix” these findings. Preserve prior baselines for comparison.

---

## 1. Primary deliverable

Build a **small, end-to-end, opt-in LLM graph-maintenance loop** that works with the existing graph engine:

```text
Human-authored discovery/spec materials   Git implementation (read-only ingestion)
                  |                                  |
                  +--------------+-------------------+
                                 |
                      Source snapshots / changes
                                 |
                       LLM graph proposal
                (nodes, edges, edits, questions)
                                 |
                  Evidence + provenance validation
                                 |
                     Human review / decision
                                 |
                Existing GraphADLC propose/commit
                                 |
             Updated graph + readable spec projection
                                 |
                   Existing influence/readiness
                                 |
          Candidate obligations + unresolved questions
                                 |
             Subsequent changes repeat the loop
```

A successful prototype demonstrates **at least two successive updates** to the *same* graph, including a dependency initially absent from the graph and later discovered from source or implementation evidence. It must preserve normal Git-based code and generate a plain, readable, spec-only projection.

Build a bounded prototype, not a general-purpose agent platform.

### Research questions (keep separate)

**RQ1 — Construction fidelity.** Can the LLM identify the right entities, relationships, assumptions and unresolved questions from independently authored materials without introducing unsupported facts?

**RQ2 — Maintenance fidelity.** When source specs or code change, can it propose correct incremental graph deltas rather than reconstructing/replacing the graph or silently losing relationships?

**RQ3 — Influence benefit.** Does the *maintained* graph support more accurate or less expensive impact analysis than LLM analysis of the same documents/code without a persistent graph? Include graph-building and review costs.

**RQ4 — Lifecycle benefit (exploratory).** Over several transitions, does the persistent graph improve readiness, reconciliation, or traceable decision-making? Do not claim a general lifecycle efficiency gain from a tiny prototype.

A negative result remains a valid deliverable.

## 2. Scope and explicit non-goals

**Implement now:**

1. Source snapshot/input abstraction for independently written specs and optional Git code snapshots/diffs.
2. An LLM adapter with explicit model configuration, versioned prompt(s), structured output and recorded calls; a deterministic fake for automated tests.
3. Graph proposal generation for existing node/edge types; support additions and targeted edits, plus *explicit, reviewable* removal proposals.
4. Evidence-linked provenance, confidence/status, unresolved questions and contradiction flags for every substantive proposal.
5. Human review gate: inspect, accept/reject/defer proposals; never silently commit model-generated changes.
6. Translation of accepted proposals into **existing** `GraphADLC.propose` / `GraphADLC.commit` transactions, with normal revision/conflict checks.
7. Graph-change challenge: search relevant source/code for plausible missing relationships and present them as candidates before/alongside influence analysis.
8. A small longitudinal evaluation and a clear generated `RESULTS.md`.

**Do not build:** knowledge fabric, weighted edges or learned probabilities, autonomous code-writing/swarm architecture, UI, a graph database, broad Git synchronization, organizational role-management platform, automated requirement approval, or a new propagation engine. Do not redesign the ontology unless an existing invariant makes a thin adapter impossible; document the incompatibility before changing it.

**Do not silently enable contract-boundary pruning.** For this milestone, measure influence without pruning by default. If comparing pruning, report it as a separate exploratory condition with explicit scope/coverage checks and preserve SC-14 as an unsafe example until the evidence model is adequate.

## 3. Architectural contracts

### A. Authority boundaries

- **Stakeholder statements and approved decisions:** source artifacts and explicit human approvals remain authoritative for intent.
- **Graph:** authoritative structured specification state **after** validated/authorized commit; not omniscient truth.
- **Code:** Git remains authoritative. Store repository, immutable commit IDs, file/symbol references, and content hashes when useful; do not store source as authoritative graph payloads.
- **Tests/evidence:** keep the evidence's actual origin and coverage; an LLM description of a test run is not a test run.
- **LLM:** may propose and challenge; may not autonomously approve business changes, manufacture evidence, or bypass the graph transaction pipeline.

### B. Proposal model (sidecar; adapt to existing schemas)

Use the current node/edge enums and payload-validation rules. An intermediate proposal format could look like:

```json
{
  "proposal_id": "P-0007",
  "base_graph_revision": 3,
  "intent": "Introduce missing dependency discovered in code",
  "operations": [
    {
      "kind": "add_edge",
      "from": "CMP-ORDERS",
      "to": "CMP-INVENTORY",
      "relation": "depends_on",
      "reason": "Orders calls the inventory reservation interface",
      "source_evidence": [
        {
          "source_id": "git:example@<commit>:src/orders.py",
          "locator": "function reserve_stock; exact span resolved at runtime",
          "excerpt": "<verbatim excerpt produced from an actual input>"
        }
      ],
      "confidence": "candidate"
    }
  ],
  "unresolved_questions": [],
  "possible_contradictions": []
}
```

This is an **illustrative envelope**, not a replacement for the engine's delta schema. Resolve stable IDs and translate fields through a dedicated adapter; don't invent an edge or code excerpt in actual output. Cite source spans/line numbers, document revision or Git SHA, and record the prompt/model/configuration that led to the proposal.

A proposal may be **supported**, **hypothesized**, or **insufficiently supported**. The LLM's numeric confidence alone must never promote it to a verified fact. If an assertion isn't evidenced, surface a question instead of inserting a definitive edge.

### C. Incremental graph maintenance

- Align existing entities by stable IDs; don't create duplicates every time the same requirement is paraphrased.
- A source being silent about an existing relationship is **not** evidence to delete that relationship.
- Proposed removals and changes to approved requirements need explicit evidence and human authorization.
- Use `base_graph_revision` and existing per-entity revision semantics; on conflict, re-read and regenerate/rebase the proposal safely.
- Store accepted and rejected proposals and their rationale. Rejected suggestions should not be silently resubmitted unchanged.
- Keep unknown/missing-edge risk visible; graph completeness cannot be inferred from an absence of edges.

### D. Human-readable projections

Produce a **spec-only** Markdown export from the committed graph with a coherent narrative and traceable IDs: goals, capabilities, requirements, contracts, assumptions, decisions, and unresolved questions. It must be readable without graph tooling. Distinguish *approved* from *proposed* where the schema supports it. The projection is read-only/generated, or its human edits must be reintroduced as proposals—not written back blindly.

Keep a **code-only** view simply by retaining ordinary source in Git. No generated source files or special IDE are needed. Link graph code-artifact references to Git but do not conflate Git history with graph-revision history.

## 4. Recommended implementation phases

### M0 — Baseline and repo inspection

- Inspect current branch, `graph_adlc` API, CLI, schemas, evaluation harness, fixtures, and tests; verify reports against checkout.
- Run existing tests and demo. Record exact commands and observed results.
- Identify the smallest integration seam: new package/module under `graph_adlc/` or isolated `evaluation/llm_graph/`, using existing public APIs. Avoid duplicating core graph logic.
- State any mismatch between reports and actual code.

### M1 — Source-native input and proposal contract

- Add source catalog: project, path, snapshot hash, version, source type (`stakeholder_notes`, `requirements`, `architecture`, `code`, `test`, `decision`), and source span references.
- Use at least one **independently authored source document** as input. Do not render the document from the target graph and then claim to evaluate graph extraction.
- Build a deterministic input packer with bounded scope, source IDs and truncation disclosure.
- Define and validate the proposal JSON envelope and its mapping to existing graph deltas. Validate evidence references against the supplied source snapshot, including excerpts and locators wherever feasible.
- Add tests for unsupported edge types, stale source hashes, invented citations, duplicate IDs, unsupported deletions and revision conflicts.

### M2 — LLM proposer and review gate

- Provide an injectable interface such as `propose_update(graph_snapshot, source_bundle, change_event) -> Proposal`.
- Start with one configurable provider; don't hard-code or print credentials. If none is configured, the deterministic fake supports tests, but **do not claim an actual LLM evaluation**.
- Use a versioned prompt instructing the model to (a) identify entities, (b) propose typed links and targeted edits, (c) preserve source provenance, (d) label hypotheses, (e) raise conflicts/questions, and (f) avoid treating text absence as a deletion.
- A separate validation step checks schema, source support and admissibility using the existing engine. Syntax/schema validity **does not imply semantic correctness**.
- Implement a simple review command/API to show an itemized diff, evidence and graph-influence preview, then accept/reject/defer changes. A human gate may initially be CLI-based; no web UI.
- Commit approved operations only through the existing revision-aware transaction path. Keep the prior graph intact on failed validation or rejection.

### M3 — Repeated changes and missing-edge challenge

- Take a source edit and/or an implementation code diff after initial graph construction. Propose *incremental* edits against the same persistent graph.
- Add a `challenge_impact` or equivalent operation: given an intended change and candidate influence domain, inspect the relevant source and optionally code for likely dependencies absent from the graph. Return **candidate** relationships with supporting spans and questions; don't silently insert them.
- After review/commit, rerun influence on the updated graph and record the **before/after** impact set, new obligations, and unresolved gaps.
- Include a code-discovered dependency and at least one harmless wording-only change; the latter should not automatically force a system-wide semantic update. If the existing engine's aspect model cannot represent purely editorial changes, record the limitation and use a safe no-op proposal rather than changing the engine globally without a separate test.

### M4 — Small longitudinal evaluation

Build two small, distinct project histories (preferably one based on genuinely authored materials and one on a real small Git repository or independently prepared code fixture). Each should have:

1. A discovery/spec input independent of the graph.
2. Initial graph seed proposed by the LLM.
3. A refinement that changes readiness or constraints.
4. An implementation/source observation that reveals a missing relationship or contradiction.
5. A subsequent change requiring impact analysis and reconciliation.

The same graph must survive across events; do **not** reset it for every question. For each step, store source inputs, graph revision, model trace, proposal, validation findings, reviewer decisions, and impact before/after.

**Experiment arms (with comparable source access and explicit budgets):**

| Arm | Representation and procedure | Purpose |
|---|---|---|
| D | LLM reads independent documents/code and directly proposes impacts, no persistent graph | Real document-centric LLM baseline |
| G0 | Existing deterministic influence on starting graph, without new LLM maintenance | Isolate benefit of maintenance |
| G1 | LLM proposes/maintains graph; deterministic influence | Core v0.3 hypothesis |
| G2 (optional) | G1 + LLM missing-edge challenge, then deterministic influence after approval | Incremental semantic completeness |
| H (diagnostic, optional) | Independently curated reference graph with existing engine | Approximate upper-bound diagnosis, not a fair product baseline |

Don't compare D against graph-generated documents. Use **the same original sources** for D/G1/G2. Track model calls, context size, tokens if reported, latency, human corrections and graph maintenance actions; different resource budgets must be explicit. Freeze prompts/procedures before evaluating held-out events. Avoid giving the model oracle labels or v0.2 evaluation diagnoses as task-specific hints.

**Evaluation outputs:**

- Graph entity/link correctness (supported, contradicted, missing, uncertain) with reviewer rationale.
- Missing-edge recovery, hallucinated links, correct change classifications, duplicate entity rate.
- Impact target and obligation precision/recall *where independently adjudicated*, critical misses, false alarms and negative cases.
- Longitudinal stability: proportion of changes handled as deltas, duplicate churn, provenance drift, reversions/corrections, invalidation and reconciliation behavior.
- Cost: LLM calls/tokens, input sizes, reviewer decisions/corrections, execution and evidence costs where actually measured.
- A case-by-case record of errors and why they happened.

An independent human oracle is preferred. If not available, label results **exploratory, single-reviewer** and report unknowns rather than creating artificial certainty. At least one challenge case should resemble v0.2's **missing edge**, and one should recreate **SC-14** to ensure that a model's assurance does not justify unsafe pruning.

### M5 — Reproducible output and report

- One command runs deterministic tests and replay; a separate explicit command runs live LLM trials if credentials are available.
- Preserve full provenance and machine-readable results; include redacted model metadata and prompt versions.
- Generate `evaluation/llm_graph/RESULTS.md` with claims supported by actual runs; include null/unknown fields when no measurements exist.
- Keep v0.1/v0.2 tests passing and avoid changes that retroactively alter their recorded experimental baselines. If a core fix is unavoidable, version it and report before/after independently.

## 5. Suggested file layout (adapt to actual repo)

```text
graph-adlc/
  graph_adlc/
    ... existing engine ...
    llm_graph/                 # source snapshots, proposal, review, adapters
      source.py
      proposal.py
      model_adapter.py
      validation.py
      review.py
      maintenance.py
  evaluation/
    ... existing v0.2 ...
    llm_graph/
      sources/                 # independent specs and optional Git fixtures
      histories/               # sequential change events
      prompts/                 # pinned prompts + version hashes
      oracle/                  # never passed to prediction
      tests/
      run_replay.py
      RESULTS.md
  docs/
    V03_DESIGN.md
```

If the repo already has suitable modules, reuse them. Do not create redundant infrastructure simply to match this sketch.

## 6. Minimum acceptance cases

Implement automated tests covering at least the following, in addition to existing suites:

| ID | Acceptance behavior |
|---|---|
| L01 | Initial graph can be proposed from a source document **not generated from the graph**; accepted changes commit through existing revision checks. |
| L02 | Every substantive proposed node/edge is linked to resolvable source evidence or explicitly marked as hypothesized/needs review. |
| L03 | Fabricated or stale source references are rejected or prominently quarantined; not silently accepted as fact. |
| L04 | Same source reprocessed without changes does not create duplicate nodes/edges (idempotency or explicit duplicate warning). |
| L05 | Editing one requirement yields a targeted delta, preserving unrelated IDs and revisions. |
| L06 | Silence in a new source version does not automatically delete pre-existing relationships. |
| L07 | Proposed semantic conflict produces an unresolved question/decision rather than an unsupported approved replacement. |
| L08 | Reject/defer leaves the committed graph untouched; accepted changes are auditable and use normal `propose`/`commit`. |
| L09 | An implementation/Git diff identifies a plausible missing dependency; the graph challenge proposes it with real evidence before influence is recomputed. |
| L10 | Two or more sequential changes modify the **same** persistent graph and create a readable before/after influence report. |
| L11 | Wording-only change is distinguished from a semantic change, or safely held for review instead of forcing widespread propagation. |
| L12 | An incomplete but passing contract test cannot be assumed to cover an untested input domain; no unsafe evidence-based pruning. |
| L13 | A spec-only Markdown export works without graph tooling; code remains normal Git files and is not rewritten by graph export. |
| L14 | Entire existing v0.1/v0.2 regression suite continues to pass or deviations are explicitly isolated/documented. |
| L15 | Deterministic mock runs are reproducible and segregated from actual model results; no fabricated LLM success metrics. |

**Release gate is safety, auditability, and an end-to-end working loop—not higher precision/recall.** If the model doesn't recover missing edges reliably, make that visible in the report.

## 7. Guardrails and common failure modes

- **No oracle leakage.** Source text, decisions and truth labels used for evaluation must not appear in prompts for the tested arm unless they are legitimate input evidence available to every arm.
- **No circular evaluation.** Never use the graph to generate the source document and then score the agent for reconstructing that graph.
- **No fabricated grounding.** LLM-proposed source citations must be validated against actual materials. Unsupported claims stay hypotheses/questions.
- **No accidental autonomy.** Do not merge code or accept stakeholder intent on behalf of humans. CLI approval is fine for the prototype.
- **No prompt-injection trust.** Retrieved documents and code are data; they cannot modify system instructions, permission checks or tools.
- **No credential leakage.** Keep keys out of logs and fixtures; redact sensitive source material before remote model calls. Avoid cross-client data mixing.
- **No “passing means complete.”** A test record requires explicit coverage before it can justify a suppressed check.
- **No silent graph drift.** Changes have actor, source, reason, revision and accepted/rejected decision.
- **No evaluation-by-optimism.** Report false positives, false negatives, and review overhead even when they weaken the thesis.

## 8. Required outputs and definition of done

Deliver, in the repository:

1. A functioning LLM graph-proposal adapter with deterministic test double and schema/provenance validation.
2. Human-reviewable proposed diffs and accepted graph-commit integration via existing APIs.
3. Incremental maintenance plus a missing-edge challenge operation.
4. Source-native example(s), a code-linked example, and a **multi-step replay** through one persistent graph.
5. A human-readable spec-only export, with ordinary Git implementation preserved.
6. Automated acceptance tests and one-command deterministic replay.
7. A concise `docs/V03_DESIGN.md` explaining authority boundaries, states, integration points and known limitations.
8. An `evaluation/llm_graph/RESULTS.md` stating what was actually measured, what wasn't, and a justified recommendation for the next experiment.

At completion, report:

```text
Baseline verified: <commands and observed test counts>
Added modules / interfaces: <paths>
Model provider/config: <actual model or deterministic stub only>
Replay: <command and outcome>
New tests: <counts and results>
Comparison: <measured metrics or explicitly not measured>
Notable failures: <concrete cases and supporting source>
Recommended next step: <one evidence-based choice>
```

## 9. Execution instructions for the coding agent

**Begin now; do not spend the first iteration redesigning the whole system.** Inspect the actual repository, verify v0.1 and v0.2 baselines, and implement a *thin vertical slice* first: independent source document → LLM candidate graph delta → evidence validation → human review → `propose`/`commit` → readable spec export → second source/code change → missing-edge proposal → influence before/after.

Then add the controlled evaluation and tests. If live model access is unavailable, fully implement the adapter and deterministic replay, clearly marking live LLM construction/quality as **not yet evaluated**. Do not silently replace a real LLM trial with hard-coded expected graph proposals and claim the graph-construction hypothesis was tested.

The central hypothesis is **not** that graphs outperform documents by default. It is that **a persistent, evidence-grounded graph continually maintained by humans and agents can improve development coordination and impact reasoning over time**. Build and evaluate the smallest credible test of that hypothesis.
