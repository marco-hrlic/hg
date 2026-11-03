# graph-adlc v0.3 — LLM-assisted graph maintenance: design

Handoff: [`docs/v0_3/KICKOFF.md`](v0_3/KICKOFF.md). Results: [`evaluation/llm_graph/RESULTS.md`](../evaluation/llm_graph/RESULTS.md).

v0.3 adds an opt-in layer, `graph_adlc/llm_graph/`, on top of the unchanged v0.1 engine. In this layer a model may
**propose** and **challenge**. Graph state still changes only through reviewed `GraphADLC.commit` transactions.

## 1. The loop

```text
sources (files + immutable Git commits/diffs)          SourceBundle / SourceCatalog      source.py
        │  deterministic packer: line numbers, sha256, truncation disclosure
        ▼
LLMProposer.propose_update(graph, bundle, change_event)  versioned prompt, recorded call  model_adapter.py
        │  Proposal envelope: ops + citations + support + questions + contradictions     proposal.py
        ▼
validate_proposal: citations → support → policy → engine dry run (GraphADLC.propose)     validation.py
        │  per-op verdict: admissible | quarantined | duplicate | invalid
        ▼
review gate: every op gets accept | reject | defer (quarantined needs acknowledge)      review.py, cli.py
        │  accepted ops → one ordinary engine delta → GraphADLC.commit (strict base + expected_rev)
        ▼
ledger: proposals, calls, decisions, provenance (sidecar tables)                         ledger.py
        ▼
spec-only Markdown projection; influence before/after; challenge for missing edges       projection.py, challenge.py
        ▼
next change event (same persistent graph)                                                maintenance.py
```

## 2. Authority boundaries

| Party | Authoritative for | Enforced by |
|---|---|---|
| Stakeholder notes, requirements, decision records | business intent | `authority_conflict`: an `update_node` on a goal, capability, requirement or contract whose only verified evidence is code or tests is quarantined. A contradiction becomes a blocking **question** node, never a replacement (`_contradiction_questions`). |
| Graph | structured spec state *after* a reviewed commit | Only `apply_review` → `GraphADLC.commit` writes nodes and edges. `propose`, validation and challenge are read-only. |
| Git | code | Code is read with `git show <sha>:<path>`. Citations pin the commit SHA in `source_id` (`git:<repo>@<sha>:<path>`). Nothing is written to the repository, and the projection never contains code. |
| Evidence | what was actually run, and its coverage | Model text is never evidence. `contract_preserved` claims are flagged. `assess_preservation_evidence` requires declared coverage of the change's declared input domain before a record can justify pruning. |
| Model | nothing | It can propose, challenge and raise questions. It cannot approve, commit, create evidence or bypass validation. Its numeric confidence is recorded and never used. |

## 3. States

**Proposal** (`llm_proposals.status`): `validated` → `reviewed` (no graph change) or `committed` (has a `change_id`)
| `invalid` (unparseable, or the engine refused) | `stale` (the graph moved and a safe rebase was impossible; regenerate).

**Operation verdict** (validation), from most to least severe:

* `invalid`: a fabricated or stale citation, a schema or ontology violation, an engine rejection, a removal
  supported only by silence, or a revision conflict. It **cannot be accepted**.
* `quarantined`: unsupported, insufficient support, a suspected duplicate, an existing id with different content, a
  removal (always), a contradiction, an authority conflict, a suspected or declared editorial update, or a
  resubmission of a rejected suggestion. It can be accepted only with an explicit `acknowledge`.
* `duplicate`: already in the graph. Accepting it is a no-op, which makes reprocessing a source idempotent.
* `admissible`.

**Support**: `supported` (a verified excerpt and a claim of support), `hypothesized` (stored as an edge with status
`hypothesized`, which v0.1 already treats conservatively: it raises `investigate_unknown` and never prunes), or
`insufficient` (quarantined; it should be raised as a question).

**Citation**: `verified` (the whitespace-normalised excerpt occurs in the cited lines, or is relocated with a note),
`stale` (it cites a superseded version known to the catalog), `fabricated` (unknown source or hash, out-of-range
lines, or an excerpt that is not present), or `unlocatable` (empty excerpt).

## 4. Integration points (existing APIs only)

* `GraphADLC.propose(delta)` is used as the admissibility dry run and the influence preview.
* `GraphADLC.commit(delta)` writes. The actor is `llm-graph:<proposal> reviewed-by:<reviewers>`; the rationale names
  the proposal and op ids; every edge's `evidence_ref` is `<proposal>/<op>`.
* On `RevisionConflict` the proposal is re-validated against the current graph. It is rebased only if every
  accepted op is still admissible, and is otherwise marked `stale` (`NeedsRegeneration`).
* `GraphADLC.impact(node, aspect)` provides the candidate influence domain for the challenge and for the probes.
* The ledger lives in extra `llm_*` tables in the same SQLite file. The engine schema is unchanged.

## 5. Incremental maintenance rules

* Stable ids: the prompt asks the model to reuse existing ids and document ids. The validator flags paraphrased
  duplicates (same kind, same normalised title) and turns exact repeats into no-ops.
* Silence is not deletion. Removals need a verified citation **and** an acknowledged human decision.
* Rejected suggestions are fingerprinted (content without ids). A resubmission with the same evidence is quarantined
  as `previously_rejected`; one with new evidence is noted, not blocked.
* Wording-only changes: the engine has no editorial aspect, and v0.3 does not add one. Instead a `no_change` op
  records the classification, and an `update_node` that is editorial (declared, or normalised-equal text) is held for
  review, so it is not committed as a semantic change by default.

## 6. Missing-edge challenge

`challenge_impact(app, change, bundle)` computes the engine's domain for the intended change. It then runs:

* a **deterministic import scan**: Python `import`/`from` statements in files realised by registered artifacts,
  mapped back to components. It returns candidate `depends_on` edges with the import line as a verified citation,
  plus questions for imports of files that no component realises; and
* optionally an **LLM challenger** (prompt `challenge.v1`).

Both produce ordinary proposals that go through the same validation and review. Nothing is inserted silently.

## 7. Model adapter

* `AnthropicMessagesClient` uses the Messages API over stdlib HTTP. The model id must be configured explicitly
  (`--model` / `GADLC_LLM_MODEL`; there is no default). The key is read at call time from the environment variable
  `api_key_env` and is never stored in records. Temperature is 0.
* `OpenAIChatClient` has the same contract over OpenAI Chat Completions. It uses JSON mode, reads the key from
  `OPENAI_API_KEY`, and if a model rejects `temperature`, retries once at the default and records that.
  `client_from_env()` picks the provider with `GADLC_LLM_PROVIDER` (`anthropic` | `openai`); when it is unset,
  OpenAI is used if only `OPENAI_API_KEY` is present.
* `ScriptedClient` is the deterministic fake. Every record it produces has `is_fake: true`, and reports label any
  result built on it as a pipeline self-check.
* Call records store the prompt id, version and sha256; the input sha256 and size; the source manifest with hashes;
  the truncation disclosure; the raw output text; usage; latency; and the parse status.
* Prompts are pinned in `evaluation/llm_graph/prompts/PROMPTS.lock.json`. A test fails if one changes.
* Prompt-injection stance: the system prompt and templates declare source text to be untrusted data. Model output is
  only ever parsed as data, and nothing the model emits can change tools, permissions or the transaction path.

## 8. Known limitations

* No live model was run when this was built. RQ1–RQ3 are **not evaluated**.
* The import scan understands only absolute Python imports. Dynamic calls, other languages and runtime wiring are
  invisible to it.
* Excerpt verification shows text exists, not that the claim follows from it. Semantic correctness is the reviewer's
  job.
* Engine limits found by the replay (unchanged here, by design):
  * a terminal `check_contract` on a consumer component never reaches the requirement it implements;
  * an implementation change to a provider does not reach tests that verify its contract;
  * a work unit whose implementing component is flagged stays `verified`;
  * a requirement with no verifying test is 'verified' with zero checks;
  * `reconcile --rebase` on an `active` unit errors.
* Ledger writes after a commit are not in the same transaction as the commit. The change log's actor and rationale
  name the proposal, so the two can be reconciled after a crash.
* The coverage gate needs both the evidence and the change to declare matching domain dimensions. Without that it
  refuses, which is safe but blunt.
