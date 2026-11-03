# graph-adlc v0.2 — influence evaluation harness

This is a reproducible comparison of three ways of choosing what to review or re-test after a spec, contract,
code or structure change:

| Method | What it does |
|---|---|
| **A** | Document baseline. A frozen, deterministic ID/cross-reference retrieval over a readable spec rendered from the same graph. 1 hop, budget 12 sections. |
| **A2** | Sensitivity condition only: A with 2 hops. It is reported separately and is not a primary comparison. |
| **B** | The unmodified v0.1 engine: `GraphADLC.propose(delta)` with no evidence references, so pruning cannot fire. |
| **C** | The same engine with `evidence_refs` = every evidence record available in the scenario, so v0.1's admissible contract-preservation pruning can fire. |

The handoff and the measurement contract are in [`../docs/v0_2/`](../docs/v0_2/). The findings are in
[`RESULTS.md`](RESULTS.md).

## Reproduce

```bash
cd graph-adlc
python -m pip install -e '.[test]'
evaluation/run_all.sh
```

The script runs the v0.1 suite, writes the predictions, scores them against oracle v1.0 and v1.1, then runs the
evaluation tests. The individual steps are:

```bash
PYTHONPATH=evaluation python -m graph_adlc_eval.run    --dataset evaluation/scenarios --out evaluation/out
PYTHONPATH=evaluation python -m graph_adlc_eval.report --predictions evaluation/out/predictions.jsonl \
    --oracle evaluation/oracle --dataset evaluation/scenarios --out evaluation/out
PYTHONPATH=evaluation python -m pytest -q evaluation/tests
```

A full run takes about 1 s. Everything except `out/timing.json` is byte-for-byte deterministic, and a test checks
this. There is no RNG.

## Layout

```text
evaluation/
  PREREGISTRATION.md, config.json   frozen method/normalization/scoring choices (committed with the oracle, before any run)
  scenarios/manifest.json           dataset 1.0: 24 scenarios
  scenarios/cases/SC-xx.json        seed ref, ordered setup (commits + evidence records), proposed delta, tags
  scenarios/seeds/orders_seed.json  second, independent fixture (orders/inventory/notifications); the payment seed is ../fixtures/
  scenarios/documents/SC-xx.md      rendered readable spec + change request (method A's only input; regenerated with --write-documents)
  oracle/LABELING_GUIDE.md          labeling rules and disclosed threats to validity
  oracle/v1/SC-xx.json              blind oracle: ground facts, necessary/unknown checks, severity, material impacts
  oracle/review_v1_1.jsonl          post-unblinding decision for every predicted pair the v1 oracle did not label
  graph_adlc_eval/                  dataset, document (renderer + A), methods (A/B/C adapters), normalize, run, scoring, report
  tests/                            evaluation regression gate (18 tests)
  out/                              generated: predictions.jsonl, raw/, metrics_v1*.json, per_scenario_v1*.csv, report_v1*.md, costs, timing
  RESULTS.md                        interpretation
```

## Guarantees and how they are enforced

* **No oracle leakage.** `run` never reads `oracle/`. A test runs the whole prediction pipeline with file reads
  under `oracle/` made to raise. A scenario file that contains oracle keys is rejected by the loader.
* **No mutation.** Each (scenario, method) pair seeds its own temporary SQLite DB and replays the setup through the
  public API (`seed_graph`, `commit`, `record_evidence`). The DB fingerprint is compared before and after each
  `propose`. Input file hashes are checked after the run and recorded in `out/run_manifest.json`.
* **Same facts for A, B and C.** All three see the same post-setup graph and the same evidence records. A sees them
  as text. B gets no `evidence_refs`, so its pruning can never fire. C gets all of them.
* **Normalization**, pre-registered in `config.json`:
  * identity obligation map;
  * artifact targets collapse to their realizing component;
  * targets directly modified by a node operation are excluded and listed in `out/excluded_predictions.jsonl`;
  * records are deduplicated on `(scenario, method, target, obligation)`.
* **Scoring** follows `docs/v0_2/EVALUATION_PROTOCOL.md` §C:
  * unknown labels are removed from the judged universe;
  * a zero denominator gives `null`/`N/A`;
  * macro recall covers only non-empty-gold cases;
  * negative cases and critical misses are reported separately.

## Interpreting outputs

`out/report_v1.md` is the primary, pre-registered result. `out/report_v1_1.md` is the same scoring after the logged
post-unblinding review. Each report contains:

* pair-level and target-level confusion counts;
* unknown best/worst-case figures;
* per-family results;
* the B-vs-C pruning audit, with the oracle label of every removed check;
* every miss and false positive by scenario;
* machine cost proxies.

Explanation paths for B and C (edge IDs) are in `out/raw/SC-xx.{B,C}.json`. For A, the cross-reference that
selected each target is in `out/raw/SC-xx.A.json`. **An explainable false positive is still a false positive.**
