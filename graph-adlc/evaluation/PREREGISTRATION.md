# Pre-registration (frozen with oracle v1.0)

This file is committed together with `config.json`, the scenario corpus (dataset 1.0) and `oracle/v1/`. That commit
comes **before** the method adapters were written or any method was executed on a scenario. Only a committability
check (`dataset.validate_only`) had been run, and it exposes no impact information. Any later change to these files
must create a new dataset or oracle version and must be reported in `RESULTS.md`.

* **Methods.** A, A2 (sensitivity only), B and C, as defined in `config.json`. Nothing about B or C is tuned. The v0.1
  engine is used unmodified.
* **Primary metrics.** Pair-level micro precision/recall/F1 against oracle v1.0, critical misses, negative-case
  false alarms, and B-vs-C pruning safety.
* **Secondary metrics.** Target-level metrics, macro recall over non-empty-gold cases, per-family breakdown,
  best/worst-case sensitivity for unknown labels, and cost proxies.
* **Post-unblinding review.** After the first scoring, every predicted pair not in the oracle is reviewed. Each
  decision (stays not-required / becomes necessary / becomes unknown) is logged with a rationale in
  `oracle/review_v1_1.jsonl`, and all methods are re-scored against v1.1. Both versions are reported. v1.0 stays the
  primary result.
* **Hypothesis directions (stated, not assumed).** H1: B recall > A recall. H2: C removes checks relative to B
  without removing any oracle-required check. H3: graph misses concentrate in missing-edge and multihop cases.
