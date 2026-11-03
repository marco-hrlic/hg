# Oracle labeling guide (v1.0, frozen before any method was run on the corpus)

The oracle states which **(target, obligation)** checks are *necessary* for a proposed change. It is written from
the scenario description, the readable seed specification and the scenario's **ground facts**. Ground facts are
stipulated statements about code and behaviour that code inspection would reveal; they are only in `oracle/`. No
method can see them. It is never written from engine output.

## Universe

* **Targets.** Any node except the ones directly modified by a node operation (`update_node`, `add_node`,
  `remove_node`). Endpoints of an added or removed edge *are* targets.
* **Artifacts collapse to components.** Every artifact is realized by exactly one component in both fixtures, so
  artifacts are judged at component level. Scoring applies the same collapse to every method's predictions.
* **Contracts as targets.** A contract gets `check_contract` only when its own text may need to change.

## Obligation semantics

| Obligation | Necessary when |
|---|---|
| `review_implementation` | the component's code must (or very likely must) change to satisfy the changed spec |
| `check_contract` | a requirement/component constrained by, or relying on, a changed contract or provider behaviour must be re-verified against it |
| `rerun_test` | the test verifies something whose behaviour/meaning changed, **or** executes changed code (minor if only indirectly) |
| `review_spec` | a spec node (goal, capability, requirement, question) may now be textually inconsistent with a changed parent, child, question or sibling spec, with no contract as the medium |
| `investigate_unknown` | the change crosses a dependency whose existence is unconfirmed |

* At most one obligation per target is necessary, the one that best describes the required action.
* **Test-covered requirements.** If a registered test verifies a requirement and is itself necessary, a
  `check_contract` on that requirement is *discretionary*. It is labeled `unknown`, unless the facts show the test
  cannot detect the change. If no registered test covers the requirement, `check_contract` on it is necessary
  whenever its behaviour may change.

## Labels

* `necessary_checks`: required, with severity `critical` (would ship a money/data-correctness or contract-breaking
  defect), `major` (a requirement could be unmet) or `minor` (hygiene or low probability).
* `unknown_checks`: genuinely discretionary or undeterminable. These are excluded from the judged universe and
  counted separately.
* `explicitly_not_required`: targets the reviewer considered and rejected. This list is informational. Anything
  not listed as necessary or unknown counts as not required.
* `material_impacts`: artifacts whose behaviour or content actually changes. This is distinct from needing review.

## Known threats to validity (disclosed, not mitigated)

1. **Single reviewer.** No second independent reviewer was available, so there was no disagreement adjudication.
2. **Not blind to the rules.** The reviewer (the same agent that built the harness) read the v0.1 rule table before
   writing the oracle. It did not see any prediction for these scenarios. Scenario design may still be biased
   toward or against known rule behaviour.
3. **Stipulated facts.** There is no real code. Ground facts are synthetic, so "missed check" is a proxy, not an
   escaped defect.
4. The post-unblinding review (`oracle/review_v1_1.jsonl`) is logged separately, and scores are reported for both
   v1.0 and v1.1.
