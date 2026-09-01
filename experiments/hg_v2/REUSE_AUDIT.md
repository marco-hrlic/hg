# HG-v2 reuse audit

Written before any HG-v2 code, as required by the implementation plan (§23). The
question for each component is not "does something similar exist?" but "can the
existing thing carry the HG-v2 experiment without being bent out of shape?".

The repository is an implementation of the **M2-LLM protocol**: an *ensemble
diversity* experiment. Its central artifact is an `item x ensemble-member`
prediction matrix, and almost every abstraction above the I/O layer is typed on
that matrix (`ErrorMatrix`), on the four ensemble conditions
(`HS/HD/DS/DD`), or on human anchor panels.

HG-v2 asks a different question with a different unit of observation: **one
episode per item** — private AI state, a route, at most one query, one external
response with a measurable transmission cost, one final action. There is no
ensemble and no anchor panel. So the reusable layer turns out to be everything
*below* the matrix (config loading, JSONL artifacts, run directories,
provenance, figures, bootstrap reporting, test conventions) and nothing at or
above it.

---

## Component-by-component

### Item-level artifact I/O — `src/hg/io.py`

```
existing module:   hg.io (write_jsonl, read_jsonl, write_json, write_yaml,
                   write_csv, git_commit, run_id, ensure_dir)
reusable as-is:    yes, all of it
needs extension:   no
new code required: none
reason:            Already exactly the "expensive artifact is the item-level
                   row, not the plot" discipline HG-v2 needs (plan §2.2). The
                   atomic temp-file-then-rename write is what makes an
                   interrupted sweep leave a whole file or none.
```

### Run directory layout — `hg.io.RunDir`

```
existing module:   hg.io.RunDir
reusable as-is:    partly — root/config.resolved.yaml/report.md/figures are
                   exactly right
needs extension:   subdirectory names (predictions/, matrices/, bootstrap/) are
                   M2 vocabulary; HG-v2 stores episodes/
new code required: GroundingRunDir(RunDir) — a ~15-line subclass
reason:            Subclass rather than edit: the M2 layout is a contract with
                   existing results directories and with hg.pipeline.reanalyze.
                   Inheriting keeps config_path/figure()/report_path identical
                   so `results/` stays legible across both experiments.
```

### Config resolution — `src/hg/config.py`

```
existing module:   hg.config (load_experiment, ExperimentConfig, resolved_dict)
reusable as-is:    no — ExperimentConfig *requires* dataset + conditions +
                   models(family, checkpoint), and validate_panel encodes the
                   HS/HD/DS/DD diversity rule. Phase 0 has no models at all.
needs extension:   the YAML reading primitives (_read_yaml, _expand_env) are
                   private but generic
new code required: hg.config.read_config_file (3-line public wrapper, additive)
                   + hg.grounding.config.GroundingConfig
reason:            Config-first is a repository principle (plan §2.3), so
                   HG-v2 gets a real typed config object, but forcing the
                   index task into ExperimentConfig would mean inventing a fake
                   dataset, a fake model and a fake condition to satisfy
                   validators that exist to protect a different claim.
```

### Item-level schema — `src/hg/schema.py`

```
existing module:   hg.schema.ItemResult
reusable as-is:    no
needs extension:   no — leave it untouched
new code required: hg.grounding.episodes.GroundingEpisode
reason:            ItemResult.validate() rejects any row whose `condition` is
                   not one of HS/HD/DS/DD, and its fields (member_id,
                   adapter_seed, anchor_status, train_split) are ensemble
                   bookkeeping. A grounding episode instead needs route, query,
                   response, external bits, and loss before *and* after
                   grounding. Same conventions though — flat, JSON-only,
                   validate(), to_dict/from_dict, an explicit MINIMUM_COLUMNS
                   tuple — so the two artifacts read alike on disk.
```

### Metrics — `src/hg/metrics/`

```
existing module:   hg.metrics.core, hg.metrics.dependence, hg.metrics.matrix
reusable as-is:    no — every function takes an ErrorMatrix (n_items x
                   n_members, anchor statuses, consensus masks)
needs extension:   no
new code required: hg.grounding.costs (loss, response bits, coordination
                   overhead, information efficiency, per-arm aggregation)
reason:            The M2 metrics measure agreement structure across ensemble
                   members. HG-v2 Phase 0 measures transmission cost against a
                   decision deficit. No shared quantity exists; a shim would be
                   longer than the ~150 lines of new code.
```

### Bootstrap / uncertainty — `src/hg/metrics/bootstrap.py`

```
existing module:   hg.metrics.bootstrap
reusable as-is:    BootstrapResult (the reporting dataclass) — yes, reused
                   directly, including .format() and .to_dict()
needs extension:   bootstrap_metric/paired_bootstrap are typed on ErrorMatrix
                   and resample via matrix.resample_items
new code required: bootstrap_mean over a plain float sequence (~15 lines)
reason:            Reusing the *result* type keeps CI reporting identical
                   across experiments, which matters when both appear in the
                   same results tree. Reusing the resampler would mean giving
                   ErrorMatrix a second, episode-shaped meaning.
```

### Figures — `src/hg/report/figures.py`

```
existing module:   hg.report.figures
reusable as-is:    the pattern and the _have_matplotlib guard
needs extension:   yes — one generic line-plot helper added to this module
new code required: plot_grounding_sweep (in the existing module, not a new one)
reason:            The matplotlib-optional / always-write-the-CSV behaviour is
                   a deliberate repository property and must not be
                   re-implemented per experiment. The new helper follows it.
```

### Report rendering / decision gates — `src/hg/report/decision.py`

```
existing module:   hg.report.decision (decision_gate, evaluate_predictions,
                   render_report)
reusable as-is:    no
needs extension:   no
new code required: a small report renderer in experiments/hg_v2/analyze.py
reason:            decision_gate maps M2 outcomes A/B/C onto predictions P1-P6
                   about clean consensus failure. HG-v2's gates (G0-G5) are a
                   different ladder. Keeping them out of src/ until Phase 1
                   avoids inventing a framework for one table (plan §30).
```

### Simulator / task generation — `src/hg/inference/simulator.py`, `src/hg/tasks/synthetic.py`

```
existing module:   hg.inference.simulator, hg.tasks.synthetic
reusable as-is:    no
needs extension:   no
new code required: hg.grounding.synthetic (index task + Phase 0 policies)
reason:            The existing simulator answers "which member gets item i
                   wrong, given latent difficulty and lineage?" — it models an
                   ensemble's error structure. Phase 0 needs an information
                   asymmetry instead: the AI holds K, the source holds
                   X_1..X_n, and J = X_K. Nothing in the existing generator
                   expresses two-sided private state.
                   Convention reused: a task version string, seeded
                   numpy.default_rng, and fixtures that are obviously
                   synthetic.
```

### Inference backends — `src/hg/inference/`

```
existing module:   hg.inference.base / hf / vllm_backend / runner
reusable as-is:    not used in Phase 0 (no LLM, by design — plan §5)
needs extension:   deferred to Phase 2
new code required: none now
reason:            Phase 2 should drive the frozen primary LM through the
                   existing Backend interface and its canonicalizer rather
                   than opening a second model-loading path. Nothing in Phase 0
                   should pre-empt that; the episode record is the seam.
```

### Routing — `src/hg/routing/`

```
existing module:   hg.routing.strategies (the localization race)
reusable as-is:    no — it ranks *items to audit* under a label budget; HG-v2
                   routes *one item* between ACT and ASK
needs extension:   Phase 1 adds hg.routing.decision_state and a grounding
                   router alongside it, not inside it
new code required: none in Phase 0
reason:            Both are "spend external effort where it pays", but the
                   decision variable differs (which items vs. which route), so
                   merging them now would produce an abstraction neither use
                   fits. Phase 0 uses fixed policies and needs no router.
```

### Fine-tuning — `src/hg/finetune/`

```
existing module:   hg.finetune.lora, hg.finetune.budget
reusable as-is:    not used before gate G5
new code required: none
reason:            Plan §2.1 and §27: no custom training until the frozen-LM
                   effect is established.
```

### Seeds / reproducibility / provenance

```
existing module:   hg.io.git_commit, hg.io.run_id, hg.config.resolved_dict,
                   numpy.random.default_rng everywhere, RunManifest
reusable as-is:    git_commit, run_id, the seeded-rng convention — yes
needs extension:   RunManifest fields (condition, dataset_config, members,
                   train_overlap) are ensemble-specific
new code required: a plain-dict manifest written with write_json
reason:            Plan §25's list (resolved config, commit, timestamp, seed,
                   task version, environment) is satisfied without borrowing
                   fields whose names would be lies here (there is no
                   condition, no member, no adapter).
```

### CLI — `src/hg/cli.py`

```
existing module:   hg.cli
reusable as-is:    n/a
needs extension:   deferred
new code required: none — HG-v2 ships experiments/hg_v2/run_phase0.py and
                   analyze.py, mirroring experiments/m2_llm/
reason:            `hg <stage> --config` is typed on ExperimentConfig
                   throughout. Adding an `hg ground` subcommand is worth doing
                   once the config object has stopped moving; a per-experiment
                   script is the existing convention for a new phase anyway.
```

### Tests

```
existing module:   tests/conftest.py, tests/test_*.py
reusable as-is:    the conventions — src on sys.path, generative fixtures whose
                   truth is known so tests assert recovery rather than
                   execution, one test module per src module
needs extension:   no; conftest's fixtures are all ErrorMatrix-shaped
new code required: tests/test_grounding.py
reason:            Plan §24. The index theorem is the one place in this
                   repository where the correct answer is known exactly, so the
                   test asserts the exact value, not a tolerance, wherever the
                   quantity is deterministic.
```

---

## Net new infrastructure

Phase 0: modules under `src/hg/grounding/` (channels, episodes, costs,
synthetic, config, phase0), one 3-line additive helper in `hg.config`, one
generic plot helper appended to `hg.report.figures`, and the
`experiments/hg_v2/` scripts.

Phase 1 added `src/hg/grounding/{noisy_index,phase1}.py`, the route-cost
section of `costs.py`, and `src/hg/routing/{decision_state,grounding_router}.py`.

Phase 2 added `src/hg/grounding/{route_choice,phase2}.py`, `src/hg/queries/`
(the plan's §3 layout, including `lm_query_synthesizer.py`),
`src/hg/inference/grounding_simulator.py`, two prompt versions in
`hg.inference.prompts`, and `scripts/spark_deploy.sh`.

One existing module needed a genuine fix rather than an extension:
`hg.inference.hf._label_logprob` scored `" " + label`, which measures the wrong
token after a chat template and so silently corrupted `scoring_mode: rank` --
the default -- for any chat-templated model. Found only by running a real
checkpoint on the Spark. This affected the M2-LLM path too.

**No existing module is modified in a way that changes its behaviour**, so the
M2-LLM experiment and its 162 tests are untouched.

## Deviations from the plan's suggested layout, and why

1. **`configs/experiments/hg_v2_phase0_index.yaml`, not
   `experiments/hg_v2/configs/phase0_index.yaml`.** The plan (§3) says to
   prefer repository conventions over its own suggested tree, and every config
   in this repo lives under `configs/{models,datasets,experiments}/`.
2. **A reduced route enum** (`ACT`, `ASK_GENERIC`, `ASK_TARGETED`), which §4.2
   explicitly permits. `THINK/RETRIEVE/OBSERVE/DELIBERATE` are not defined yet
   precisely so that nobody can quietly collapse `ASK` and `DELIBERATE`
   (§20) before the distinction has an experiment attached to it.
3. **A fourth Phase 0 arm, `ask_targeted_ablated`.** §24 requires a joint
   sufficiency test — remove the source's knowledge of the queried variable and
   perfect recovery must become impossible. Making it a first-class arm rather
   than only a unit test means the negative control appears in the results
   table, where it is much harder to forget.
4. **`hg.grounding.costs` rather than extending `hg.metrics`.** See above: no
   shared quantity with the ErrorMatrix metrics.
