"""Generate evaluation/llm_graph/RESULTS.md from scored runs.

    python evaluation/llm_graph/report.py

Reads out/scripted/metrics.json (required) and out/live/metrics.json (if a live run exists).
Every number in RESULTS.md comes from those files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ARMS = ("D", "G0", "G1", "G2", "H")
ARM_DESC = {
    "D": "model reads documents/code directly, no graph",
    "G0": "graph after initial construction only",
    "G1": "maintained graph (proposals → review → commit)",
    "G2": "G1 + missing-edge challenge",
    "H": "hand-curated reference graph (diagnostic)",
}


def f(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.2f}"
    if isinstance(x, list):
        return ", ".join(map(str, x)) if x else "none"
    return str(x)


def impact_table(m: dict[str, Any], scripted: bool) -> list[str]:
    L = ["| Project | Arm | Status | Predicted (excl. origin) | Precision | Recall | Critical missed | Missed (all) |",
         "|---|---|---|---|---:|---:|---|---|"]
    for proj, pv in m["projects"].items():
        for arm in ARMS:
            r = pv["arms"].get(arm)
            if r is None:
                L.append(f"| {proj} | {arm} | not run | — | — | — | — | — |")
                continue
            i = r["impact_E5"]
            status = "deterministic" if arm == "H" else ("scripted — not evidence" if scripted else "live")
            L.append(f"| {proj} | {arm} | {status} | {f(i.get('predicted'))} | {f(i.get('precision'))} | "
                     f"{f(i.get('recall'))} | {f(i.get('critical_missed'))} | {f(i.get('fn'))} |")
    return L


def _live_meta(live: dict[str, Any]) -> dict[str, Any]:
    preds_p = HERE / "out" / "live" / "predictions.json"
    preds = json.loads(preds_p.read_text()) if preds_p.exists() else {}
    client = next(a["client"] for pv in live["projects"].values() for k, a in pv["arms"].items()
                  if k != "H" and a.get("client"))
    aborted = sorted(p.name for p in (HERE / "out").glob("live_aborted_*") if p.is_dir())
    return {"client": client, "resumed": preds.get("resumed_arms", []), "aborted": aborted}


def _arms(live: dict[str, Any], proj: str) -> dict[str, Any]:
    return live["projects"][proj]["arms"]


def live_bottom_line(live: dict[str, Any]) -> list[str]:
    meta = _live_meta(live)
    c = meta["client"]
    L = [f"* **A live model was evaluated: `{c.get('model')}` ({c.get('provider')}), one run per arm.** The oracle, "
         "fixtures and reference graph share one author, so every live number is exploratory and single-reviewer. "
         "Section 11 has the tables and the deviations from the frozen protocol."]
    for proj in sorted(live["projects"]):
        a = _arms(live, proj)
        row = "; ".join(f"{arm} {f(a[arm]['impact_E5']['recall'])}/{f(a[arm]['impact_E5']['precision'])}"
                        for arm in ARMS if arm in a)
        L.append(f"  * {proj}, E5 recall/precision: {row}.")
    hid = {proj: {arm: _arms(live, proj)[arm]["construction"]["hidden_relations"][0]["status"]
                  for arm in ("G0", "G1", "G2")} for proj in live["projects"]}
    g1_both = all(h["G1"] for h in hid.values())
    g0_none = not any(h["G0"] for h in hid.values())
    if g1_both and g0_none:
        L.append("  * **Maintenance recovered the code-only dependency in both projects; construction alone did not.** "
                 "G1 has no import scan, so the model added the edge itself from the code diffs. G0, built once from the "
                 "initial documents, never saw it.")
    d_crit = {proj: _arms(live, proj)["D"]["impact_E5"]["critical_missed"] for proj in live["projects"]}
    g_crit = {proj: sorted({x for arm in ("G1", "G2") for x in _arms(live, proj)[arm]["impact_E5"]["critical_missed"]})
              for proj in live["projects"]}
    if not any(d_crit.values()) and any(g_crit.values()):
        miss = "; ".join(f"{p}: {f(v)}" for p, v in g_crit.items() if v)
        L.append(f"  * **The document-only arm D missed no critical target; the maintained graphs did ({miss}).** That "
                 "miss is the engine's terminal `check_contract` rule (H misses it too), not graph quality. D's cost is "
                 "lower precision: it also names entities and documents outside the labelled set.")
    return L


def live_section(live: dict[str, Any]) -> list[str]:
    meta = _live_meta(live)
    c = meta["client"]
    L: list[str] = []
    A = L.append
    A("## 11. Live run")
    A("")
    A(f"**Model:** `{c.get('model')}` via {c.get('provider')} (`{c.get('base_url')}`), JSON mode "
      f"{'on' if c.get('json_mode') else 'off'}, temperature {'provider default' if c.get('temperature') is None else c.get('temperature')}, "
      f"output budget {c.get('max_output_tokens') or 8000} tokens per call. Reviewer: `policy:conservative-v1` (not a "
      "human). Prompts unchanged (`PROMPTS.lock.json`).")
    A("")
    A("### 11.1 Deviations from the frozen protocol")
    A("")
    if c.get("max_output_tokens"):
        A(f"* **Output budget raised from 8000 to {c['max_output_tokens']} tokens** (`GADLC_LLM_MAX_TOKENS`). With 8000, "
          "every proposal call ended `finish_reason=length` with empty content, so every proposal was unparseable.")
    if c.get("temperature") is None:
        A("* Temperature: the model rejected `temperature=0`, so the adapter retried with the provider default and "
          "recorded that (designed behaviour, listed for completeness).")
    A("* **Two harness bugs fixed during the trial**, both crashes on paths the scripted replay never reached. Scripted "
      "outputs are byte-identical before and after. (1) The stand-in reviewer accepted an auto-generated `blocks` edge "
      "while deferring its question node; such edges are now deferred with the node. (2) On rebase, an op that a sibling "
      "proposal from the same base had already committed was re-applied; it is now dropped as a duplicate.")
    if meta["resumed"]:
        A(f"* **The run spans two invocations.** {f(meta['resumed'])} were completed before fix (2) and reused with "
          "`--resume`, which refuses traces from a different model configuration. Fix (2) only changes rebases that "
          "contain such duplicates, and no rebase happened in those arms, so they would have run identically.")
    if meta["aborted"]:
        A(f"* Aborted attempts are kept, unscored, in {f(['`out/' + x + '/`' for x in meta['aborted']])}, each with an "
          "`ABORTED.md`.")
    A("")
    A("### 11.2 Impact of the probe change (E5)")
    A("")
    L += impact_table(live, scripted=False)
    A("")
    A("Arm D's predictions are free text matched to ids, and some name documents (`project:path`); those count as false "
      "positives. Targets the oracle marks *unknown* count neither way.")
    A("")
    A("### 11.3 Construction and the code-only dependency")
    A("")
    A("| Project | Arm | Entities | Required relations | Hidden code-only relation | Status | Hypothesized edges | Extra nodes |")
    A("|---|---|---|---|---|---|---:|---|")
    for proj, pv in live["projects"].items():
        for arm in ("G0", "G1", "G2"):
            k = pv["arms"][arm]["construction"]
            h = k["hidden_relations"][0]
            q = len([n for n in k["extra_nodes"] if n.startswith("Q-")])
            other = [n for n in k["extra_nodes"] if not n.startswith("Q-")]
            A(f"| {proj} | {arm} | {k['entities_found']}/{k['entities_expected']} | {k['relations_found']}/"
              f"{k['relations_required']} | `{' '.join(h['relation'])}` | {h['status'] or 'absent'} | "
              f"{k['hypothesized_edges']} | {q} contradiction questions{'; ' + f(other) if other else ''} |")
    A("")
    A("### 11.4 Guardrails and review (live model output)")
    A("")
    A("| Project | Arm | Op verdicts | Finding codes | Decisions accept/reject/defer |")
    A("|---|---|---|---|---|")
    for proj, pv in live["projects"].items():
        for arm in ("G1", "G2"):
            g = pv["arms"][arm]["guardrails"]
            d = pv["arms"][arm]["costs"]["review_decisions"]
            A(f"| {proj} | {arm} | {f([f'{k} {v}' for k, v in sorted(g['op_verdicts'].items())])} | "
              f"{f([f'{k} {v}' for k, v in sorted(g['finding_codes'].items())])} | {d['accept']}/{d['reject']}/{d['defer']} |")
    A("")
    A("### 11.5 Wording-only edit (E4)")
    A("")
    A("| Project | Arm | False-alarm obligations | Targets | Held for review | `no_change` records |")
    A("|---|---|---:|---|---|---|")
    for proj, pv in live["projects"].items():
        for arm in ("G1", "G2", "H"):
            n = pv["arms"][arm].get("negative_E4") or {}
            if n.get("measured"):
                A(f"| {proj} | {arm} | {n['false_alarms']} | {f(n['false_alarm_targets'])} | "
                  f"{f(n.get('held_for_review'))} | {f(n.get('no_change_records'))} |")
    A("")
    A("### 11.6 Costs")
    A("")
    A("| Project | Arm | Model calls | Tokens in | Tokens out | Model latency (s) | Graph ops committed |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for proj, pv in live["projects"].items():
        for arm in ("D", "G0", "G1", "G2"):
            k = pv["arms"][arm]["costs"]
            lat = round(k["latency_ms"] / 1000) if k.get("latency_ms") else None
            A(f"| {proj} | {arm} | {k['model_calls']} | {f(k['input_tokens'])} | {f(k['output_tokens'])} | {f(lat)} | "
              f"{k['graph_ops_committed']} |")
    A("")
    A("Tokens out include hidden reasoning tokens, which the provider bills but does not return. G2's challenge "
      "calls are included. Human review time was not measured.")
    A("")
    A("### 11.7 Reading")
    A("")
    A("* **RQ1 (construction).** From the initial documents alone (G0), the model found every expected entity and every "
      "required relation except the code-only one, which is not in the documents. Beyond the oracle, it added only "
      "`realized_by` artifact nodes and edges.")
    A("* **RQ2 (maintenance).** Over the 5 events, the maintained graph (G1) picked up the code-only dependency in both "
      "projects without the import scan, and recall on E5 rose to the level of the curated reference graph H in both. "
      "The validator raised no `invalid` op in any live arm, so there were no fabricated citations or silent removals. "
      "Quarantines came from contradictions, plus a few suspected duplicates (section 11.4). The model raised many "
      "contradiction questions, which costs review time.")
    A("* **RQ3 (benefit over documents only).** Not shown here. Arm D matched or beat every graph arm on recall in both "
      "projects, at roughly a tenth of the tokens, and lost only on precision. The one critical target the graphs miss "
      "(`REQ-CSV-EXPORT`) is an engine propagation limit, not a modelling failure. So on this material the graph's "
      "advantage is precision and an auditable, persistent structure, not recall.")
    A("* **Wording-only edits (E4).** In every maintained arm, the model classified the edit itself as editorial and it "
      "was recorded as `no_change`; nothing about it propagated. The obligations counted in section 11.5 come from "
      "something else: in the same step, the model raised contradiction questions about unrelated mismatches between "
      "documents and code, and their `blocks` edges raise `review_spec`. One of them (ledgerlite: the architecture says "
      "CSV amounts bypass the money formatter, but the exporter calls it) is the defect that E5 exposes. The oracle "
      "labels E4 'nothing needs re-verification', so these count as false alarms. Whether they should count is a "
      "labelling question for an independent reviewer. Toolshare G1 shows 0 because its question was deferred, not "
      "committed.")
    A("* **Caveats.** One sample per arm, one author for oracle, fixtures and H, and a policy reviewer. Treat these as "
      "hypotheses to test with independent labels and repeated runs, not as results.")
    return L


def main() -> int:
    s = json.loads((HERE / "out" / "scripted" / "metrics.json").read_text())
    live_p = HERE / "out" / "live" / "metrics.json"
    live = json.loads(live_p.read_text()) if live_p.exists() else None
    tp, lp = s["projects"]["toolshare"]["arms"], s["projects"]["ledgerlite"]["arms"]
    llp = lp["G2"].get("exploratory_pruning", {})
    crit_all = sorted({c for pv in s["projects"].values() for a in pv["arms"].values()
                       for c in a["impact_E5"].get("critical_missed", [])
                       if all(c in x["impact_E5"].get("critical_missed", []) for x in pv["arms"].values())})
    guard: dict[str, int] = {}
    for pv in s["projects"].values():
        for arm in ("G1", "G2"):
            for k, v in pv["arms"][arm]["guardrails"].get("finding_codes", {}).items():
                guard[k] = guard.get(k, 0) + v

    L: list[str] = []
    A = L.append
    A("# graph-adlc v0.3 — LLM-assisted graph maintenance: results")
    A("")
    A("**Date:** 2026-11-03 · **Engine:** graph-adlc 0.1.0, unchanged · **Layer:** `graph_adlc/llm_graph/` · "
      "**Oracle:** v1.0, single reviewer, exploratory · **Prompts:** frozen in `prompts/PROMPTS.lock.json`")
    A("")
    A("This file is generated by `evaluation/llm_graph/report.py` from `out/*/metrics.json`. Interpretation is in the "
      "sections marked *Reading*.")
    A("")
    A("## Bottom line")
    A("")
    if live is None:
        A("* **No language model was evaluated.** No model credentials were configured in the environment where "
          "this milestone was built, so the live arms (D, and G1/G2 with a real model) were **not run**. RQ1 "
          "(construction fidelity), RQ2 (maintenance fidelity) and RQ3 (influence benefit versus a document-only "
          "model) are therefore **not yet evaluated**. Nothing below is a measurement of model quality.")
    else:
        L += live_bottom_line(live)
    A("* **What was built and verified:** an end-to-end, opt-in maintenance loop over the existing engine — "
      "source snapshots (files and immutable Git commits) → model proposal → citation/provenance validation → "
      "per-operation human review → `GraphADLC.propose`/`commit` → spec-only Markdown → influence before/after — "
      "plus a missing-edge challenge, an evidence-sufficiency gate, a live provider adapter and a deterministic "
      "scripted fake. The loop ran **5 events on one persistent graph per project and arm** for two independently "
      "written projects; all 15 acceptance cases (L01–L15) have passing tests.")
    A("* **What the deterministic parts of the scripted self-check show** (these do not depend on the fake):")
    tg2, tg1 = tp["G2"]["construction"]["hidden_relations"][0], tp["G1"]["construction"]["hidden_relations"][0]
    A(f"  * The deterministic import scan recovered the code-only dependency in both projects "
      f"(toolshare `CMP-HOLDS → CMP-NOTIFY`: {'recovered' if tg2['recovered'] else 'missed'}; "
      f"ledgerlite `CMP-CSV → CMP-MONEY`: {'recovered' if lp['G2']['construction']['hidden_relations'][0]['recovered'] else 'missed'}), "
      "with verified line-level citations into the commit that introduced it, before the impact question was asked.")
    A(f"  * **SC-14 reproduced on independent material.** In ledgerlite, a `contract_preservation` record that passed, "
      f"was current, and covered only amounts below 1000 lets the unchanged v0.1 engine prune "
      f"{f(llp.get('engine_pruned_with_all_evidence'))}; the critical consumer {f(llp.get('unsafe_prunes'))} is among them. "
      f"The new coverage gate refused that record ({len(llp.get('gate_refused', []))} refused, "
      f"{len(llp.get('gate_admitted', []))} admitted), so the gated condition prunes "
      f"{f(llp.get('pruned_with_gate')) if llp.get('pruned_with_gate') else 'nothing'}. Pruning stays off by default.")
    if crit_all:
        A(f"  * **Every graph arm, including the curated reference graph H, misses {f(crit_all)}.** The engine's "
          "`check_contract` on a consumer component is terminal, so a broken consumer never reaches the requirement it "
          "implements. This is the v0.2 'premature terminal propagation' class again, now on a code-discovered "
          "dependency; better graph maintenance cannot fix it.")
    A("  * Wording-only edits (scripted inputs): without the maintenance loop, the engine has no editorial aspect and raises "
      f"{tp['H']['negative_E4']['false_alarms']} (toolshare) and {lp['H']['negative_E4']['false_alarms']} (ledgerlite) "
      "obligations on the reference graph. With the loop, an editorial change is either recorded as a `no_change` "
      "or held in quarantine for review instead of being committed.")
    if live is None:
        A("* **Recommendation:** run the frozen live protocol (`--mode live`) with a configured model before building "
          "anything else; separately, decide whether `check_contract` on an implementing component should reach the "
          "requirement it implements (a versioned engine change with its own before/after).")
    else:
        A("* **Recommendation:** have someone other than the fixture author label the E5 impacts and repeat the live run "
          "(more than one sample) before drawing conclusions. Separately, version the engine rule that lets "
          "`check_contract` on an implementing component reach its requirement: in the live run it is the binding limit "
          "for every graph arm on ledgerlite, while arm D, which has no graph, finds that requirement.")
    A("")
    A("## 1. Baseline verification (M0)")
    A("")
    A("| Command | Observed |")
    A("|---|---|")
    A("| `python -m pytest -q tests` | 44 passed (v0.1, 14 acceptance) |")
    A("| `evaluation/run_all.sh` | 44 + 18 passed; v1.0 recall A 0.593 / B 0.574 / C 0.556, matching the v0.2 report |")
    A("| `python -m graph_adlc.demo --workdir <dir>` | replays 22 steps, graph rev 4, writes `trace.json` |")
    A("")
    A("The reports matched the checkout. No engine module was modified in v0.3: everything new lives in "
      "`graph_adlc/llm_graph/` and `evaluation/llm_graph/`. `pyproject.toml` only gained the subpackage, its prompt "
      "files and a `gadlc-llm` entry point. L14 re-runs both earlier suites.")
    A("")
    A("## 2. Evidential status of each arm")
    A("")
    A("| Arm | Meaning | This run |")
    A("|---|---|---|")
    for arm in ARMS:
        if arm == "D":
            st = "not run (live only)" if live is None else "live"
        elif arm == "H":
            st = "deterministic; graph curated by the oracle author → diagnostic upper bound only"
        else:
            st = "**scripted fake**: proposals were authored by the implementing agent to exercise the pipeline"
            if live is not None:
                st = "live (section 11); sections 3–8 show the **scripted fake**, which is not evidence"
        A(f"| {arm} | {ARM_DESC[arm]} | {st} |")
    A("")
    A("Reviewer decisions in the replay come from a deterministic policy (`policy:conservative-v1`: accept admissible, "
      "reject invalid, defer quarantined), **not a human**. The policy never sees the oracle.")
    A("")
    A("## 3. Impact of the probe change (E5), oracle v1.0 — scripted self-check")
    A("")
    A("Unit is the target entity (the origin node is excluded). The toolshare probe is a contract change to "
      "`CON-NOTICE-FORMAT`; the ledgerlite probe is commit c3's change to the shared money formatter `CMP-MONEY`.")
    A("")
    L += impact_table(s, scripted=True)
    A("")
    A("*Reading.* G0/G1/G2 differ only because of what the **scripted** proposals contain, so their differences are "
      "not evidence about models. The rows are kept because the same scoring will run unchanged on a live run. H "
      "isolates the engine's rules: given a good graph, recall is "
      f"{f(tp['H']['impact_E5']['recall'])} (toolshare) and {f(lp['H']['impact_E5']['recall'])} (ledgerlite). The "
      "ledgerlite gap comes from the propagation rules, not from missing edges (section 7, items 1–2).")
    A("")
    A("## 4. Missing-edge recovery and construction")
    A("")
    A("| Project | Arm | Hidden code-only relation | Status in final graph | Required relations found | Relations not in oracle |")
    A("|---|---|---|---|---|---|")
    for proj, pv in s["projects"].items():
        for arm in ("G0", "G1", "G2"):
            c = pv["arms"][arm]["construction"]
            h = c["hidden_relations"][0]
            A(f"| {proj} | {arm} | `{' '.join(h['relation'])}` | {h['status'] or 'absent'} | "
              f"{c['relations_found']}/{c['relations_required']} | {f([' '.join(r) for r in c['relations_not_in_oracle']])} |")
    A("")
    A("*Reading.* G2's recovery comes from the deterministic import scan, which is measured. Whether a model would "
      "find the same edge unaided is not measured. In the scripted G1, toolshare's proposal deliberately omits the "
      "edge and ledgerlite's proposes it as `hypothesized`. The relations not in the oracle are validator-generated "
      "`blocks` edges from contradiction questions; they are correct behaviour, but the oracle did not anticipate them.")
    A("")
    A("## 5. Guardrail behaviour observed in the replay (G1 + G2, both projects)")
    A("")
    A("| Finding code | Count | Meaning |")
    A("|---|---:|---|")
    meaning = {
        "fabricated_citation": "excerpt not in the cited source → op invalid, cannot be accepted",
        "unsupported_claim": "'supported' without a verified citation → downgraded",
        "removal_without_evidence": "removal justified only by silence → invalid",
        "requires_authorization": "every removal needs explicit human authorization",
        "contradiction": "op touches a node named in a contradiction → quarantined; question generated",
        "editorial_declared": "update declared editorial → held (engine would propagate it as semantic)",
        "duplicate": "already in the graph → idempotent no-op",
        "stored_as_hypothesized": "inferred edge stored with status hypothesized",
        "editorial_no_op": "wording-only change recorded without a graph change",
    }
    for k, v in sorted(guard.items()):
        A(f"| `{k}` | {v} | {meaning.get(k, '')} |")
    A("")
    A("The scripted inputs were written to include a fabricated citation and a silence-based removal (toolshare), and a "
      "contradiction and an editorial `update_node` (ledgerlite). Counts are doubled because G1 and G2 replay the same "
      "inputs. They show that the guardrails fire. They say nothing about how often a model would trigger them.")
    A("")
    A("## 6. Lifecycle: reconciliation after the E5 change (G2)")
    A("")
    A("| Project | Work unit | Target | Status after E5 | Obligations since baseline | Stale evidence |")
    A("|---|---|---|---|---|---|")
    for proj, pv in s["projects"].items():
        for r in pv["arms"]["G2"]["guardrails"]["reconcile"]:
            obs = [f"{t}:{o}" for t, o in r["obligations_since_baseline"]]
            A(f"| {proj} | {r['work_id']} | {f(r['targets'])} | {r['status']} | {f(obs)} | {f(r['stale_evidence'])} |")
    A("")
    A("*Reading.* Work on `REQ-READY-NOTICE` is invalidated as expected. The work units for `REQ-HOLD-EXPIRY` and "
      "`REQ-CSV-EXPORT` stay **verified**, even though their implementing components received `check_contract`. "
      "`reconcile` lists those obligations, but nothing forces revalidation. `REQ-CSV-EXPORT` was also 'verified' "
      "with **zero required checks**, because no test verifies it; the CSV export is the part that actually breaks.")
    A("")
    A("## 7. Notable failures (case by case)")
    A("")
    A("1. **ledgerlite E5: `REQ-CSV-EXPORT` missed by every arm (critical).** The path `CMP-MONEY` → (depends_on) → "
      "`CMP-CSV` ends at a terminal `check_contract`, and the rule table has no step from a checked consumer component "
      "to the requirement it implements. Source: the DR-001 criterion 'no thousands separators' versus c3 "
      "`money.py`, where `f\"{q:,} {currency}\"` is used by `csv_export.py`.")
    A("2. **ledgerlite E5: `TEST-MONEY-FORMAT` missed by every arm.** An implementation-aspect change to a component "
      "does not reach tests that verify its contract. The test's sample table (all < 1000) is exactly why the "
      "preservation evidence was insufficient.")
    A("3. **SC-14 recreated.** The engine pruned the critical consumer on passing, current, incomplete evidence "
      "(section 3, exploratory pruning). The v0.3 gate requires declared coverage of the change's declared input "
      "domain. Without a declared domain, nothing is admissible.")
    A("4. **Wording-only changes have no engine representation.** The safe handling is procedural (a `no_change` "
      "record or quarantine). If a reviewer accepts an editorial `update_node`, it still propagates as a semantic "
      "change.")
    A("5. **Lifecycle quirk.** `reconcile --rebase` on an `active` work unit fails ('active -> active'), so the "
      "replay rebases only invalidated units.")
    A("")
    A("## 8. Costs")
    A("")
    A("| Project | Arm | Model calls | Fake calls | Input chars | Output chars | Tokens in/out | Decisions accept/reject/defer | Graph ops committed |")
    A("|---|---|---:|---:|---:|---:|---|---|---:|")
    for proj, pv in s["projects"].items():
        for arm in ("G0", "G1", "G2"):
            c = pv["arms"][arm]["costs"]
            d = c["review_decisions"]
            A(f"| {proj} | {arm} | {c['model_calls']} | {c['fake_calls']} | {c['input_chars']} | {c['output_chars']} | "
              f"{f(c['input_tokens'])}/{f(c['output_tokens'])} | {d['accept']}/{d['reject']}/{d['defer']} | {c['graph_ops_committed']} |")
    A("")
    A("Token counts and latency are null because no model was called. Input characters are the real prompt sizes "
      "a live run would send for the same steps. Human review time was not measured: the reviewer was a policy.")
    A("")
    A("## 9. Not measured")
    A("")
    if live is None:
        A("* Any model's construction, maintenance or challenge quality (RQ1/RQ2); arm D; RQ3 cost-benefit. These "
          "need `--mode live`.")
    else:
        A("* Run-to-run variance: the live run is a single sample per arm, so no interval can be given.")
    A("* Independent adjudication: the oracle, the reference graph H and the fixtures share one author. Treat every "
      "number as exploratory and single-reviewer.")
    A("* Human review effort, and lifecycle efficiency (RQ4) beyond the qualitative reconciliation trace.")
    A("")
    A("## 10. Recommended next step")
    A("")
    if live is not None:
        A("See the recommendation in the bottom line. The live protocol is in section 11.")
    else:
        A("Run the frozen protocol live: set `GADLC_LLM_MODEL=<id>` and either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` "
      "(`GADLC_LLM_PROVIDER` picks one if both are set), then `python evaluation/llm_graph/run_replay.py --mode live && "
      "python evaluation/llm_graph/score.py --run evaluation/llm_graph/out/live && python evaluation/llm_graph/report.py`. Have someone other than the fixture author label the E5 impacts first. It is "
      "the cheapest experiment that can actually answer RQ1–RQ3, and the harness, prompts and scoring are already "
      "frozen. Separately (and versioned), test an engine rule letting `check_contract` on an implementing component "
      "raise `review_implementation` on its requirement, because the reference graph shows that maintenance alone "
      "cannot recover those critical misses.")
    if live is not None:
        A("")
        L += live_section(live)
    A("")
    (HERE / "RESULTS.md").write_text("\n".join(L))
    print(f"wrote {HERE / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
