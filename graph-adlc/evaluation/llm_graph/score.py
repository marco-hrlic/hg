"""Score replay predictions against the oracle. The only module that reads ``oracle/``.

    python evaluation/llm_graph/score.py --run evaluation/llm_graph/out/scripted
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _ratio(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def score_impact(pred: dict[str, Any] | None, gold: dict[str, Any]) -> dict[str, Any]:
    if pred is None:
        return {"measured": False}
    origin = gold["origin"]
    targets = set(pred.get("targets", [])) - {origin}
    nec = {n["target"] for n in gold["necessary"]}
    crit = {n["target"] for n in gold["necessary"] if n["critical"]}
    unk = set(gold.get("unknown", []))
    tp, fn = targets & nec, nec - targets
    fp = targets - nec - unk
    return {"measured": True, "predicted": sorted(targets), "tp": sorted(tp), "fp": sorted(fp), "fn": sorted(fn),
            "unknown_flagged": sorted(targets & unk), "precision": _ratio(len(tp), len(tp) + len(fp)),
            "recall": _ratio(len(tp), len(nec)), "critical_missed": sorted(crit - targets),
            "critical_total": len(crit)}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def score_construction(graph: dict[str, Any] | None, gold: dict[str, Any]) -> dict[str, Any]:
    if not graph:
        return {"measured": False}
    nodes = {n["id"]: n for n in graph["nodes"] if not n.get("deleted")}
    edges = [e for e in graph["edges"] if not e.get("deleted")]
    rels = {(e["src"], e["relation"], e["dst"]): e for e in edges}
    ents = {e["id"] for e in gold["entities"]}
    opt_ents = set(gold.get("optional_entities", []))
    req = {tuple(r) for r in gold["relations_required"]}
    opt = {tuple(r) for r in gold.get("relations_optional", [])}
    extra_nodes = sorted(set(nodes) - ents - opt_ents)
    extra_rels = sorted(r for r in rels if r not in req | opt)
    titles = Counter((n["kind"], _norm(n["title"])) for n in nodes.values())
    hidden = []
    for h in gold.get("hidden_relations", []):
        r = tuple(h["relation"])
        hidden.append({"relation": list(r), "recovered": r in rels,
                       "status": rels[r]["status"] if r in rels else None})
    return {"measured": True, "entities_expected": len(ents), "entities_found": len(ents & set(nodes)),
            "entity_recall": _ratio(len(ents & set(nodes)), len(ents)), "missing_entities": sorted(ents - set(nodes)),
            "extra_nodes": extra_nodes,
            "relations_required": len(req), "relations_found": len(req & set(rels)),
            "relation_recall": _ratio(len(req & set(rels)), len(req)), "missing_relations": sorted(req - set(rels)),
            "relations_not_in_oracle": [list(r) for r in extra_rels],
            "hypothesized_edges": sum(e["status"] == "hypothesized" for e in edges),
            "duplicate_titles": sum(c - 1 for c in titles.values() if c > 1), "hidden_relations": hidden}


def guardrail_events(run_dir: Path, project: str, arm: str) -> dict[str, Any]:
    p = run_dir / project / arm / "trace.json"
    if not p.exists():
        return {}
    t = json.loads(p.read_text())
    codes: Counter = Counter()
    verdicts: Counter = Counter()
    for ev in t.get("events", []):
        steps = [ev.get("proposal_step")] + list((ev.get("challenge") or {}).get("steps", {}).values())
        for st in steps:
            for o in ((st or {}).get("validation") or {}).get("ops", {}).values():
                verdicts[o["verdict"]] += 1
                for f in o["findings"]:
                    codes[f.split(":", 1)[1]] += 1
    reconcile = [l for ev in t.get("events", []) for l in ev.get("lifecycle_after", []) if l.get("action") == "reconcile"]
    return {"op_verdicts": dict(sorted(verdicts.items())), "finding_codes": dict(sorted(codes.items())),
            "reconcile": [{k: r.get(k) for k in ("work_id", "targets", "status", "obligations_since_baseline", "stale_evidence")}
                          for r in reconcile],
            "probe_by_event": [{"event": ev["event"], "before": (ev.get("probe_before") or {}).get("targets"),
                                "after": (ev.get("probe_after") or {}).get("targets")} for ev in t.get("events", [])
                               if "probe_before" in ev]}


def score_run(run_dir: Path) -> dict[str, Any]:
    preds = json.loads((run_dir / "predictions.json").read_text())
    out: dict[str, Any] = {"mode": preds["mode"], "evidential_status": preds["evidential_status"], "projects": {}}
    for project, arms in preds["projects"].items():
        oracle = json.loads((HERE / "oracle" / f"{project}.json").read_text())
        res: dict[str, Any] = {"oracle_version": oracle["oracle_version"], "labeler": oracle["labeler"], "arms": {}}
        for arm, p in arms.items():
            pr = (p.get("prediction") or {}).get("E5")
            neg = (p.get("negative") or {}).get("E4") or {}
            if "engine_semantic_probe" in neg:
                neg_targets = neg["engine_semantic_probe"]["targets"]
            else:
                neg_targets = sorted({t for ts in (neg.get("required_actions") or {}).values() for t in ts})
            r = {"impact_E5": score_impact(pr, oracle["impact"]["E5"]),
                 "negative_E4": {"measured": bool(neg), "false_alarm_targets": neg_targets,
                                 "false_alarms": len(neg_targets), "held_for_review": neg.get("deferred_ops", []),
                                 "no_change_records": neg.get("no_change_records", [])} if neg else {"measured": False},
                 "construction": score_construction(p.get("final_graph"), oracle["construction"]) if arm != "H" else {"measured": False, "note": "reference graph"},
                 "costs": p.get("costs"), "client": p.get("client"),
                 "guardrails": guardrail_events(run_dir, project, arm)}
            ep = (p.get("prediction") or {}).get("exploratory_pruning")
            if ep:
                crit = {n["target"] for n in oracle["impact"]["E5"]["necessary"] if n["critical"]}
                pruned = sorted({x["consumer_id"] for x in ep["engine_pruning_with_all_evidence"]["pruned"]})
                r["exploratory_pruning"] = {
                    "engine_pruned_with_all_evidence": pruned,
                    "unsafe_prunes": sorted(set(pruned) & crit),
                    "gate_admitted": [a["evidence_id"] for a in ep["evidence_gate"]["admissible"]],
                    "gate_refused": ep["evidence_gate"]["refused"],
                    "pruned_with_gate": sorted({x["consumer_id"] for x in ep["engine_pruning_with_gate_admitted_evidence"]["pruned"]}),
                }
            res["arms"][arm] = r
        out["projects"][project] = res
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(HERE / "out" / "scripted"))
    a = ap.parse_args(argv)
    run = Path(a.run)
    m = score_run(run)
    (run / "metrics.json").write_text(json.dumps(m, indent=1, sort_keys=True) + "\n")
    print(f"wrote {run / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
