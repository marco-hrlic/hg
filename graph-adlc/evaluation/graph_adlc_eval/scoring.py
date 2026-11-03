"""Pair- and target-level scoring against an oracle version (EVALUATION_PROTOCOL.md §C/§D)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

Pair = tuple[str, str]


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else round(num / den, 4)


def f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None or (p + r) == 0:
        return None if p is None or r is None else 0.0
    return round(2 * p * r / (p + r), 4)


def load_oracle(oracle_dir: Path, version: str = "v1") -> dict[str, dict[str, Any]]:
    """Load oracle v1 and, for v1.1, apply the logged post-unblinding review decisions."""
    base = {}
    for f in sorted((oracle_dir / "v1").glob("SC-*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        base[rec["scenario_id"]] = rec
    if version == "v1":
        return base
    if version != "v1.1":
        raise ValueError(version)
    review = oracle_dir / "review_v1_1.jsonl"
    for ln in review.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        d = json.loads(ln)
        rec = base[d["scenario_id"]]
        key = (d["target_id"], d["obligation"])
        rec["necessary_checks"] = [c for c in rec["necessary_checks"] if (c["target_id"], c["obligation"]) != key]
        rec["unknown_checks"] = [c for c in rec["unknown_checks"] if (c["target_id"], c["obligation"]) != key]
        if d["decision"] == "necessary":
            rec["necessary_checks"].append({"target_id": d["target_id"], "obligation": d["obligation"],
                                            "severity": d["severity"], "reason": d["rationale"], "added_in": "v1.1"})
        elif d["decision"] == "unknown":
            rec["unknown_checks"].append({"target_id": d["target_id"], "obligation": d["obligation"],
                                          "reason": d["rationale"], "added_in": "v1.1"})
    return base


def load_predictions(path: Path) -> list[dict[str, Any]]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def score(predictions: list[dict[str, Any]], oracle: dict[str, dict[str, Any]],
          scenarios: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    by = defaultdict(set)
    for p in predictions:
        by[(p["scenario_id"], p["method"])].add((p["target_id"], p["obligation"]))
    per_scenario, totals = [], {}
    for m in methods:
        agg = defaultdict(int)
        fam = defaultdict(lambda: defaultdict(int))
        macro_r, macro_tr, crit = [], [], []
        neg = {"cases": 0, "cases_with_false_alarm": 0, "false_alarm_pairs": 0}
        for sc in scenarios:
            sid = sc["id"]
            o = oracle[sid]
            gold: set[Pair] = {(c["target_id"], c["obligation"]) for c in o["necessary_checks"]}
            unk: set[Pair] = {(c["target_id"], c["obligation"]) for c in o["unknown_checks"]}
            sev = {(c["target_id"], c["obligation"]): c["severity"] for c in o["necessary_checks"]}
            pred = by[(sid, m)]
            P, G = pred - unk, gold - unk
            tp, fp, fn = P & G, P - G, G - P
            unk_pred, unk_missed = pred & unk, unk - pred
            # target level (unknown targets: those whose every oracle mention is unknown)
            unk_t = {t for t, _ in unk} - {t for t, _ in G}
            Pt, Gt = {t for t, _ in P} - unk_t, {t for t, _ in G}
            ttp, tfp, tfn = Pt & Gt, Pt - Gt, Gt - Pt
            row = {"scenario_id": sid, "method": m, "family": sc["family"], "tags": sc["tags"],
                   "gold": len(G), "predicted": len(P), "tp": len(tp), "fp": len(fp), "fn": len(fn),
                   "unknown_predicted": len(unk_pred), "unknown_not_predicted": len(unk_missed),
                   "precision": _ratio(len(tp), len(tp) + len(fp)), "recall": _ratio(len(tp), len(tp) + len(fn)),
                   "t_tp": len(ttp), "t_fp": len(tfp), "t_fn": len(tfn),
                   "t_precision": _ratio(len(ttp), len(ttp) + len(tfp)), "t_recall": _ratio(len(ttp), len(ttp) + len(tfn)),
                   "missed": sorted(f"{t}:{ob}[{sev[(t, ob)]}]" for t, ob in fn),
                   "false_positives": sorted(f"{t}:{ob}" for t, ob in fp),
                   "critical_missed": sorted(f"{t}:{ob}" for t, ob in fn if sev[(t, ob)] == "critical")}
            row["f1"] = f1(row["precision"], row["recall"])
            per_scenario.append(row)
            for k in ("gold", "predicted", "tp", "fp", "fn", "unknown_predicted", "unknown_not_predicted",
                      "t_tp", "t_fp", "t_fn"):
                agg[k] += row[k]
                fam[sc["family"]][k] += row[k]
            agg["critical_gold"] += sum(1 for p in G if sev[p] == "critical")
            agg["critical_missed"] += len(row["critical_missed"])
            crit += [f"{sid} {x}" for x in row["critical_missed"]]
            if G:
                macro_r.append(row["recall"])
                macro_tr.append(row["t_recall"])
            else:
                neg["cases"] += 1
                if P:
                    neg["cases_with_false_alarm"] += 1
                    neg["false_alarm_pairs"] += len(P)
        tot = dict(agg)
        tot.update({
            "micro_precision": _ratio(agg["tp"], agg["tp"] + agg["fp"]),
            "micro_recall": _ratio(agg["tp"], agg["tp"] + agg["fn"]),
            "macro_recall_nonempty_gold": round(sum(macro_r) / len(macro_r), 4) if macro_r else None,
            "t_micro_precision": _ratio(agg["t_tp"], agg["t_tp"] + agg["t_fp"]),
            "t_micro_recall": _ratio(agg["t_tp"], agg["t_tp"] + agg["t_fn"]),
            "t_macro_recall_nonempty_gold": round(sum(macro_tr) / len(macro_tr), 4) if macro_tr else None,
            "negative_cases": neg, "critical_missed_list": crit,
            # unknown-label sensitivity
            "best_case": {"precision": _ratio(agg["tp"] + agg["unknown_predicted"],
                                              agg["tp"] + agg["unknown_predicted"] + agg["fp"]),
                          "recall": _ratio(agg["tp"] + agg["unknown_predicted"],
                                           agg["tp"] + agg["unknown_predicted"] + agg["fn"])},
            "worst_case": {"precision": _ratio(agg["tp"], agg["tp"] + agg["fp"] + agg["unknown_predicted"]),
                           "recall": _ratio(agg["tp"], agg["tp"] + agg["fn"] + agg["unknown_not_predicted"])},
            "by_family": {f: {**dict(v), "precision": _ratio(v["tp"], v["tp"] + v["fp"]),
                              "recall": _ratio(v["tp"], v["tp"] + v["fn"])} for f, v in sorted(fam.items())},
        })
        tot["micro_f1"] = f1(tot["micro_precision"], tot["micro_recall"])
        tot["t_micro_f1"] = f1(tot["t_micro_precision"], tot["t_micro_recall"])
        totals[m] = tot
    return {"per_scenario": per_scenario, "totals": totals}


def pruning_audit(predictions: list[dict[str, Any]], raw_dir: Path, oracle: dict[str, dict[str, Any]],
                  scenario_ids: list[str]) -> list[dict[str, Any]]:
    by = defaultdict(set)
    for p in predictions:
        by[(p["scenario_id"], p["method"])].add((p["target_id"], p["obligation"]))
    rows = []
    for sid in scenario_ids:
        raw_c = json.loads((raw_dir / f"{sid}.C.json").read_text(encoding="utf-8"))
        pruned = raw_c["impact"]["pruned"]
        removed = sorted(by[(sid, "B")] - by[(sid, "C")])
        added = sorted(by[(sid, "C")] - by[(sid, "B")])
        o = oracle[sid]
        gold = {(c["target_id"], c["obligation"]) for c in o["necessary_checks"]}
        unk = {(c["target_id"], c["obligation"]) for c in o["unknown_checks"]}
        if not (removed or added or pruned or raw_c["evidence_refs_passed"]):
            continue
        rows.append({
            "scenario_id": sid, "evidence_refs": raw_c["evidence_refs_passed"],
            "setup_evidence": [s for s in raw_c["setup"] if s["type"] == "evidence"],
            "pruned_boundaries": pruned, "added_by_C": [f"{t}:{ob}" for t, ob in added],
            "removed": [{"target_id": t, "obligation": ob,
                         "oracle": "required" if (t, ob) in gold else ("unknown" if (t, ob) in unk else "not_required")}
                        for t, ob in removed],
        })
        rows[-1]["pruning_misses"] = [r for r in rows[-1]["removed"] if r["oracle"] == "required"]
    return rows
