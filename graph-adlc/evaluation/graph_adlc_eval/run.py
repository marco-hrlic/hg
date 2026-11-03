"""Run all methods over the corpus and write normalized predictions.

    python -m graph_adlc_eval.run --dataset evaluation/scenarios --out evaluation/out

This module never reads ``evaluation/oracle``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from pathlib import Path
from typing import Any

import graph_adlc

from . import HARNESS_VERSION
from .dataset import Scenario, file_sha256, input_files, load_dataset, project_root
from .methods import run_document, run_graph
from .normalize import normalize

METHODS = ("A", "A2", "B", "C")


def load_config(root: Path) -> dict[str, Any]:
    with open(root / "evaluation" / "config.json", encoding="utf-8") as fh:
        return json.load(fh)


def run_method(s: Scenario, method: str, cfg: dict[str, Any]) -> dict[str, Any]:
    if method in ("A", "A2"):
        a = cfg["methods"]["A"]
        mcfg = {**a, **cfg["methods"][method]} if method == "A2" else a
        return run_document(s, method, mcfg, a["obligation_by_target_kind"])
    return run_graph(s, method)


def _dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def costs_for(s: Scenario, method: str, res: dict[str, Any], scored: list[dict[str, Any]]) -> dict[str, Any]:
    g = res["graph"]
    row = {"scenario_id": s.id, "method": method, "predicted_pairs": len(scored),
           "predicted_targets": len({p["target_id"] for p in scored}),
           "graph_nodes": len(g.live_nodes()), "graph_edges": len(g.live_edges()),
           "setup_graph_edits": sum(len(st.get("delta", {}).get("operations", [])) or 1
                                    for st in s.setup if st["type"] == "commit"),
           "proposal_ops": len(s.proposal["operations"]),
           "evidence_records_available": len(res["available_evidence"]),
           "evidence_records_used": 0}
    if method in ("A", "A2"):
        tr = res["raw"]["trace"]
        row.update({"context_sections": len(tr["context_sections"]), "context_chars": tr["context_chars"],
                    "document_chars": tr["document_chars"], "over_budget": len(tr["over_budget"])})
    else:
        nodes = {s_.get("origin_id") for s_ in res["predictions"]} | {p["target_id"] for p in res["predictions"]}
        for p in res["predictions"]:
            for eid in p["path_edge_ids"]:
                e = g.edges.get(eid)
                if e is not None:
                    nodes |= {e.src, e.dst}
        nodes.discard(None)
        row.update({"context_sections": len(nodes), "pruned_boundaries": len(res["raw"]["impact"]["pruned"]),
                    "evidence_records_used": len({p["evidence_id"] for p in res["raw"]["impact"]["pruned"]}),
                    "provisional_pairs": sum(p["provisional"] for p in scored),
                    "low_confidence_pairs": sum(p["confidence"] == "low" for p in scored)})
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="evaluation/scenarios")
    ap.add_argument("--out", default="evaluation/out")
    ap.add_argument("--repeats", type=int, default=3, help="timing repeats (timing.json only)")
    ap.add_argument("--write-documents", action="store_true",
                    help="also refresh the committed readable snapshots in <dataset>/documents")
    args = ap.parse_args(argv)
    root = project_root()
    dataset, out = Path(args.dataset).resolve(), Path(args.out).resolve()
    cfg = load_config(root)
    scenarios = load_dataset(dataset)
    equivalence = cfg["normalization"]["obligation_equivalence_map"]

    predictions, excluded, costs, timing = [], [], [], []
    hashes: dict[str, str] = {}
    for s in scenarios:
        for f in input_files(s):
            hashes[str(f.relative_to(root))] = file_sha256(f)
        for method in METHODS:
            res = run_method(s, method, cfg)
            times = [res["elapsed_s"]] + [run_method(s, method, cfg)["elapsed_s"] for _ in range(args.repeats - 1)]
            scored, dropped = normalize(s.id, res["predictions"], res["graph"], s.proposal, equivalence)
            for r in scored + dropped:
                r.update({"dataset_version": s.dataset_version, "scenario_sha256": s.sha256})
            predictions += scored
            excluded += dropped
            costs.append(costs_for(s, method, res, scored))
            timing.append({"scenario_id": s.id, "method": method, "median_s": statistics.median(times),
                           "runs": len(times)})
            _dump(out / "raw" / f"{s.id}.{method}.json", {"scenario_id": s.id, "method": method,
                                                          "predictions_raw": res["predictions"], **res["raw"]})
            if method == "A":
                text = res["document"] + "\n---\n\n" + res["change_request"]
                (out / "documents").mkdir(parents=True, exist_ok=True)
                (out / "documents" / f"{s.id}.md").write_text(text, encoding="utf-8")
                if args.write_documents:
                    (dataset / "documents").mkdir(parents=True, exist_ok=True)
                    (dataset / "documents" / f"{s.id}.md").write_text(text, encoding="utf-8")
        for f in input_files(s):  # no-mutation check on inputs
            if file_sha256(f) != hashes[str(f.relative_to(root))]:
                raise RuntimeError(f"input file mutated during run: {f}")

    order = {m: i for i, m in enumerate(METHODS)}
    predictions.sort(key=lambda r: (r["scenario_id"], order[r["method"]], r["target_id"], r["obligation"]))
    excluded.sort(key=lambda r: (r["scenario_id"], order[r["method"]], r["target_id"], r["obligation"]))
    _jsonl(out / "predictions.jsonl", predictions)
    _jsonl(out / "excluded_predictions.jsonl", excluded)
    _jsonl(out / "costs.jsonl", costs)
    _dump(out / "timing.json", {"note": "wall-clock, environment dependent; not part of the determinism check",
                                "python": sys.version.split()[0], "platform": platform.platform(),
                                "rows": timing})
    cfg_bytes = (root / "evaluation" / "config.json").read_bytes()
    _dump(out / "run_manifest.json", {
        "harness_version": HARNESS_VERSION, "graph_adlc_version": graph_adlc.__version__,
        "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(), "methods": list(METHODS),
        "dataset_version": sorted({s.dataset_version for s in scenarios}), "scenarios": [s.id for s in scenarios],
        "input_sha256": dict(sorted(hashes.items())), "rng_seed": None,
        "predictions_sha256": file_sha256(out / "predictions.jsonl")})
    print(f"{len(scenarios)} scenarios x {len(METHODS)} methods -> {len(predictions)} scored predictions, "
          f"{len(excluded)} excluded; written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
