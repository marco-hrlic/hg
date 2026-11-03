"""Replay the v0.3 project histories.

    # deterministic (scripted fake model; no network; byte-reproducible outputs)
    python evaluation/llm_graph/run_replay.py --mode scripted

    # live model trial (explicit opt-in). Anthropic: $ANTHROPIC_API_KEY; OpenAI: $OPENAI_API_KEY
    # (+ GADLC_LLM_PROVIDER=openai if both keys are set)
    GADLC_LLM_MODEL=<model id> python evaluation/llm_graph/run_replay.py --mode live

Scripted and live outputs go to separate directories and are never merged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("GADLC_FIXED_TIME", "2026-11-03T00:00:00Z")

import replay_lib as rl  # noqa: E402
from graph_adlc.llm_graph import LLMNotConfigured, client_from_env  # noqa: E402

PROJECTS = ("toolshare", "ledgerlite")


def dump(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("scripted", "live"), default="scripted")
    ap.add_argument("--out", default=None, help="output dir (default evaluation/llm_graph/out/<mode>)")
    ap.add_argument("--projects", nargs="*", default=list(PROJECTS))
    ap.add_argument("--arms", nargs="*", default=None, help="subset of arms to run (default: all for the mode)")
    ap.add_argument("--influence-rules", choices=("v1", "v1.1"), default="v1",
                    help="engine influence rule set (default v1, the frozen engine); see graph_adlc/influence.py")
    ap.add_argument("--replay-from", default=None,
                    help="live only: a previous live run dir. Its recorded model outputs are reused whenever the prompt "
                         "is byte-identical; any other call goes to the live model. Isolates engine changes.")
    ap.add_argument("--replay-strict", action="store_true",
                    help="with --replay-from: fail instead of calling the live model when a prompt does not match")
    ap.add_argument("--resume", action="store_true",
                    help="reuse arms whose trace.json already exists in --out (a trace is written only when its arm "
                         "completes); refuses if a reused trace's model configuration differs from the current one")
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else HERE / "out" / a.mode
    live = a.mode == "live"
    if live:
        try:
            client = client_from_env()
        except LLMNotConfigured as e:
            print(f"live mode not configured: {e}", file=sys.stderr)
            return 2
        if not client.available():
            print(f"live mode not configured: ${client.api_key_env} is not set", file=sys.stderr)
            return 2
        factory = lambda project, bundle_fn: client  # noqa: E731
        arms = ("D", "G0", "G1", "G2", "H")
    else:
        factory = rl.scripted_client_factory
        arms = ("G0", "G1", "G2", "H")
    if a.replay_from and not live:
        print("--replay-from needs --mode live (calls that cannot be replayed go to the live model)", file=sys.stderr)
        return 2
    if a.arms:
        unknown = set(a.arms) - set(arms)
        if unknown:
            print(f"unknown arms for --mode {a.mode}: {sorted(unknown)}", file=sys.stderr)
            return 2
        arms = tuple(x for x in arms if x in a.arms)
    predictions = {"mode": a.mode, "evidential_status": (
        "live model run" if live else
        "PIPELINE SELF-CHECK: model outputs are scripted fixtures authored by the implementing agent; "
        "nothing here measures a language model"), "projects": {}}
    if a.influence_rules != "v1":
        predictions["influence_rules"] = a.influence_rules
    replay_log: dict[str, list[dict]] = {}
    resumed = []
    with rl.tempdir() as tmp:
        for project in a.projects:
            predictions["projects"][project] = {}
            for arm in arms:
                work = Path(tmp) / project
                done = out / project / arm / "trace.json"
                if a.resume and done.is_file():
                    trace = json.loads(done.read_text())
                    if live and trace.get("client") and _config(trace["client"]) != _config(client.describe()):
                        print(f"{project}/{arm}: stored trace used {_config(trace['client'])}, current client is "
                              f"{_config(client.describe())}; refusing to mix", file=sys.stderr)
                        return 2
                    resumed.append(f"{project}/{arm}")
                    print(f"{project}/{arm}: reused existing trace")
                else:
                    if (out / project / arm).exists():
                        shutil.rmtree(out / project / arm)  # partial output of an arm that did not complete
                    arm_client, arm_factory = client if live else None, factory
                    if a.replay_from:
                        src = Path(a.replay_from) / project / arm / "trace.json"
                        recs = rl.recorded_calls(json.loads(src.read_text())) if src.is_file() else {}
                        arm_client = rl.ReplayClient(recs, client, str(Path(a.replay_from)), strict=a.replay_strict)
                        arm_factory = lambda project, bundle_fn, c=arm_client: c  # noqa: E731
                    if arm == "D":
                        trace = rl.run_direct(project, arm_client, work)
                    else:
                        trace = rl.run_arm(project, arm, arm_factory, work, out / project / arm, live,
                                           influence_rules=a.influence_rules)
                    if a.replay_from:
                        trace["replay"] = {"from": str(Path(a.replay_from)), "calls": arm_client.served}
                        replay_log[f"{project}/{arm}"] = arm_client.served
                    dump(done, trace)
                predictions["projects"][project][arm] = {
                    "prediction": trace.get("prediction"), "negative": trace.get("negative"),
                    "final_graph": trace.get("final_graph"), "ledger": trace.get("ledger"),
                    "client": trace.get("client"),
                    "costs": cost_summary(trace)}
                print(f"{project}/{arm}: done")
    if resumed:
        predictions["resumed_arms"] = resumed
    if a.replay_from:
        predictions["replay"] = {"from": str(Path(a.replay_from)), "live_calls": {
            k: [c["call_key"] for c in v if c["served"] == "live"] for k, v in replay_log.items()},
            "replayed_calls": sum(c["served"] == "replayed" for v in replay_log.values() for c in v)}
    dump(out / "predictions.json", predictions)
    print(f"wrote {out}")
    return 0


def _config(client: dict) -> dict:
    """The parts of a live client description that must match for traces to belong to one run."""
    return {k: client.get(k) for k in ("provider", "model", "base_url", "max_output_tokens", "is_fake")}


def cost_summary(trace: dict) -> dict:
    calls = []
    decisions = {"accept": 0, "reject": 0, "defer": 0}
    ops_committed = 0
    for ev in trace.get("events", []):
        steps = [ev.get("proposal_step")] + list((ev.get("challenge") or {}).get("steps", {}).values())
        for st in steps:
            if not st:
                continue
            if st.get("call"):
                c = st["call"]
                calls.append({"input_chars": c["input"]["chars"], "output_chars": c["output"]["chars"],
                              "usage": c.get("usage"), "latency_ms": c.get("latency_ms"), "is_fake": c["is_fake"]})
            for d in st.get("decisions", []):
                decisions[d["decision"]] += 1
            o = st.get("outcome") or {}
            if o.get("change_id"):
                ops_committed += len([x for x in o["accepted"] if x not in o.get("no_change_records", [])])
    if trace.get("call"):
        c = trace["call"]
        calls.append({"input_chars": c["input"]["chars"], "output_chars": c["output"]["chars"],
                      "usage": c.get("usage"), "latency_ms": c.get("latency_ms"), "is_fake": c["is_fake"]})
    tok_in = [c["usage"].get("input_tokens") for c in calls if c.get("usage")]
    tok_out = [c["usage"].get("output_tokens") for c in calls if c.get("usage")]
    return {"model_calls": len(calls), "input_chars": sum(c["input_chars"] for c in calls),
            "output_chars": sum(c["output_chars"] for c in calls),
            "input_tokens": sum(tok_in) if tok_in else None, "output_tokens": sum(tok_out) if tok_out else None,
            "latency_ms": (sum(c["latency_ms"] for c in calls if c["latency_ms"]) or None) if calls else None,
            "fake_calls": sum(c["is_fake"] for c in calls),
            "review_decisions": decisions, "graph_ops_committed": ops_committed}


if __name__ == "__main__":
    raise SystemExit(main())
