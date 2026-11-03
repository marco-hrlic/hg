import json
import re

from graph_adlc import viz

SNAP = {"graph_revision": 3,
        "nodes": [{"id": "REQ-A", "kind": "requirement", "title": "A </script><b>x</b>", "rev": 1},
                  {"id": "CMP-A", "kind": "component", "title": "Comp", "rev": 1},
                  {"id": "TEST-A", "kind": "test", "title": "Test", "rev": 1},
                  {"id": "CMP-OLD", "kind": "component", "title": "Gone", "rev": 2, "deleted": True}],
        "edges": [{"id": "E1", "src": "CMP-A", "dst": "REQ-A", "relation": "implements", "status": "confirmed", "rev": 1},
                  {"id": "E2", "src": "TEST-A", "dst": "REQ-A", "relation": "verifies", "status": "hypothesized",
                   "rev": 1}]}
PROBE = {"event": "E5", "node_id": "REQ-A", "aspect": "semantic", "description": "probe"}


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _run_dir(tmp_path):
    run = tmp_path / "evaluation" / "out" / "live"
    _write(run / "p" / "G1" / "trace.json", {
        "project": "p", "arm": "G1", "probe": PROBE, "final_graph": SNAP,
        "prediction": {"E5": {"targets": ["CMP-A", "REQ-A"], "pairs": [], "pruned": [], "provisional": []}},
        "client": {"provider": "openai", "model": "m", "is_fake": False, "api_key_env": "X"}})
    _write(run / "p" / "D" / "trace.json", {"project": "p", "arm": "D", "prediction": {}})
    _write(run / "p" / "H" / "trace.json", {"project": "p", "arm": "H", "probe": PROBE, "prediction": {}})
    _write(tmp_path / "evaluation" / "reference" / "p.json", {**SNAP, "nodes": SNAP["nodes"][:3]})
    return run


def test_run_dir_loads_graph_arms_skips_d_and_finds_h_reference(tmp_path):
    entries, notes = viz.load_inputs([str(_run_dir(tmp_path))])
    assert [e["arm"] for e in entries] == ["G1", "H"]  # arm order, D skipped
    assert any("p/D" in n for n in notes)
    g1, h = entries
    assert g1["probe"]["origin"] == "REQ-A" and g1["probe"]["targets"] == ["CMP-A", "REQ-A"]
    assert g1["client"] == {"provider": "openai", "model": "m", "is_fake": False}  # nothing else copied
    assert len(h["nodes"]) == 3


def test_missing_reference_is_a_note_not_a_crash(tmp_path):
    run = _run_dir(tmp_path)
    (tmp_path / "evaluation" / "reference" / "p.json").unlink()
    entries, notes = viz.load_inputs([str(run)])
    assert [e["arm"] for e in entries] == ["G1"] and any("p/H" in n for n in notes)


def test_layout_is_deterministic_rows_follow_kind_order_and_union_of_arms(tmp_path):
    entries, _ = viz.load_inputs([str(_run_dir(tmp_path))])
    lay = viz.layout(entries)["p"]
    assert [k for k, _ in lay["rows"]] == ["requirement", "component", "test"]
    assert set(lay["coords"]) == {"REQ-A", "CMP-A", "TEST-A", "CMP-OLD"}  # deleted nodes get a slot too
    ys = {k: lay["coords"][n][1] for n, k in [("REQ-A", "requirement"), ("CMP-A", "component"), ("TEST-A", "test")]}
    assert ys["requirement"] < ys["component"] < ys["test"]
    assert viz.layout(entries)["p"] == lay


def test_render_is_self_contained_and_escapes_script_breakout(tmp_path):
    f = tmp_path / "snap.json"
    _write(f, SNAP)
    entries, _ = viz.load_inputs([str(f)])
    page = viz.render(entries, {})
    assert page.count("</script>") == 2  # only the page's own two script blocks close
    assert not re.search(r"<(script|link)[^>]+(src|href)=", page)  # no external resources
    blob = page.split('id="data">', 1)[1].split("</script>", 1)[0]
    assert json.loads(blob)["graphs"][0]["nodes"][0]["title"] == "A </script><b>x</b>"


def test_oracle_overlay_and_cli(tmp_path, capsys):
    run = _run_dir(tmp_path)
    oracle = tmp_path / "oracle"
    _write(oracle / "p.json", {"project": "p", "oracle_version": "1.0", "labeler": "x",
                               "impact": {"E5": {"origin": "REQ-A", "necessary": [{"target": "TEST-A", "critical": True}]}}})
    out = tmp_path / "g.html"
    assert viz.main([str(run), "--oracle", str(oracle), "-o", str(out)]) == 0
    data = json.loads(out.read_text().split('id="data">', 1)[1].split("</script>", 1)[0])
    assert data["oracles"]["p"]["impact"]["E5"]["necessary"][0]["target"] == "TEST-A"
    assert "wrote" in capsys.readouterr().out
    empty = tmp_path / "empty" / "x" / "D" / "trace.json"
    _write(empty, {"project": "x", "arm": "D"})
    assert viz.main([str(empty), "-o", str(tmp_path / "n.html")]) == 1


def test_wide_rows_wrap_instead_of_widening_the_page():
    nodes = [{"id": f"Q-{i:02d}", "kind": "question", "title": "q"} for i in range(14)]
    nodes.append({"id": "REQ-A", "kind": "requirement", "title": "r"})
    lay = viz.layout([{"project": "p", "nodes": nodes, "edges": []}])["p"]
    xs = {lay["coords"][n["id"]][0] for n in nodes if n["kind"] == "question"}
    ys = {lay["coords"][n["id"]][1] for n in nodes if n["kind"] == "question"}
    assert len(xs) == viz.PER_LINE and len(ys) == 3  # 14 nodes -> 6 + 6 + 2
    assert lay["width"] <= 2 * viz.PAD + viz.LABEL_W + (viz.PER_LINE - 1) * viz.COL_W + viz.NODE_W
