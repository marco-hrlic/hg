"""HG-v2 grounding: the index theorem, joint sufficiency, and the artifacts.

Plan section 24. The index problem is the one place in this repository where
the correct answer is known exactly rather than approximately, so wherever a
quantity is deterministic these tests assert the exact value -- a tolerance
would let a real accounting bug through.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hg.config import ConfigError
from hg.grounding.channels import (
    BitVectorSource,
    QuerySpec,
    Route,
    generic_bit_query,
    log2_alphabet,
    targeted_bit_query,
)
from hg.grounding.config import load_grounding_config
from hg.grounding.costs import (
    OVERHEAD_DEFINED,
    OVERHEAD_INSUFFICIENT,
    OVERHEAD_NO_COMMUNICATION,
    aggregate_all,
    aggregate_arm,
    coordination_overhead,
    empirical_response_entropy,
    information_efficiency,
)
from hg.grounding.episodes import (
    GroundingEpisode,
    GroundingRunDir,
    read_episodes,
    write_episodes,
)
from hg.grounding.phase0 import (
    RESIDUAL_DEFICIT_BITS,
    build_analysis,
    compare_analyses,
    run_phase0,
    run_sweep,
    theory_expectation,
)
from hg.grounding.synthetic import (
    POLICIES,
    IndexTaskSpec,
    generate_index_items,
    run_arm,
    run_policy,
)
from hg.schema import SchemaError

SMALL_N = (2, 3, 4, 8)


def arms_by_policy(episodes, **kwargs):
    return {a.policy: a for a in aggregate_all(episodes, **kwargs)}


def episodes_for(n_bits: int, n_items: int = 400, seed: int = 0):
    items = generate_index_items(
        IndexTaskSpec(n_bits=n_bits, n_items=n_items, seed=seed)
    )
    out = []
    for policy in POLICIES:
        out.extend(run_arm(policy, items, seed=seed))
    return items, out


# ---------------------------------------------------- the index theorem ------
@pytest.mark.parametrize("n", SMALL_N)
def test_generic_response_costs_n_bits_and_targeted_costs_one(n):
    """B_generic = n and B_targeted = 1, exactly, at every n."""
    _, episodes = episodes_for(n)
    arms = arms_by_policy(episodes)

    assert arms["ask_generic"].mean_external_bits == float(n)
    assert arms["ask_targeted"].mean_external_bits == 1.0
    assert arms["act"].mean_external_bits == 0.0


@pytest.mark.parametrize("n", SMALL_N)
def test_both_ask_policies_decide_perfectly(n):
    """Decision quality is identical, so only the cost separates them."""
    _, episodes = episodes_for(n)
    arms = arms_by_policy(episodes)

    assert arms["ask_generic"].accuracy == 1.0
    assert arms["ask_targeted"].accuracy == 1.0
    assert arms["ask_generic"].mean_loss_after == arms["ask_targeted"].mean_loss_after == 0.0


@pytest.mark.parametrize("n", SMALL_N)
def test_coordination_overhead_matches_theory(n):
    """Omega = n - 1 for generic escalation and exactly 0 for a targeted query."""
    _, episodes = episodes_for(n)
    arms = arms_by_policy(episodes)

    assert arms["ask_generic"].coordination_overhead == n - 1.0
    assert arms["ask_targeted"].coordination_overhead == 0.0
    assert arms["ask_targeted"].overhead_status == OVERHEAD_DEFINED


def test_act_accuracy_approaches_chance():
    """H(J | M_A) = 1 bit means acting on M_A alone is a coin flip.

    A statistical claim, so this is the one place a tolerance is right: at 4000
    items the standard error is ~0.008 and 4 SE is ~0.03.
    """
    _, episodes = episodes_for(8, n_items=4000)
    arm = arms_by_policy(episodes)["act"]
    assert abs(arm.accuracy - 0.5) < 0.03
    assert arm.delta_distortion == 0.0  # nothing was acquired, nothing improved


def test_targeted_cost_is_flat_while_generic_grows():
    """The scaling claim, checked across the sweep rather than per n."""
    generic, targeted = [], []
    for n in (2, 4, 8, 16, 32):
        _, episodes = episodes_for(n, n_items=200)
        arms = arms_by_policy(episodes)
        generic.append(arms["ask_generic"].mean_external_bits)
        targeted.append(arms["ask_targeted"].mean_external_bits)

    assert generic == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert targeted == [1.0] * 5


def test_residual_deficit_is_one_bit_by_construction():
    _, episodes = episodes_for(4)
    assert all(
        e.decision_state["residual_deficit_bits"] == RESIDUAL_DEFICIT_BITS
        for e in episodes
    )
    assert all(e.decision_state["exact"] is True for e in episodes)
    # No amount of internal computation resolves X_K: it is not in M_A at all.
    assert all(e.decision_state["internal_reducibility"] == 0.0 for e in episodes)


# ------------------------------------------------------- joint sufficiency ---
@pytest.mark.parametrize("n", SMALL_N)
def test_removing_the_queried_variable_breaks_perfect_recovery(n):
    """Plan section 24: without the relevant source information, recovery fails.

    The ablated arm pays the same 1 bit, so this separates "paid for
    information" from "acquired information" -- the distinction the whole
    architecture rests on.
    """
    _, episodes = episodes_for(n, n_items=1000)
    arms = arms_by_policy(episodes)
    ablated = arms["ask_targeted_ablated"]

    assert ablated.accuracy < 0.6
    assert not ablated.sufficient
    assert ablated.mean_external_bits == 1.0  # cost paid regardless
    assert abs(ablated.delta_distortion) < 0.06  # no real improvement


def test_ablated_source_reports_itself_uninformative():
    """The response is well formed but flagged, not silently wrong."""
    _, episodes = episodes_for(4, n_items=100)
    ablated = [e for e in episodes if e.policy == "ask_targeted_ablated"]
    assert all(e.external_response["informative"] is False for e in ablated)
    # ...and the honest arms are flagged informative.
    honest = [e for e in episodes if e.policy in ("ask_generic", "ask_targeted")]
    assert all(e.external_response["informative"] is True for e in honest)


def test_source_withholding_an_unasked_bit_does_not_hurt():
    """Only the decision-relevant variable matters: sufficiency is about J."""
    source = BitVectorSource([1, 0, 1, 1], known=[True, True, False, True])
    response = source.respond(targeted_bit_query(0, 4))
    assert response.content == "1"
    assert response.informative is True


# ------------------------------------------------------------ Omega semantics ---
def test_omega_is_undefined_without_communication():
    """ACT must not appear to beat the source-coding bound.

    Omega = B - H(J | M_A) evaluates to -1 for a policy that never
    communicates. That is not compression; it is an ungrounded decision, and
    reporting it as a number is the most misleading thing this analysis could do.
    """
    _, episodes = episodes_for(4, n_items=100)
    act = arms_by_policy(episodes)["act"]
    assert act.coordination_overhead is None
    assert act.overhead_status == OVERHEAD_NO_COMMUNICATION
    # The raw definition still returns the negative value; the guard is at the
    # arm level, deliberately.
    assert coordination_overhead(0.0, 1.0) == -1.0


def test_omega_is_undefined_for_insufficient_communication():
    """The ablated arm spends 1 bit and resolves nothing: Omega = 0 would read
    as optimal."""
    _, episodes = episodes_for(4, n_items=100)
    ablated = arms_by_policy(episodes)["ask_targeted_ablated"]
    assert ablated.coordination_overhead is None
    assert ablated.overhead_status == OVERHEAD_INSUFFICIENT


def test_information_efficiency_is_none_at_zero_bits():
    """Not inf, not nan: acting for free is the absence of external
    information use, and a sentinel would corrupt any mean it entered."""
    assert information_efficiency(0.5, 0.0) is None
    assert information_efficiency(0.5, 1.0) == 0.5


@pytest.mark.parametrize("n", SMALL_N)
def test_targeted_query_is_n_times_more_efficient_than_generic(n):
    """Equal distortion reduction over n times fewer bits."""
    _, episodes = episodes_for(n, n_items=500)
    arms = arms_by_policy(episodes)
    generic, targeted = arms["ask_generic"], arms["ask_targeted"]

    assert generic.delta_distortion == targeted.delta_distortion
    assert targeted.information_efficiency == pytest.approx(
        generic.information_efficiency * n
    )


# ------------------------------------------------------- channels and costs ---
def test_log2_alphabet():
    assert log2_alphabet(2) == 1.0
    assert log2_alphabet(256) == 8.0
    assert log2_alphabet(1) == 0.0
    with pytest.raises(ValueError):
        log2_alphabet(0)


def test_generic_query_alphabet_is_exact_but_not_enumerated():
    q = generic_bit_query(64)
    assert q.response_bits == 64.0
    assert q.response_alphabet_size == 2**64
    # 2**64 strings cannot be listed; the size above is the exact statement.
    assert q.candidate_values is None


def test_targeted_query_rejects_an_out_of_range_index():
    with pytest.raises(ValueError):
        targeted_bit_query(4, 4)


def test_query_spec_rejects_inconsistent_alphabet():
    with pytest.raises(ValueError):
        QuerySpec(
            variable_names=("X0",),
            response_bits=1.0,
            natural_language_query="?",
            response_alphabet_size=4,
            candidate_values=("0", "1"),
        )


def test_empirical_entropy_is_withheld_when_undersampled():
    """A plug-in entropy over 2**n symbols from a few hundred samples is bias,
    not a measurement."""
    _, episodes = episodes_for(16, n_items=300)
    by_policy: dict[str, list] = {}
    for e in episodes:
        by_policy.setdefault(e.policy, []).append(e)

    assert empirical_response_entropy(by_policy["ask_generic"]) is None
    # A binary response alphabet is estimable, and it is a fair coin.
    h = empirical_response_entropy(by_policy["ask_targeted"])
    assert h is not None and abs(h - 1.0) < 0.05


def test_aggregate_arm_refuses_mixed_arms():
    _, episodes = episodes_for(4, n_items=10)
    with pytest.raises(ValueError, match="mixes policies"):
        aggregate_arm(episodes)


def test_aggregate_arm_refuses_mixed_task_sizes():
    _, a = episodes_for(4, n_items=10)
    _, b = episodes_for(8, n_items=10)
    act = [e for e in a + b if e.policy == "act"]
    with pytest.raises(ValueError, match="mixes task sizes"):
        aggregate_arm(act)


# ---------------------------------------------------------- serialization ----
def test_episode_round_trips_losslessly(tmp_path):
    """Plan section 24: a saved episode must round-trip unchanged."""
    _, episodes = episodes_for(8, n_items=50)
    path = tmp_path / "episodes.jsonl"
    assert write_episodes(path, episodes) == len(episodes)

    reloaded = read_episodes(path, strict=True)
    assert len(reloaded) == len(episodes)
    for original, back in zip(episodes, reloaded):
        assert original.to_dict() == back.to_dict()
        assert original == back


def test_metrics_survive_the_disk_round_trip(tmp_path):
    _, episodes = episodes_for(8, n_items=200)
    path = tmp_path / "episodes.jsonl"
    write_episodes(path, episodes)
    assert [a.to_dict() for a in aggregate_all(episodes)] == [
        a.to_dict() for a in aggregate_all(read_episodes(path))
    ]


def test_episode_validation_rejects_a_paying_act_route():
    """Acting now is by definition free of external communication."""
    e = GroundingEpisode(
        item_id="x", task="index", policy="act", route=Route.ACT.value,
        initial_action="0", initial_loss=1.0, final_action="0", final_loss=1.0,
        external_raw_bits=3.0,
    )
    with pytest.raises(SchemaError, match="external bits"):
        e.validate()


def test_episode_validation_rejects_a_cost_mismatch():
    """The bit accounting *is* the result, so a mismatch is a hard error."""
    item = generate_index_items(IndexTaskSpec(n_bits=4, n_items=1, seed=0))[0]
    e = run_policy("ask_targeted", item, seed=0)
    e.external_raw_bits = 7.0
    with pytest.raises(SchemaError, match="cost accounting"):
        e.validate()


def test_episode_validation_rejects_an_unknown_route():
    e = GroundingEpisode(
        item_id="x", task="index", policy="p", route="deliberate",
        initial_action="0", initial_loss=1.0, final_action="0", final_loss=1.0,
    )
    with pytest.raises(SchemaError, match="unknown route"):
        e.validate()


def test_reading_an_unknown_field_fails_loudly(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"item_id":"x","task":"index","route":"act","policy":"p",'
                    '"initial_action":"0","initial_loss":1.0,"final_action":"0",'
                    '"final_loss":1.0,"external_raw_bits":0.0,"external_cost":0.0,'
                    '"surprise":1}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="unexpected fields"):
        read_episodes(path)


def test_missing_required_column_fails_loudly(tmp_path):
    path = tmp_path / "short.jsonl"
    path.write_text('{"item_id":"x","task":"index","route":"act"}\n', encoding="utf-8")
    with pytest.raises(SchemaError, match="missing required columns"):
        read_episodes(path)


# ------------------------------------------------------ determinism / seeds ---
def test_the_task_is_deterministic_given_the_seed():
    a = generate_index_items(IndexTaskSpec(n_bits=8, n_items=100, seed=3))
    b = generate_index_items(IndexTaskSpec(n_bits=8, n_items=100, seed=3))
    assert a == b
    c = generate_index_items(IndexTaskSpec(n_bits=8, n_items=100, seed=4))
    assert a != c


def test_every_policy_sees_the_same_act_now_guess():
    """What makes the arms paired, and delta-distortion comparable between them."""
    items = generate_index_items(IndexTaskSpec(n_bits=8, n_items=100, seed=0))
    guesses = {
        policy: [e.initial_action for e in run_arm(policy, items, seed=0)]
        for policy in POLICIES
    }
    reference = guesses["act"]
    assert all(g == reference for g in guesses.values())


def test_one_bit_tasks_are_rejected():
    """With n = 1 the index carries no information and a targeted query *is* a
    generic one, so the experiment would be vacuous."""
    with pytest.raises(ValueError, match="n_bits must be >= 2"):
        IndexTaskSpec(n_bits=1, n_items=10)


# ----------------------------------------------------------------- config ----
def test_the_shipped_phase0_config_loads(repo_root):
    cfg = load_grounding_config(
        repo_root / "configs" / "experiments" / "hg_v2" / "phase0_index.yaml"
    )
    assert cfg.sweep.n_bits == [2, 4, 8, 16, 32, 64]
    assert set(cfg.policies) == set(POLICIES)
    assert cfg.resolved_dict()["task"]["n_bits"] == cfg.sweep.n_bits


def test_config_rejects_an_unknown_policy(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("task:\n  kind: index\n  n_bits: [4]\npolicies: [act, telepathy]\n")
    with pytest.raises(ConfigError, match="unknown policies"):
        load_grounding_config(p)


def test_config_rejects_an_unknown_gate_key(tmp_path):
    """A misspelled threshold must not silently leave the gate at its default."""
    p = tmp_path / "bad.yaml"
    p.write_text("task:\n  kind: index\n  n_bits: [4]\ngate:\n  act_accuracy_tolerance: 0.1\n")
    with pytest.raises(ConfigError, match="unknown gate keys"):
        load_grounding_config(p)


def test_config_rejects_a_degenerate_sweep(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("task:\n  kind: index\n  n_bits: [1, 4]\n")
    with pytest.raises(ConfigError, match=">= 2"):
        load_grounding_config(p)


# ------------------------------------------------------- the phase 0 gate ----
def phase0_config(tmp_path, **over):
    import yaml

    body = {
        "name": "test_phase0",
        "task": {"kind": "index", "n_bits": [2, 4], "n_items": 400},
        "policies": list(POLICIES),
        "seed": 0,
        "n_bootstrap": 0,
        "output_root": str(tmp_path / "results"),
    }
    body.update(over)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(body), encoding="utf-8")
    return load_grounding_config(p)


def test_gate_passes_on_the_real_sweep(tmp_path):
    cfg = phase0_config(tmp_path)
    analysis = build_analysis(cfg, run_sweep(cfg))
    assert analysis["gate"]["passed"], analysis["gate"]["failures"]
    assert analysis["gate"]["n_checks"] > 0


def test_gate_catches_a_corrupted_bit_count(tmp_path):
    """The gate has to be able to fail, or it is decoration."""
    cfg = phase0_config(tmp_path)
    episodes = run_sweep(cfg)
    for e in episodes:
        if e.policy == "ask_targeted":
            e.external_raw_bits = 2.0
            if e.external_response:
                e.external_response = {**e.external_response, "raw_bits": 2.0}
    gate = build_analysis(cfg, episodes)["gate"]
    assert not gate["passed"]
    assert any("targeted_bits_constant" in f for f in gate["failures"])


def test_gate_catches_a_broken_targeted_decode(tmp_path):
    cfg = phase0_config(tmp_path)
    episodes = run_sweep(cfg)
    for e in episodes:
        if e.policy == "ask_targeted":
            e.final_action = "0"
            e.final_loss = 0.0 if e.hidden_reference_state["decision_variable"] == "0" else 1.0
    gate = build_analysis(cfg, episodes)["gate"]
    assert not gate["passed"]
    assert any("ask_accuracy" in f for f in gate["failures"])


def test_theory_expectation_covers_every_policy():
    for policy in POLICIES:
        assert set(theory_expectation(policy, 8)) == {"accuracy", "external_bits", "omega"}
    with pytest.raises(ValueError):
        theory_expectation("no_such_policy", 8)


def test_compression_summary_reports_the_headline(tmp_path):
    cfg = phase0_config(tmp_path)
    analysis = build_analysis(cfg, run_sweep(cfg))
    rows = {c["n_bits"]: c for c in analysis["compression"]}
    assert rows[4]["equal_decision_quality"] is True
    assert rows[4]["bits_saved"] == 3.0
    assert rows[4]["compression_ratio"] == 4.0


# ---------------------------------------------- analysis reproducibility -----
def test_analysis_from_saved_artifacts_matches_the_live_run(tmp_path):
    """Plan section 24: metrics computed from saved artifacts must match the
    live run's, and section 8's criterion 6 -- regenerate without rerunning."""
    cfg = phase0_config(tmp_path, n_bootstrap=50)
    run, analysis, problems = run_phase0(cfg)

    assert problems == []
    assert analysis["gate"]["passed"], analysis["gate"]["failures"]

    # And again, entirely from disk, through the public entry point.
    from hg.grounding.phase0 import analyze_run_dir

    again = analyze_run_dir(run.root)
    assert compare_analyses(analysis, again) == []
    assert run.analysis_path.exists()
    assert run.report_path.exists()
    assert (run.root / "manifest.json").exists()
    assert run.config_path.exists()


def test_a_run_directory_is_never_silently_overwritten(tmp_path, monkeypatch):
    """Plan section 25.

    Run ids embed a second-resolution timestamp, so in practice two runs
    collide only when they start within the same second. The id generator is
    pinned here so the guard itself is under test rather than the clock -- the
    earlier version of this test passed or failed depending on whether the two
    writes straddled a second boundary.
    """
    from hg.grounding import phase0 as phase0_module

    monkeypatch.setattr(phase0_module, "run_id", lambda *a, **k: "pinned_run_id")

    cfg = phase0_config(tmp_path)
    episodes = run_sweep(cfg)
    run = phase0_module.write_run(cfg, episodes)
    assert run.root.name == "pinned_run_id"
    assert GroundingRunDir(run.root).available_policies()

    with pytest.raises(FileExistsError, match="already exists"):
        phase0_module.write_run(cfg, episodes)


def test_manifest_records_provenance(tmp_path):
    from hg.io import read_json
    from hg.grounding.phase0 import write_run

    cfg = phase0_config(tmp_path)
    run = write_run(cfg, run_sweep(cfg))
    manifest = read_json(run.manifest_path)
    for key in (
        "run_id", "task_version", "seed", "n_bits_sweep", "policies",
        "created_at", "environment", "residual_deficit_bits",
    ):
        assert key in manifest, key
    # Phase 0 must never look like a model result.
    assert manifest["model_checkpoint"] is None


def test_artifacts_are_strict_json(tmp_path):
    """No NaN or Infinity tokens anywhere in the saved artifacts.

    `json.dump` emits bare `NaN`/`Infinity` by default, which is not valid JSON
    and which a conforming reader in another language will reject. An artifact
    that only Python can read is not the durable artifact this pipeline
    promises, so the check is on the bytes rather than on the objects.
    """
    import json

    cfg = phase0_config(tmp_path, n_bootstrap=20)
    run, _, _ = run_phase0(cfg)

    for path in [run.analysis_path, run.manifest_path, *sorted((run.root / "episodes").glob("*.jsonl"))]:
        text = path.read_text(encoding="utf-8")
        for token in ("NaN", "Infinity", "-Infinity"):
            assert token not in text, f"{path.name} contains a bare {token} token"
        if path.suffix == ".jsonl":
            for line in text.splitlines():
                json.loads(line, parse_constant=_reject_constant)
        else:
            json.loads(text, parse_constant=_reject_constant)


def _reject_constant(name: str):
    raise AssertionError(f"non-standard JSON constant in artifact: {name}")


def test_a_resolved_config_remembers_its_original_path(tmp_path):
    """Reproduce instructions must point at a file a person edits, not at the
    copy inside a run directory."""
    from hg.io import write_yaml

    cfg = phase0_config(tmp_path)
    original = cfg.source_path
    resolved = tmp_path / "run" / "config.resolved.yaml"
    write_yaml(resolved, cfg.resolved_dict())

    reloaded = load_grounding_config(resolved)
    assert reloaded.source_path == original
    # ...and everything else survives the round trip.
    assert reloaded.resolved_dict() == cfg.resolved_dict()
