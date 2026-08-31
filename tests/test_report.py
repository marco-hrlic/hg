"""Decision gates and prediction verdicts."""

from __future__ import annotations

import math

from hg.metrics.bootstrap import BootstrapResult
from hg.report.decision import decision_gate, evaluate_predictions, render_report


def boot(name, point, lo, hi):
    return BootstrapResult(name, point, lo, hi, 0.01, 100, 100)


def test_zero_residual_is_gate_a():
    v = decision_gate(clean_residual=0.0, n_clean=1000)
    assert v.gate == "A"
    assert "adjudication" in v.next_action


def test_residual_whose_interval_includes_zero_is_gate_a_with_a_caveat():
    v = decision_gate(clean_residual=0.01, residual_ci=(-0.002, 0.02), n_clean=1000)
    assert v.gate == "A"
    assert any("interval including zero" in c for c in v.caveats)


def test_falling_curve_with_nonzero_residual_is_gate_b():
    v = decision_gate(
        clean_residual=0.05,
        residual_ci=(0.03, 0.07),
        survival_shape={"shape": "still_falling", "caveat": "not flattened"},
        n_clean=1000,
    )
    assert v.gate == "B"


def test_plateau_is_gate_c():
    v = decision_gate(
        clean_residual=0.05,
        residual_ci=(0.03, 0.07),
        survival_shape={"shape": "apparent_plateau", "caveat": "upper bound only"},
        n_clean=1000,
    )
    assert v.gate == "C"
    assert any("upper bound" in c for c in v.caveats)


def test_no_survival_curve_defaults_to_gate_b_not_c():
    """Gate C licenses building the grounding architecture. It must not be
    reachable from a nonzero endpoint with no evidence the curve flattened."""
    v = decision_gate(clean_residual=0.05, residual_ci=(0.03, 0.07), n_clean=1000)
    assert v.gate == "B"
    assert any("no survival curve" in c for c in v.caveats)


def test_gate_d_requires_a_capability_trend():
    shape = {"shape": "apparent_plateau"}
    assert decision_gate(
        clean_residual=0.05, survival_shape=shape, n_clean=1000
    ).gate == "C"
    assert decision_gate(
        clean_residual=0.05, survival_shape=shape, n_clean=1000, capability_trend=0.02
    ).gate == "D"


def test_small_clean_subset_is_flagged_on_every_gate():
    v = decision_gate(clean_residual=0.0, n_clean=20)
    assert any("provisional" in c for c in v.caveats)


def test_undefined_residual_is_undetermined():
    v = decision_gate(clean_residual=float("nan"), n_clean=0)
    assert v.gate == "undetermined"


def test_gate_b_notes_when_the_curve_tracks_shared_difficulty():
    v = decision_gate(
        clean_residual=0.02,
        survival_shape={
            "shape": "still_falling",
            "caveat": "not flattened",
            "excess_verdict": "consistent_with_shared_difficulty",
        },
        n_clean=1000,
    )
    assert v.gate == "B"
    assert any("expect closure" in c for c in v.caveats)


def test_p1_supported_when_the_gap_excludes_zero():
    verdicts = {
        p.name: p
        for p in evaluate_predictions(diff_hs_dd=boot("C_HS^clean - C_DD^clean", 0.06, 0.03, 0.09))
    }
    assert verdicts["P1"].status == "supported"


def test_p1_undetermined_when_the_gap_spans_zero():
    verdicts = {
        p.name: p
        for p in evaluate_predictions(diff_hs_dd=boot("C_HS^clean - C_DD^clean", 0.01, -0.02, 0.04))
    }
    assert verdicts["P1"].status == "undetermined"


def test_p2_judges_measurability_not_magnitude():
    """Protocol section 11 deliberately does not preregister P2's magnitude."""
    small = {p.name: p for p in evaluate_predictions(clean_dd=boot("C_DD^clean", 0.001, 0.0005, 0.002))}
    assert small["P2"].status == "supported"
    spans = {p.name: p for p in evaluate_predictions(clean_dd=boot("C_DD^clean", 0.01, -0.001, 0.02))}
    assert spans["P2"].status == "not_supported"


def test_p3_tracks_the_direction_of_q_contested():
    up = {p.name: p for p in evaluate_predictions(q_contested_hs=0.2, q_contested_dd=0.5)}
    assert up["P3"].status == "supported"
    down = {p.name: p for p in evaluate_predictions(q_contested_hs=0.5, q_contested_dd=0.2)}
    assert down["P3"].status == "not_supported"


def test_p4_requires_conversion_not_mere_removal():
    conv = {p.name: p for p in evaluate_predictions(conversion={"conversion_ratio": 1.4})}
    assert conv["P4"].status == "supported"
    removal = {p.name: p for p in evaluate_predictions(conversion={"conversion_ratio": 0.3})}
    assert removal["P4"].status == "not_supported"


def test_p6_is_undetermined_below_gate_c_not_refuted():
    gate = decision_gate(clean_residual=0.0, n_clean=1000)
    verdicts = {
        p.name: p
        for p in evaluate_predictions(
            race={"verdict": "disagreement_wins", "note": "x"}, gate=gate
        )
    }
    assert verdicts["P6"].status == "undetermined"
    assert "untested rather than refuted" in verdicts["P6"].detail
    assert verdicts["P5"].status == "supported"


def test_p6_supported_at_gate_c_when_targeting_wins():
    gate = decision_gate(
        clean_residual=0.05, survival_shape={"shape": "apparent_plateau"}, n_clean=1000
    )
    verdicts = {
        p.name: p
        for p in evaluate_predictions(
            race={"verdict": "targeting_wins", "note": "x", "localization_surplus": 0.2},
            gate=gate,
        )
    }
    assert verdicts["P6"].status == "supported"


def test_report_puts_warnings_before_the_headline():
    text = render_report(
        experiment="e",
        conditions={"DD": {"n_items": 10, "n_members": 5, "C_E": 0.1, "C_E_clean": 0.1}},
        gate=decision_gate(clean_residual=0.0, n_clean=1000),
        warnings=["something is off"],
    )
    assert text.index("something is off") < text.index("Decision gate")
