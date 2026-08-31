"""Decision gates (protocol section 16) and preregistered predictions (11).

This module deliberately does two things that are easy to skip:

* It reports a gate verdict *and* the evidence that would have changed it, so a
  gate reached on 40 clean items is not written up the same way as one reached
  on 4,000.
* It refuses to promote gate C -- "a stable clean shared residual survives",
  the one that licenses building the grounding architecture -- on the strength
  of a nonzero endpoint alone. Section 18's "insufficient diversity range"
  confound says a nonzero residual at E=5 is an upper bound on an
  information-limited floor unless the survival curve has flattened, and gate B
  exists precisely for that case.

The preregistered predictions are evaluated mechanically against thresholds
written down before the run (`prereg/m2_llm_v1.md`), with one exception: P2's
magnitude is deliberately not preregistered, per section 11, so P2 is judged
only on whether a clean residual is *measurable* -- a CI excluding zero -- not
on how big it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..metrics.bootstrap import BootstrapResult


@dataclass(slots=True)
class GateVerdict:
    gate: str  # "A" | "B" | "C" | "D" | "undetermined"
    title: str
    next_action: str
    evidence: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "title": self.title,
            "next_action": self.next_action,
            "evidence": self.evidence,
            "caveats": self.caveats,
        }


GATES = {
    "A": (
        "Clean DD residual is approximately zero",
        "Prioritize heterogeneous internal ensembles and adjudication for "
        "contested items. Epistemic external-grounding pressure appears weak "
        "on this task.",
    ),
    "B": (
        "Clean DD residual is above zero but still falling with added diversity",
        "Extend the diversity axis. Do not yet claim an information-limited floor.",
    ),
    "C": (
        "A stable clean shared residual survives DD",
        "Test localization and targeted external correction. This is the key "
        "condition for failure-memory value.",
    ),
    "D": (
        "Clean residual survives and grows relatively with capability",
        "Highest theory relevance. Begin explicit scaling analysis of the "
        "information race and harm race.",
    ),
}


def decision_gate(
    *,
    clean_residual: float,
    survival_shape: Mapping[str, Any] | None = None,
    residual_ci: tuple[float, float] | None = None,
    n_clean: int = 0,
    contested_share: float = float("nan"),
    capability_trend: float | None = None,
    zero_tol: float = 0.002,
    min_clean: int = 200,
) -> GateVerdict:
    """Map measured quantities onto one of the four gates.

    `capability_trend` is the slope of the clean residual against model
    capability across scales. It is the only route to gate D and is `None`
    until a scaling sweep exists, so gate D is never reached by accident from a
    single-scale run.
    """
    caveats: list[str] = []
    evidence: dict[str, Any] = {
        "clean_residual": clean_residual,
        "residual_ci": list(residual_ci) if residual_ci else None,
        "n_clean_items": n_clean,
        "contested_share_of_residual": contested_share,
        "survival_shape": (survival_shape or {}).get("shape"),
        "trailing_relative_slope": (survival_shape or {}).get("trailing_relative_slope"),
        # Whether the observed survival curve is pulling away from the
        # shared-difficulty model's curve. Does not move the gate on its own --
        # gate C still requires the curve to flatten -- but it is what says
        # whether adding families is worth the GPU time.
        "excess_verdict": (survival_shape or {}).get("excess_verdict"),
        "excess_ratio_growth": (
            (survival_shape or {}).get("excess_survival") or {}
        ).get("ratio_growth"),
    }

    if n_clean < min_clean:
        caveats.append(
            f"only {n_clean} unanimous items; below the {min_clean}-item target "
            "for a usable clean subset, so every gate below is provisional"
        )

    if math.isnan(clean_residual):
        return GateVerdict(
            "undetermined",
            "Clean residual is undefined",
            "No unanimous-anchor items survived the filter. Fix the anchor "
            "pipeline before interpreting anything else.",
            evidence,
            caveats,
        )

    ci_excludes_zero = bool(residual_ci and residual_ci[0] > 0)

    # Gate A: the residual is at or below tolerance, or its CI includes zero.
    if clean_residual <= zero_tol or (residual_ci and residual_ci[0] <= 0 <= residual_ci[1]):
        if not math.isnan(contested_share) and contested_share > 0.5:
            caveats.append(
                f"{contested_share:.0%} of what residual remains sits on contested "
                "anchors: the live problem here is anchor formation and "
                "adjudication, not epistemic grounding (protocol section 10)"
            )
        if clean_residual > zero_tol:
            caveats.append(
                f"point estimate {clean_residual:.4f} is above the {zero_tol} "
                "tolerance; gate A rests on the interval including zero, so a "
                "larger clean subset could move this to gate B"
            )
        title, action = GATES["A"]
        return GateVerdict("A", title, action, evidence, caveats)

    shape = (survival_shape or {}).get("shape")

    # Gate D requires a capability trend; it is never reached from one scale.
    if capability_trend is not None and capability_trend > 0 and shape == "apparent_plateau":
        title, action = GATES["D"]
        caveats.append(
            "Gate D is a claim about scaling. It needs at least three capability "
            "levels with overlapping evaluation items before the trend is more "
            "than two points and a line."
        )
        return GateVerdict("D", title, action, evidence, caveats)

    # Gate C requires a flattened curve, not merely a nonzero endpoint.
    if shape == "apparent_plateau":
        title, action = GATES["C"]
        if not ci_excludes_zero:
            caveats.append(
                "the residual's 95% CI includes zero, so the plateau is a shape "
                "claim without a magnitude claim behind it"
            )
        caveats.append(
            (survival_shape or {}).get(
                "caveat",
                "A flat trailing slope at the largest k bounds an "
                "information-limited floor from above; it does not establish one.",
            )
        )
        return GateVerdict("C", title, action, evidence, caveats)

    # Everything else that is nonzero: the curve has not flattened.
    title, action = GATES["B"]
    if shape is None:
        caveats.append(
            "no survival curve was computed, so there is no evidence the residual "
            "has stopped falling; gate B is the conservative default"
        )
    elif shape == "still_falling":
        caveats.append(
            (survival_shape or {}).get(
                "caveat", "The curve has not flattened. Add families."
            )
        )
        if (survival_shape or {}).get("excess_verdict") == "consistent_with_shared_difficulty":
            caveats.append(
                "The observed curve tracks a shared-difficulty model that has no "
                "floor, so the nonzero endpoint is most simply read as ordinary "
                "correlated difficulty at finite k. Extending the diversity axis "
                "is still the next step, but expect closure rather than a floor."
            )
    return GateVerdict("B", title, action, evidence, caveats)


@dataclass(slots=True)
class PredictionVerdict:
    name: str
    statement: str
    status: str  # "supported" | "not_supported" | "undetermined"
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.name,
            "statement": self.statement,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _ci_sign(result: BootstrapResult | None) -> str:
    """"positive" / "negative" / "spans_zero" / "unknown" for a CI."""
    if result is None or math.isnan(result.ci_low) or math.isnan(result.ci_high):
        return "unknown"
    if result.ci_low > 0:
        return "positive"
    if result.ci_high < 0:
        return "negative"
    return "spans_zero"


def evaluate_predictions(
    *,
    diff_hs_dd: BootstrapResult | None = None,
    clean_dd: BootstrapResult | None = None,
    q_contested_hs: float = float("nan"),
    q_contested_dd: float = float("nan"),
    conversion: Mapping[str, float] | None = None,
    race: Mapping[str, Any] | None = None,
    gate: GateVerdict | None = None,
) -> list[PredictionVerdict]:
    """Evaluate P1-P6 from protocol section 11 against the run's results."""
    out: list[PredictionVerdict] = []

    # P1 -- diversity reduction: C_DD < C_HS.
    sign = _ci_sign(diff_hs_dd)
    out.append(
        PredictionVerdict(
            "P1",
            "C_DD < C_HS (diversity reduces clean consensus failure)",
            {"positive": "supported", "negative": "not_supported",
             "spans_zero": "undetermined", "unknown": "undetermined"}[sign],
            (
                f"C_HS^clean - C_DD^clean = {diff_hs_dd.point:.4f} "
                f"[{diff_hs_dd.ci_low:.4f}, {diff_hs_dd.ci_high:.4f}]"
                if diff_hs_dd else "no paired comparison available"
            ),
            {"ci_sign": sign, "result": diff_hs_dd.to_dict() if diff_hs_dd else None},
        )
    )

    # P2 -- clean residual measurable. Magnitude deliberately not preregistered.
    if clean_dd is None:
        p2_status, p2_detail = "undetermined", "C_DD^clean was not estimated"
    elif math.isnan(clean_dd.point):
        p2_status, p2_detail = "undetermined", "C_DD^clean is undefined (no clean items)"
    elif clean_dd.ci_low > 0:
        p2_status = "supported"
        p2_detail = (
            f"C_DD^clean = {clean_dd.point:.4f} "
            f"[{clean_dd.ci_low:.4f}, {clean_dd.ci_high:.4f}], interval excludes zero"
        )
    else:
        p2_status = "not_supported"
        p2_detail = (
            f"C_DD^clean = {clean_dd.point:.4f} "
            f"[{clean_dd.ci_low:.4f}, {clean_dd.ci_high:.4f}], interval includes zero"
        )
    out.append(
        PredictionVerdict(
            "P2",
            "C_DD^clean > 0 is measurable (magnitude not preregistered)",
            p2_status,
            p2_detail,
            {"result": clean_dd.to_dict() if clean_dd else None},
        )
    )

    # P3 -- contested concentration rises with diversity.
    if math.isnan(q_contested_hs) or math.isnan(q_contested_dd):
        p3_status, p3_detail = "undetermined", "Q_contested unavailable in one condition"
    elif q_contested_dd > q_contested_hs:
        p3_status = "supported"
        p3_detail = f"Q_contested rose {q_contested_hs:.3f} -> {q_contested_dd:.3f} (HS -> DD)"
    else:
        p3_status = "not_supported"
        p3_detail = f"Q_contested did not rise: {q_contested_hs:.3f} -> {q_contested_dd:.3f}"
    out.append(
        PredictionVerdict(
            "P3", "Q_contested rises with internal diversity", p3_status, p3_detail,
            {"q_hs": q_contested_hs, "q_dd": q_contested_dd},
        )
    )

    # P4 -- error conversion: diversity converts consensus failure into
    # disagreement more than it merely removes error.
    conv = dict(conversion or {})
    ratio = conv.get("conversion_ratio", float("nan"))
    if math.isnan(ratio):
        p4_status, p4_detail = "undetermined", "conversion ratio undefined (no consensus error removed)"
    elif ratio >= 1.0:
        p4_status = "supported"
        p4_detail = (
            f"disagreement gained / consensus error removed = {ratio:.2f}: the "
            "removed failures reappeared as internally visible disagreement"
        )
    else:
        p4_status = "not_supported"
        p4_detail = (
            f"conversion ratio {ratio:.2f} < 1: diversity removed consensus error "
            "without a matching gain in disagreement"
        )
    out.append(
        PredictionVerdict("P4", "Family diversity converts consensus-wrong into disagreement",
                          p4_status, p4_detail, conv)
    )

    # P5 -- active learning under diversity.
    r = dict(race or {})
    verdict = r.get("verdict")
    if verdict in (None, "not_applicable", "undetermined"):
        p5_status, p5_detail = "undetermined", str(r.get("reason") or r.get("note") or "no race run")
    elif verdict in ("disagreement_wins", "tie"):
        p5_status = "supported"
        p5_detail = str(r.get("note"))
    else:
        p5_status = "not_supported"
        p5_detail = str(r.get("note"))
    out.append(
        PredictionVerdict(
            "P5",
            "Disagreement sampling matches or beats generic anchored targeting "
            "unless a clean shared core survives",
            p5_status, p5_detail, {"surplus": r.get("localization_surplus")},
        )
    )

    # P6 -- conditional failure-memory benefit. Only meaningful at gate C.
    at_gate_c = bool(gate and gate.gate in ("C", "D"))
    if not at_gate_c:
        p6_status = "undetermined"
        p6_detail = (
            f"P6 is conditional on a surviving clean shared core; this run is at "
            f"gate {gate.gate if gate else '?'}, so the condition is not met and "
            "the prediction is untested rather than refuted"
        )
    elif verdict == "targeting_wins":
        p6_status = "supported"
        p6_detail = "clean core survived DD and anchored targeting beat disagreement"
    elif verdict in ("disagreement_wins", "tie"):
        p6_status = "not_supported"
        p6_detail = (
            "a clean core survived DD but anchored targeting did not beat "
            "disagreement: the surviving core is not clustered enough to target"
        )
    else:
        p6_status, p6_detail = "undetermined", "clean core survived but no race was run"
    out.append(
        PredictionVerdict(
            "P6",
            "If a large clustered clean shared core survives DD, targeted "
            "anchored correction beats disagreement sampling",
            p6_status, p6_detail, {"gate": gate.gate if gate else None},
        )
    )
    return out


def render_report(
    *,
    experiment: str,
    conditions: Mapping[str, Mapping[str, Any]],
    panel: Mapping[str, Any] | None = None,
    comparisons: Sequence[BootstrapResult] = (),
    survival: Mapping[str, Any] | None = None,
    dependence: Mapping[str, Mapping[str, Any]] | None = None,
    race: Mapping[str, Any] | None = None,
    gate: GateVerdict | None = None,
    predictions: Sequence[PredictionVerdict] = (),
    warnings: Sequence[str] = (),
) -> str:
    """Render the run report as Markdown.

    Ordering is intentional: warnings first, then the gate, then the evidence.
    A reader who stops after the first screen should have seen the caveats, not
    just the headline.
    """
    lines: list[str] = [f"# M2-LLM run report: {experiment}", ""]

    if warnings:
        lines += ["## Warnings", ""]
        lines += [f"- {w}" for w in warnings]
        lines.append("")

    if gate:
        lines += [
            "## Decision gate",
            "",
            f"**Gate {gate.gate} — {gate.title}**",
            "",
            f"Next action: {gate.next_action}",
            "",
        ]
        if gate.caveats:
            lines.append("Caveats:")
            lines += [f"- {c}" for c in gate.caveats]
            lines.append("")

    if panel:
        counts = panel.get("status_counts", {})
        lines += [
            "## Human anchors",
            "",
            f"- items: {panel.get('n_items')}",
            f"- unanimous / majority / contested: "
            f"{counts.get('unanimous')} / {counts.get('majority')} / {counts.get('contested')}",
            f"- observed agreement: {_fmt(panel.get('observed_agreement'))}, "
            f"chance-corrected: {_fmt(panel.get('chance_corrected_agreement'))}",
            "",
        ]

    if conditions:
        lines += ["## Condition metrics", "",
                  "| condition | n | E | C_E | C_E^clean | D_E | Q_contested | invisible |",
                  "|---|---|---|---|---|---|---|---|"]
        for name, m in conditions.items():
            lines.append(
                f"| {name} | {m.get('n_items')} | {m.get('n_members')} | "
                f"{_fmt(m.get('C_E'))} | {_fmt(m.get('C_E_clean'))} | "
                f"{_fmt(m.get('D_E'))} | {_fmt(m.get('Q_contested'))} | "
                f"{_fmt(m.get('invisible_error_rate'))} |"
            )
        lines.append("")

    if comparisons:
        lines += ["## Paired comparisons (item bootstrap, 95% CI)", ""]
        lines += [f"- {c.format()}" for c in comparisons]
        lines.append("")

    if survival:
        lines += ["## Residual survival curve", ""]
        curve = survival.get("curve") or survival.get("clean_curve") or []
        if curve:
            lines += ["| k families | C_k^clean | sd | subsets |", "|---|---|---|---|"]
            for p in curve:
                lines.append(
                    f"| {p.get('k')} | {_fmt(p.get('mean'))} | {_fmt(p.get('sd'))} | "
                    f"{p.get('n_subsets')} |"
                )
            lines.append("")
        if survival.get("shape"):
            lines += [f"Shape: **{survival['shape']}** — {survival.get('reason','')}", ""]
        if survival.get("caveat"):
            lines += [f"> {survival['caveat']}", ""]

    if dependence:
        lines += ["## Error dependence", ""]
        for name, d in dependence.items():
            ex = d.get("excess_clean", {}) or {}
            hf = d.get("hierarchical_fit", {}) or {}
            lines += [
                f"### {name}",
                "",
                f"- mean pairwise rho: {_fmt(d.get('mean_pairwise_rho'))} "
                f"(single-factor model implies {_fmt(hf.get('implied_mean_rho'))})",
                f"- within-family rho: {_fmt((d.get('within_between') or {}).get('within_family'))}, "
                f"between-family: {_fmt((d.get('within_between') or {}).get('between_family'))}",
                f"- sigma_u: {_fmt(hf.get('sigma_u'))}",
                f"- observed C^clean {_fmt(ex.get('observed'))} vs hierarchical "
                f"null {_fmt(ex.get('hierarchical_model'))} "
                f"(excess {_fmt(ex.get('excess_over_hierarchical'))}); "
                f"naive independence null would say {_fmt(ex.get('independence_null'))}",
                "",
            ]

    if race:
        lines += ["## Localization race", ""]
        auc = race.get("auc") or {}
        for strat, v in auc.items():
            lines.append(f"- {strat}: AUC {_fmt(v)}")
        interp = race.get("interpretation") or {}
        if interp:
            lines += ["", f"Verdict: **{interp.get('verdict')}** — {interp.get('note')}", ""]
        else:
            lines.append("")

    if predictions:
        lines += ["## Preregistered predictions", "",
                  "| prediction | status | detail |", "|---|---|---|"]
        for p in predictions:
            detail = p.detail.replace("|", "\\|")
            lines.append(f"| {p.name} | {p.status} | {detail} |")
        lines.append("")

    lines += [
        "---",
        "",
        "Research discipline: first exhaust internal information → measure what "
        "genuinely remains → spend external information on that residual.",
        "",
    ]
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if math.isinf(v):
            return "∞"
        return f"{v:.4f}"
    return str(v)
