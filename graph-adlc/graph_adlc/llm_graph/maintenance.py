"""The maintenance loop: sources -> proposal -> validation -> review -> commit -> influence.

Also holds the *evidence-sufficiency gate* used before any (exploratory)
pruning: ``passed && current`` is never treated as proof of sufficient
coverage (v0.2 SC-14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..core import GraphADLC
from ..lifecycle import evidence_staleness
from ..models import Relation
from .challenge import challenge_impact
from .ledger import Ledger
from .model_adapter import LLMProposer
from .proposal import Proposal
from .review import Decision, apply_review, render_review
from .source import SourceBundle, SourceCatalog
from .validation import validate_proposal

Reviewer = Callable[[Proposal, dict[str, Any]], list[Decision]]


@dataclass
class StepResult:
    proposal: Proposal | None
    report: dict[str, Any] | None
    call_id: str | None
    call: dict[str, Any] | None
    review_text: str = ""
    outcome: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)


class MaintenanceLoop:
    def __init__(self, app: GraphADLC, proposer: LLMProposer | None = None,
                 catalog: SourceCatalog | None = None):
        self.app = app
        self.ledger = Ledger(app)
        self.proposer = proposer
        self.catalog = catalog or SourceCatalog()

    # -- propose + validate ------------------------------------------------------
    def propose(self, bundle: SourceBundle, change_event: dict[str, Any], *, call_key: str) -> StepResult:
        if self.proposer is None:
            raise RuntimeError("no proposer configured")
        self.catalog.register(bundle.documents)
        pid = self.ledger.next_proposal_id()
        g = self.app.graph()
        proposal, call = self.proposer.propose_update(g, bundle, change_event, proposal_id=pid, call_key=call_key)
        call_id = self.ledger.record_call(call)
        if proposal is None:
            stub = {"proposal_id": pid, "base_graph_revision": g.revision, "intent": "(unparseable model output)",
                    "operations": [], "origin": {"call_id": call_id}}
            self.ledger.save_proposal(stub, {"parse_error": call["parse"]["error"]}, "invalid")
            return StepResult(None, None, call_id, call, errors=[f"unparseable output: {call['parse']['error']}"])
        proposal.origin["call_id"] = call_id
        return self.validate(proposal, bundle, call_id=call_id, call=call)

    def validate(self, proposal: Proposal, bundle: SourceBundle, *, call_id: str | None = None,
                 call: dict[str, Any] | None = None) -> StepResult:
        self.catalog.register(bundle.documents)
        proposal, report = validate_proposal(self.app, proposal, bundle, self.catalog, self.ledger)
        self.ledger.save_proposal(proposal.to_dict(), report, "validated")
        return StepResult(proposal, report, call_id, call, review_text=render_review(proposal, report))

    def submit_external(self, proposal: Proposal, bundle: SourceBundle) -> StepResult:
        """Validate a proposal produced elsewhere (deterministic scan, human-authored)."""
        pid = self.ledger.next_proposal_id()
        proposal.proposal_id = pid
        proposal.base_graph_revision = self.app.graph().revision
        return self.validate(proposal, bundle)

    # -- review + commit ---------------------------------------------------------
    def review(self, step: StepResult, decisions: list[Decision], bundle: SourceBundle,
               approvals: list[str] | None = None) -> dict[str, Any]:
        assert step.proposal is not None and step.report is not None
        out = apply_review(self.app, self.ledger, step.proposal, step.report, decisions, bundle, self.catalog,
                           approvals)
        step.outcome = out
        return out

    # -- challenge -----------------------------------------------------------------
    def challenge(self, change: dict[str, Any], bundle: SourceBundle, *, use_llm: bool = False,
                  call_key: str | None = None) -> dict[str, Any]:
        self.catalog.register(bundle.documents)
        llm = self.proposer if use_llm else None
        res = challenge_impact(self.app, change, bundle, scan_proposal_id="SCAN", llm=llm,
                               llm_proposal_id="LLM-CHALLENGE", call_key=call_key)
        steps = {}
        steps["scan"] = self.submit_external(res["scan"], bundle) if res["scan"].operations or \
            res["scan"].unresolved_questions else None
        if res["llm_call"] is not None:
            call_id = self.ledger.record_call(res["llm_call"])
            if res["llm"] is not None:
                res["llm"].origin["call_id"] = call_id
                steps["llm"] = self.submit_external(res["llm"], bundle)
            else:
                steps["llm"] = StepResult(None, None, call_id, res["llm_call"],
                                          errors=[f"unparseable output: {res['llm_call']['parse']['error']}"])
        return {"domain": res["domain"], "steps": steps}


# ---------------------------------------------------------------------------
# Evidence sufficiency gate (SC-14)
# ---------------------------------------------------------------------------

def assess_preservation_evidence(app: GraphADLC, provider_id: str,
                                 affected_domain: dict[str, list[str]] | None) -> dict[str, Any]:
    """Which contract-preservation records may justify suppressing a consumer check?

    Beyond the engine's own rules (passed, current, same contract revisions), a
    record must declare ``details.coverage`` and that coverage must include
    every value of every dimension in the change's declared ``affected_domain``.
    If the change does not declare its affected domain, sufficiency is unknown
    and nothing is admissible.
    """
    g = app.graph()
    contracts = {e.dst for e in g.out_edges(provider_id, Relation.CONSTRAINED_BY)}
    admissible, refused = [], []
    for ev in app.store.list_evidence():
        if ev["kind"] != "contract_preservation" or ev["node_id"] != provider_id:
            continue
        reasons = []
        if ev["result"] != "passed":
            reasons.append("result is not passed")
        reasons += evidence_staleness(ev, g)
        if not contracts:
            reasons.append("provider has no explicit contract to preserve")
        cov = (ev.get("details") or {}).get("coverage")
        if not affected_domain:
            reasons.append("the change does not declare its affected input domain; sufficiency cannot be established")
        elif not isinstance(cov, dict):
            reasons.append("record declares no coverage; 'passed and current' is not proof of sufficient coverage")
        else:
            for dim, values in sorted(affected_domain.items()):
                missing = sorted(set(values) - set(cov.get(dim) or []))
                if missing:
                    reasons.append(f"coverage of {dim!r} lacks {missing} (covered: {sorted(cov.get(dim) or [])})")
        (refused if reasons else admissible).append({"evidence_id": ev["id"], "reasons": reasons})
    return {"provider_id": provider_id, "affected_domain": affected_domain, "admissible": admissible,
            "refused": refused,
            "policy": "pruning is disabled by default in v0.3; this gate only decides which evidence would be "
                      "eligible in an explicitly exploratory pruning condition"}
