"""v0.3: opt-in LLM-assisted specification-graph construction and maintenance.

A thin layer over the existing engine. Models may *propose* and *challenge*;
graph changes only happen through reviewed ``GraphADLC.commit`` transactions.
"""

from .challenge import challenge_impact, influence_domain, scan_code_dependencies
from .ledger import Ledger
from .maintenance import MaintenanceLoop, StepResult, assess_preservation_evidence
from .model_adapter import (AnthropicMessagesClient, DirectImpactAnalyst, LLMNotConfigured, LLMProposer,
                            ScriptedClient, load_prompt)
from .projection import export_spec_markdown
from .proposal import Citation, Proposal, ProposalFormatError, ProposedOp, parse_proposal
from .review import Decision, NeedsRegeneration, ReviewError, apply_review, policy_reviewer, render_review
from .source import (SourceBundle, SourceCatalog, SourceDocument, git_diff, load_file_source, load_git_sources,
                     pack_sources)
from .validation import validate_proposal

__all__ = [
    "AnthropicMessagesClient", "Citation", "Decision", "DirectImpactAnalyst", "LLMNotConfigured", "LLMProposer",
    "Ledger", "MaintenanceLoop", "NeedsRegeneration", "Proposal", "ProposalFormatError", "ProposedOp",
    "ReviewError", "ScriptedClient", "SourceBundle", "SourceCatalog", "SourceDocument", "StepResult",
    "apply_review", "assess_preservation_evidence", "challenge_impact", "export_spec_markdown", "git_diff",
    "influence_domain", "load_file_source", "load_git_sources", "load_prompt", "pack_sources", "parse_proposal",
    "policy_reviewer", "render_review", "scan_code_dependencies", "validate_proposal",
]
