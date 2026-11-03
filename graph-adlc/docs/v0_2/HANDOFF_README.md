# Graph-Driven ADLC — next-milestone handoff

This package converts the 2026-11-03 v0.1 implementation report into an implementation-ready **evaluation milestone**. It is intentionally scoped to testing the graph's influence accuracy, pruning safety and maintenance overhead before introducing agents or wider architecture.

**Read in this order:**

1. `HANDOFF.md` — research question, comparison methods, dataset, milestones and definition of done.
2. `EVALUATION_PROTOCOL.md` — oracle boundaries, normalized prediction schema, metrics and anti-leakage rules.
3. `AGENT_PROMPT.md` — compact kickoff text for a coding agent.

Place the documents in the existing `graph-adlc/` repo and ask the coding agent to execute `AGENT_PROMPT.md`. The `reference/` directory includes the user-supplied v0.1 implementation report and original prototype specification for a self-contained handoff. The agent must still inspect the actual repository rather than trust the report alone. The package contains **instructions only**, not an executable evaluation or fabricated benchmark data.
