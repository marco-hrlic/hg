# Aborted live run (attempt 1): not results

Model `gpt-6.1-terra` via OpenAI Chat Completions, 2026-11-03. Stopped by hand during toolshare/G2.

Every `propose_update` call returned `finish_reason: "length"` with **empty content** at exactly
`max_completion_tokens = 8000` output tokens (presumably all spent on hidden reasoning). Every proposal is therefore
`invalid` (unparseable), and the G0 and G1 graphs are empty. Arm D's single call completed normally (2412 output tokens).

Nothing here measures graph construction or maintenance quality. Not scored, not in RESULTS.md. The
output-token budget has to change before a rerun, and that change belongs to the protocol.
