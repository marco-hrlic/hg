# Aborted live run (attempt 2): not results

Model `gpt-6.1-terra` via OpenAI, `GADLC_LLM_MAX_TOKENS=32000`, 2026-11-03.

toolshare/D and toolshare/G0 completed. G0's proposal call was well formed: 11,041 output tokens, normal stop, and all
33 ops accepted. This confirms that the 32k budget fixes attempt 1's empty outputs.

The harness then crashed in toolshare/G1 at P-0004 with `ReviewError: auto-q1-b1: endpoint Q-P-0004-C1 is added by
auto-q1, which is not accepted`. The stand-in reviewer `policy:conservative-v1` accepted an admissible auto `blocks`
edge but deferred its quarantined question node. That is a harness bug, now fixed: such edges are deferred along with
their node. The crash discarded G1's trace, so the run was restarted from scratch. Not scored.
