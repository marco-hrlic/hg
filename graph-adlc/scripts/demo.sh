#!/usr/bin/env bash
# Copy-paste replay of the payment-retry vertical slice with the gadlc CLI.
# Run from the graph-adlc/ directory after `pip install -e .`
# (or replace `gadlc` with `python -m graph_adlc`).
set -euo pipefail
DB=${DB:-demo.sqlite}
rm -f "$DB"

# A. Discovery: partial seed -> REQ-RETRY blocked, REQ-CHARGE / REQ-REPORT ready
gadlc init     --db "$DB" --seed fixtures/payment_seed.json
gadlc next     --db "$DB"

# B. Refinement: read-only preview, then atomic commit (graph rev 2)
gadlc propose  --db "$DB" --delta fixtures/retry_refinement.json
gadlc commit   --db "$DB" --delta fixtures/retry_refinement.json
gadlc frontier --db "$DB"

# C. Implementation: scoped context, work unit with spec-revision baseline, evidence
gadlc context  --db "$DB" REQ-RETRY
gadlc work create REQ-RETRY --db "$DB"
gadlc work start  WORK-1    --db "$DB"
gadlc work submit WORK-1    --db "$DB" --code-rev git:a1b2c3 --evidence TEST-RETRY:test_result:passed
gadlc status   --db "$DB"

# D/E. Contradiction: provider idempotency window -> explainable impact, stale evidence (graph rev 3)
gadlc propose  --db "$DB" --delta fixtures/idempotency_window_change.json
gadlc commit   --db "$DB" --delta fixtures/idempotency_window_change.json
gadlc evidence list --db "$DB"
gadlc next     --db "$DB"
gadlc work submit WORK-1 --db "$DB" --code-rev git:a1b2c3 --evidence TEST-RETRY || echo "(refused as expected: exit $?)"

# F. Reconciliation: inspect, update spec (graph rev 4), rebase, fresh evidence -> verified
gadlc reconcile WORK-1 --db "$DB"
gadlc commit   --db "$DB" --delta fixtures/retry_window_reconciliation.json
gadlc reconcile WORK-1 --db "$DB" --rebase
gadlc work submit WORK-1 --db "$DB" --code-rev git:d4e5f6 \
  --evidence TEST-RETRY:test_result:passed --evidence CON-IDEMPOTENCY:contract_check:passed
gadlc status   --db "$DB"
gadlc next     --db "$DB"
gadlc log      --db "$DB"
gadlc export   --db "$DB" --out snapshot.json
