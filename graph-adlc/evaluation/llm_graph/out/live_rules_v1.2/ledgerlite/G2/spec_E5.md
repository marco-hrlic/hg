# ledgerlite specification (after E5, arm G2)

> Generated from the specification graph at revision 9. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-BILLING** — Billing and accounting _(source: ledgerlite:docs/requirements.md L5-5; via P-0001/CHG-2)_
  Small studios can bill clients and hand clean numbers to their accountant.
  - **CAP-ACCOUNTING-EXPORT** — Accounting export _(source: ledgerlite:docs/requirements.md L14-14; via P-0001/CHG-2)_
    Hand invoice data to the accountant.
    - **REQ-CSV-EXPORT** — Monthly CSV export _(source: ledgerlite:docs/requirements.md L16-17; ledgerlite:docs/requirements.md L16-19; via P-0004/CHG-5)_
      Finance can export all invoices of a month as a CSV file for the accountant's bookkeeping tool.
      Acceptance criteria:
      - The file has the columns invoice_id, date, client, net, tax, total.
      - Amounts are plain numbers with "." as decimal separator, no thousands separators and no currency code.
      Constrained by: DR-001
      Blocked by open question(s): Q-P-0002-C1, Q-P-0005-C2, Q-P-0007-C1, Q-P-0008-C1
  - **CAP-INVOICING** — Invoicing _(source: ledgerlite:docs/requirements.md L7-7; via P-0001/CHG-2)_
    Produce invoices for clients.
    - **REQ-PDF-INVOICE** — Downloadable PDF invoice _(source: ledgerlite:docs/requirements.md L9-12; via P-0001/CHG-2)_
      Each invoice can be downloaded as a PDF that a client can read and pay from.
      Acceptance criteria:
      - The PDF shows every line item, the tax and the total.
      - Amounts are shown in the studio's currency with two decimals.
      Blocked by open question(s): Q-P-0002-C3

## Contracts

- **CON-MONEY-FORMAT** — Money format [status: accepted] _(source: ledgerlite:docs/requirements.md L22-23; via P-0001/CHG-2)_
  Amounts are rendered with exactly two decimals and "." as the decimal separator, followed by the ISO currency code (for example "250.00 EUR").
  Applies to: CMP-MONEY
- **DR-001** — Accountant's CSV import layout [status: accepted] _(source: ledgerlite:docs/decisions/DR-001-csv-layout.md L1-8; via P-0002/CHG-3)_
  The accountant's bookkeeping tool imports CSV columns invoice_id, date, client, net, tax, total. Amounts must be plain numbers with a dot decimal separator, no thousands separators and no currency code. The import silently drops rows with amounts it cannot parse.
  Applies to: REQ-CSV-EXPORT

## Components (responsibilities only)

- **CMP-CSV** — CSV exporter _(source: ledgerlite:docs/architecture.md L9-11; via P-0001/CHG-2)_
  Exports invoice data as CSV; the current implementation writes the supplied amount values without the shared formatter.
  Implements: REQ-CSV-EXPORT
  Depends on: CMP-MONEY
- **CMP-MONEY** — Money formatting _(source: ledgerlite:docs/architecture.md L7-7; via P-0001/CHG-2)_
  Shared money formatter.
- **CMP-PDF** — PDF invoice renderer _(source: ledgerlite:docs/architecture.md L8-8; git:ledgerlite L1-1; via P-0001/CHG-2)_
  Invoice renderer; the current repository describes its implementation as a text layout stub.
  Implements: REQ-PDF-INVOICE
  Depends on: CMP-MONEY

## Decisions

- **Q-CSV-LAYOUT** — Accountant's CSV layout: Which columns and number format does the accountant's import expect? _(source: ledgerlite:docs/requirements.md L18-20; ledgerlite:docs/requirements.md L21-21; via P-0002/CHG-3)_

## Open questions

- **Q-P-0002-C1** (blocking) — Resolve: CSV amount input validity and handling (blocks CMP-CSV, REQ-CSV-EXPORT)
  The CSV exporter now formats amount values using the shared formatter. Whether supplied values are valid finite decimals, and how incompatible values are handled, remains unclear.
- **Q-P-0002-C2** (blocking) — Resolve: If CON-MONEY-FORMAT applies to CSV amounts, its required currency code conflicts with … (blocks CON-MONEY-FORMAT)
  If CON-MONEY-FORMAT applies to CSV amounts, its required currency code conflicts with DR-001's prohibition on currency codes. The sources do not establish that scope.
- **Q-P-0002-C3** (blocking) — Resolve: The identified PDF renderer is a text layout stub returning lines rather than a PDF; the … (blocks CMP-PDF, REQ-PDF-INVOICE)
  The identified PDF renderer is a text layout stub returning lines rather than a PDF; the supplied code does not establish that a downloadable PDF is produced elsewhere.
- **Q-P-0004-C1** (blocking) — Resolve: The architecture document says the CSV exporter does not use the money formatter, but the … (blocks CMP-CSV, CMP-MONEY)
  The architecture document says the CSV exporter does not use the money formatter, but the current code imports and calls it.
- **Q-P-0004-C2** (blocking) — Resolve: If the accepted money format contract applies to CSV amounts or every formatter call, its … (blocks CMP-CSV, CMP-MONEY, CON-MONEY-FORMAT, DR-001)
  If the accepted money format contract applies to CSV amounts or every formatter call, its currency-code requirement conflicts with the CSV layout and the exporter's empty-currency call. Its scope remains unresolved.
- **Q-P-0005-C1** (blocking) — Resolve: The architecture says the CSV exporter does not use the money formatter, but the current … (blocks CMP-CSV, CMP-MONEY)
  The architecture says the CSV exporter does not use the money formatter, but the current code imports and calls it.
- **Q-P-0005-C2** (blocking) — Resolve: If the money format contract applies to CSV amounts, its currency-code requirement … (blocks CON-MONEY-FORMAT, DR-001, REQ-CSV-EXPORT)
  If the money format contract applies to CSV amounts, its currency-code requirement conflicts with the CSV layout's prohibition on currency codes.
- **Q-P-0006-C1** (blocking) — Resolve: The architecture says the CSV exporter does not use the money formatter, but the supplied … (blocks CMP-CSV, CMP-MONEY)
  The architecture says the CSV exporter does not use the money formatter, but the supplied code calls it.
- **Q-P-0007-C1** (blocking) — Resolve: CSV amounts that round to 1,000.00 or more receive thousands separators from the shared … (blocks CMP-CSV, CMP-MONEY, DR-001, REQ-CSV-EXPORT)
  CSV amounts that round to 1,000.00 or more receive thousands separators from the shared formatter, contrary to the CSV requirement and accepted import layout. The decision warns that unparseable amounts cause rows to be silently dropped.
- **Q-P-0008-C1** (blocking) — Resolve: CSV amounts that round to 1,000.00 or more receive thousands separators, contrary to the … (blocks CMP-CSV, CMP-MONEY, DR-001, REQ-CSV-EXPORT)
  CSV amounts that round to 1,000.00 or more receive thousands separators, contrary to the CSV requirement and accepted import layout. Unparseable amounts can cause rows to be silently dropped.

## Proposed — not approved

- Unconfirmed relationship: CAP-ACCOUNTING-EXPORT refines REQ-CSV-EXPORT (LE-CAP-ACCOUNTING-EXPORT--refines--REQ-CSV-EXPORT)
- Unconfirmed relationship: CAP-INVOICING refines REQ-PDF-INVOICE (LE-CAP-INVOICING--refines--REQ-PDF-INVOICE)
- Unconfirmed relationship: GOAL-BILLING refines CAP-ACCOUNTING-EXPORT (LE-GOAL-BILLING--refines--CAP-ACCOUNTING-EXPORT)
- Unconfirmed relationship: GOAL-BILLING refines CAP-INVOICING (LE-GOAL-BILLING--refines--CAP-INVOICING)
- Deferred update_node of REQ-CSV-EXPORT (P-0002/op-1): Version 2 adds explicit CSV acceptance criteria.
- Deferred update_node of CMP-CSV (P-0004/op-1): The exporter no longer writes supplied amount values directly.
- Deferred update_node of CMP-CSV (P-0005/op-1): The current component description says supplied amounts are written without the shared formatter, but the code now calls it.
- Deferred add_node of Q-P-0006-C2 (P-0006/auto-q2): validator: contradiction requires a human decision, not a silent replacement
- Deferred add_edge of Q-P-0006-C2 blocks CON-MONEY-FORMAT (P-0006/auto-q2-b1): validator: open contradiction blocks the affected node until decided
- Deferred add_edge of Q-P-0006-C2 blocks DR-001 (P-0006/auto-q2-b2): validator: open contradiction blocks the affected node until decided
- Deferred add_edge of Q-P-0006-C2 blocks REQ-CSV-EXPORT (P-0006/auto-q2-b3): validator: open contradiction blocks the affected node until decided
- Deferred add_edge of Q-P-0006-C2 blocks CMP-CSV (P-0006/auto-q2-b4): validator: open contradiction blocks the affected node until decided
- Deferred update_node of CMP-CSV (P-0006/op-2): The existing component description says amounts are written without the shared formatter, contrary to the supplied code.
- Deferred add_node of Q-P-0007-C2 (P-0007/auto-q2): validator: contradiction requires a human decision, not a silent replacement
- Deferred add_edge of Q-P-0007-C2 blocks CMP-CSV (P-0007/auto-q2-b1): validator: open contradiction blocks the affected node until decided
- Deferred add_edge of Q-P-0007-C2 blocks CMP-MONEY (P-0007/auto-q2-b2): validator: open contradiction blocks the affected node until decided
- Deferred update_node of CMP-MONEY (P-0007/op-1): Record the current formatter behavior without changing the accepted contract.
- Deferred update_node of CMP-CSV (P-0007/op-2): Replace the stale description of raw amount output with the behavior shown by the current code.
- Deferred add_node of Q-P-0008-C2 (P-0008/auto-q2): validator: contradiction requires a human decision, not a silent replacement
- Deferred add_edge of Q-P-0008-C2 blocks CMP-CSV (P-0008/auto-q2-b1): validator: open contradiction blocks the affected node until decided
- Deferred add_edge of Q-P-0008-C2 blocks CMP-MONEY (P-0008/auto-q2-b2): validator: open contradiction blocks the affected node until decided
- Deferred update_node of CMP-MONEY (P-0008/op-1): Record the changed formatter behavior and the available plain-number helper.
- Deferred update_node of CMP-CSV (P-0008/op-2): Correct the stale component description to match the current code without changing the CSV requirement.

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
