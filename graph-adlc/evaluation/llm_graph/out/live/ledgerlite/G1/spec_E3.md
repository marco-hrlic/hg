# ledgerlite specification (after E3, arm G1)

> Generated from the specification graph at revision 4. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-BILLING** — Billing and clean accounting numbers _(source: ledgerlite:docs/requirements.md L5-5; via P-0001/CHG-2)_
  Small studios can bill clients and hand clean numbers to their accountant.
  - **CAP-ACCOUNTING-EXPORT** — Accounting export _(source: ledgerlite:docs/requirements.md L14-14; via P-0001/CHG-2)_
    Hand invoice data to the accountant.
    - **REQ-CSV-EXPORT** — Monthly CSV export _(source: ledgerlite:docs/requirements.md L16-17; ledgerlite:docs/requirements.md L16-19; via P-0003/CHG-4)_
      Finance can export all invoices of a month as a CSV file for the accountant's bookkeeping tool.
      Acceptance criteria:
      - The file has the columns invoice_id, date, client, net, tax, total.
      - Amounts are plain numbers with "." as decimal separator, no thousands separators and no currency code.
      Constrained by: CON-CSV-LAYOUT
      Blocked by open question(s): Q-P-0002-C2
  - **CAP-INVOICING** — Invoicing _(source: ledgerlite:docs/requirements.md L7-7; via P-0001/CHG-2)_
    Produce invoices for clients.
    - **REQ-PDF-INVOICE** — Downloadable PDF invoice _(source: ledgerlite:docs/requirements.md L9-12; via P-0001/CHG-2)_
      Each invoice can be downloaded as a PDF that a client can read and pay from.
      Acceptance criteria:
      - The PDF shows every line item, the tax and the total.
      - Amounts are shown in the studio's currency with two decimals.
      Blocked by open question(s): Q-P-0002-C3

## Contracts

- **CON-CSV-LAYOUT** — Accountant CSV import layout [status: accepted] _(source: ledgerlite:docs/decisions/DR-001-csv-layout.md L3-8; via P-0002/CHG-3)_
  The accountant's bookkeeping tool imports columns invoice_id, date, client, net, tax, total. Amounts must be plain numbers with "." as the decimal separator, no thousands separators and no currency code. Rows with unparseable amounts are silently dropped.
  Applies to: REQ-CSV-EXPORT
- **CON-MONEY-FORMAT** — Money format [status: accepted] _(source: ledgerlite:docs/requirements.md L22-23; via P-0001/CHG-2)_
  Amounts are rendered with exactly two decimals and "." as the decimal separator, followed by the ISO currency code (for example "250.00 EUR").
  Applies to: CMP-MONEY

## Components (responsibilities only)

- **CMP-CSV** — CSV exporter _(source: ledgerlite:docs/architecture.md L9-9; ledgerlite:docs/architecture.md L11-11; via P-0001/CHG-2)_
  Exports invoice data as CSV; the architecture states that amounts are written raw without the money formatter.
  Implements: REQ-CSV-EXPORT
  Depends on: CMP-MONEY
- **CMP-MONEY** — Money formatting _(source: ledgerlite:docs/architecture.md L7-7; via P-0001/CHG-2)_
  Shared money formatter.
- **CMP-PDF** — PDF invoice renderer _(source: ledgerlite:docs/architecture.md L8-8; git:ledgerlite L1-1; via P-0001/CHG-2)_
  Invoice renderer; the current repository file describes its implementation as a text layout stub.
  Implements: REQ-PDF-INVOICE
  Depends on: CMP-MONEY

## Decisions

- **Q-CSV-LAYOUT** — Accountant CSV layout: Which columns and number format does the accountant's import expect? _(source: ledgerlite:docs/requirements.md L18-20; ledgerlite:docs/decisions/DR-001-csv-layout.md L3-3; via P-0002/CHG-3)_

## Open questions

- **Q-P-0002-C1** (blocking) — Resolve: If the accepted money-format contract applies to CSV exports, its required currency code … (blocks CON-MONEY-FORMAT)
  If the accepted money-format contract applies to CSV exports, its required currency code conflicts with the CSV layout's prohibition on currency codes. The sources do not establish that scope.
- **Q-P-0002-C2** (blocking) — Resolve: CSV amount formatting for supported inputs (blocks CMP-CSV, REQ-CSV-EXPORT)
  The CSV exporter now formats amounts rather than writing them raw. Does this formatting produce valid plain numbers for every supported input, given that the code does not explicitly validate amount values?
- **Q-P-0002-C3** (blocking) — Resolve: The invoice renderer is a text-layout stub, so the cited implementation alone does not … (blocks CMP-PDF, REQ-PDF-INVOICE)
  The invoice renderer is a text-layout stub, so the cited implementation alone does not establish that a downloadable PDF is produced.
- **Q-P-0003-C1** (blocking) — Resolve: The architecture says the CSV exporter writes raw amounts and does not use the money … (blocks CMP-CSV)
  The architecture says the CSV exporter writes raw amounts and does not use the money formatter, while the current source calls that formatter for the exported amounts.

## Proposed — not approved

- Unconfirmed relationship: CAP-ACCOUNTING-EXPORT refines REQ-CSV-EXPORT (LE-CAP-ACCOUNTING-EXPORT--refines--REQ-CSV-EXPORT)
- Unconfirmed relationship: CAP-INVOICING refines REQ-PDF-INVOICE (LE-CAP-INVOICING--refines--REQ-PDF-INVOICE)
- Unconfirmed relationship: GOAL-BILLING refines CAP-ACCOUNTING-EXPORT (LE-GOAL-BILLING--refines--CAP-ACCOUNTING-EXPORT)
- Unconfirmed relationship: GOAL-BILLING refines CAP-INVOICING (LE-GOAL-BILLING--refines--CAP-INVOICING)
- Deferred update_node of REQ-CSV-EXPORT (P-0002/op-1): The revised requirements add CSV acceptance criteria.
- Deferred update_node of CMP-CSV (P-0003/op-2): Record the current implementation without silently discarding the conflicting architecture statement.

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
