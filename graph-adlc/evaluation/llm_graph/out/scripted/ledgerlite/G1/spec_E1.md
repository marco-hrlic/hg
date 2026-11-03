# ledgerlite specification (after E1, arm G1)

> Generated from the specification graph at revision 2. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-BILLING** — Billing and clean accounting numbers _(source: ledgerlite:docs/requirements.md L5-5; via P-0001/CHG-2)_
  Small studios can bill clients and hand clean numbers to their accountant.
  - **CAP-ACCOUNTING-EXPORT** — Accounting export _(source: ledgerlite:docs/requirements.md L14-14; via P-0001/CHG-2)_
    Hand invoice data to the accountant.
    - **REQ-CSV-EXPORT** — Monthly CSV export _(source: ledgerlite:docs/requirements.md L16-16; via P-0001/CHG-2)_
      Finance can export all invoices of a month as a CSV file for the accountant's bookkeeping tool.
      Acceptance criteria: _none recorded yet_
      Blocked by open question(s): Q-CSV-LAYOUT
  - **CAP-INVOICING** — Invoicing _(source: ledgerlite:docs/requirements.md L7-7; via P-0001/CHG-2)_
    Produce invoices for clients.
    - **REQ-PDF-INVOICE** — PDF invoice _(source: ledgerlite:docs/requirements.md L9-9; via P-0001/CHG-2)_
      Each invoice can be downloaded as a PDF that a client can read and pay from.
      Acceptance criteria:
      - The PDF shows every line item, the tax and the total.
      - Amounts are shown in the studio's currency with two decimals.

## Contracts

- **CON-MONEY-FORMAT** — Money format [status: accepted] _(source: ledgerlite:docs/requirements.md L22-22; via P-0001/CHG-2)_
  Amounts are rendered with exactly two decimals and '.' as the decimal separator, followed by the ISO currency code.
  Applies to: CMP-MONEY

## Components (responsibilities only)

- **CMP-CSV** — CSV exporter _(source: ledgerlite:docs/architecture.md L9-9; via P-0001/CHG-2)_
  CSV exporter (ledgerlite/csv_export.py).
  Implements: REQ-CSV-EXPORT
- **CMP-MONEY** — Money formatting _(source: ledgerlite:docs/architecture.md L7-7; via P-0001/CHG-2)_
  Money formatting (ledgerlite/money.py).
- **CMP-PDF** — PDF invoice renderer _(source: ledgerlite:docs/architecture.md L8-8; via P-0001/CHG-2)_
  PDF invoice renderer (ledgerlite/pdf_invoice.py).
  Implements: REQ-PDF-INVOICE
  Depends on: CMP-MONEY

## Decisions

_None recorded._

## Open questions

- **Q-CSV-LAYOUT** (blocking) — CSV layout expected by the accountant (blocks REQ-CSV-EXPORT)
  Which columns and number format does the accountant's import expect?

## Proposed — not approved

- Unconfirmed relationship: CAP-ACCOUNTING-EXPORT refines REQ-CSV-EXPORT (LE-CAP-ACCOUNTING-EXPORT--refines--REQ-CSV-EXPORT)
- Unconfirmed relationship: CAP-INVOICING refines REQ-PDF-INVOICE (LE-CAP-INVOICING--refines--REQ-PDF-INVOICE)
- Unconfirmed relationship: GOAL-BILLING refines CAP-ACCOUNTING-EXPORT (LE-GOAL-BILLING--refines--CAP-ACCOUNTING-EXPORT)
- Unconfirmed relationship: GOAL-BILLING refines CAP-INVOICING (LE-GOAL-BILLING--refines--CAP-INVOICING)

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
