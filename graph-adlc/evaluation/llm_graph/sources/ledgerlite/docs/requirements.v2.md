# Ledgerlite invoicing — requirements

Maintained by: Dana (finance operations). Version 2, 2026-09-16.

GOAL-BILLING — Small studios can bill clients and hand clean numbers to their accountant.

CAP-INVOICING — Produce invoices for clients.

REQ-PDF-INVOICE — Each invoice can be downloaded as a PDF that a client can read and pay from.
Acceptance criteria:
- The PDF shows every line item, the tax and the total.
- Amounts are shown in the studio's currency with two decimals.

CAP-ACCOUNTING-EXPORT — Hand invoice data to the accountant.

REQ-CSV-EXPORT — Finance can export all invoices of a month as a CSV file for the accountant's bookkeeping tool.
Acceptance criteria (see DR-001):
- The file has the columns invoice_id, date, client, net, tax, total.
- Amounts are plain numbers with "." as decimal separator, no thousands separators and no currency code.

Q-CSV-LAYOUT — Which columns and number format does the accountant's import expect? Resolved by DR-001.

CON-MONEY-FORMAT — Amounts are rendered with exactly two decimals and "." as the decimal separator, followed by the
ISO currency code (for example "250.00 EUR"). Status: accepted.
