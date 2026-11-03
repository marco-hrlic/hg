# Ledgerlite — architecture

Author: Omar. Version 1, 2026-09-01.

| Id | Component | File | Notes |
|---|---|---|---|
| CMP-MONEY | Money formatting | `ledgerlite/money.py` | Shared formatter. Must follow CON-MONEY-FORMAT. |
| CMP-PDF | PDF invoice renderer | `ledgerlite/pdf_invoice.py` | Implements REQ-PDF-INVOICE. Uses CMP-MONEY for every amount. |
| CMP-CSV | CSV exporter | `ledgerlite/csv_export.py` | Implements REQ-CSV-EXPORT. |

The CSV exporter writes raw decimal amounts straight from the database and does not use the money formatter.

TEST-MONEY-FORMAT (`tests/test_money.py`) checks the formatter against CON-MONEY-FORMAT with a table of sample amounts.
