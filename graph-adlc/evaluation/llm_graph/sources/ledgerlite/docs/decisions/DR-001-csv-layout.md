# DR-001: CSV layout for the accountant

Status: decided, 2026-09-15. Decided by Dana with the studio's accountant. Resolves Q-CSV-LAYOUT.

The accountant's bookkeeping tool imports a CSV with the columns `invoice_id,date,client,net,tax,total`.

Amounts must be plain numbers with "." as the decimal separator and no thousands separators and no currency code
(for example `1234.50`). The import silently drops any row whose amount it cannot parse.
