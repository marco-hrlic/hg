"""Monthly CSV export for the accountant."""

import csv
import io

from ledgerlite.money import format_amount

HEADER = ["invoice_id", "date", "client", "net", "tax", "total"]


def _amount(value) -> str:
    # Reuse the shared formatter and drop the currency code so every amount has exactly two decimals.
    return format_amount(value, "").strip()


def export_month(invoices: list[dict]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(HEADER)
    for inv in invoices:
        w.writerow([inv["id"], inv["date"], inv["client"], _amount(inv["net"]), _amount(inv["tax"]),
                    _amount(inv["total"])])
    return out.getvalue()
