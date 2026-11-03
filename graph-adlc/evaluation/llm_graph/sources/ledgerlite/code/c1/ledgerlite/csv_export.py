"""Monthly CSV export for the accountant."""

import csv
import io

HEADER = ["invoice_id", "date", "client", "net", "tax", "total"]


def export_month(invoices: list[dict]) -> str:
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(HEADER)
    for inv in invoices:
        w.writerow([inv["id"], inv["date"], inv["client"], inv["net"], inv["tax"], inv["total"]])
    return out.getvalue()
