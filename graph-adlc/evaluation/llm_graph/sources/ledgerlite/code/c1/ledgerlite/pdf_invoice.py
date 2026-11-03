"""PDF invoice rendering (text layout stub)."""

from ledgerlite.money import format_amount


def render_lines(invoice: dict) -> list[str]:
    cur = invoice["currency"]
    lines = [f"{item['label']}: {format_amount(item['amount'], cur)}" for item in invoice["items"]]
    lines.append(f"Tax: {format_amount(invoice['tax'], cur)}")
    lines.append(f"Total: {format_amount(invoice['total'], cur)}")
    return lines
