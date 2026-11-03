"""Shared money formatting (CON-MONEY-FORMAT)."""

from decimal import ROUND_HALF_UP, Decimal


def format_amount(amount: Decimal, currency: str) -> str:
    q = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{q} {currency}"


def plain_amount(amount: Decimal) -> str:
    return str(amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
