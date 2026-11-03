from decimal import Decimal

import pytest

from ledgerlite.money import format_amount

CASES = [
    (Decimal("0"), "0.00 EUR"),
    (Decimal("9.5"), "9.50 EUR"),
    (Decimal("250"), "250.00 EUR"),
    (Decimal("999.994"), "999.99 EUR"),
]


@pytest.mark.parametrize("amount,expected", CASES)
def test_format_amount(amount, expected):
    assert format_amount(amount, "EUR") == expected
