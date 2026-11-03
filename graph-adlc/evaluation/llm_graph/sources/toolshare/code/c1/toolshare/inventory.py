"""Shelf availability for catalogued tools."""

_ON_SHELF: set[str] = set()


def mark_returned(tool_id: str) -> None:
    _ON_SHELF.add(tool_id)


def mark_lent(tool_id: str) -> None:
    _ON_SHELF.discard(tool_id)


def is_on_shelf(tool_id: str) -> bool:
    return tool_id in _ON_SHELF
