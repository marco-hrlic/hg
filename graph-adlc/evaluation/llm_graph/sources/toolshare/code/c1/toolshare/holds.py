"""Hold queue and expiry."""

from collections import defaultdict, deque

from toolshare.inventory import is_on_shelf

_QUEUES: dict[str, deque] = defaultdict(deque)


def place_hold(member: str, tool: str) -> int:
    _QUEUES[tool].append(member)
    return len(_QUEUES[tool])


def next_in_queue(tool: str) -> str | None:
    if not is_on_shelf(tool) or not _QUEUES[tool]:
        return None
    return _QUEUES[tool][0]


def expire_hold(tool: str) -> str | None:
    """The first member did not collect in time: drop them and return the next member, if any."""
    if _QUEUES[tool]:
        _QUEUES[tool].popleft()
    return _QUEUES[tool][0] if _QUEUES[tool] else None
