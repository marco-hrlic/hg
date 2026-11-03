"""Hold queue and expiry."""

from collections import defaultdict, deque

from toolshare.inventory import is_on_shelf
from toolshare.notifications import send_email

PICKUP_WINDOW_HOURS = 48
_QUEUES: dict[str, deque] = defaultdict(deque)


def place_hold(member: str, tool: str) -> int:
    _QUEUES[tool].append(member)
    return len(_QUEUES[tool])


def next_in_queue(tool: str) -> str | None:
    if not is_on_shelf(tool) or not _QUEUES[tool]:
        return None
    return _QUEUES[tool][0]


def expire_hold(tool: str) -> str | None:
    """The first member did not collect within PICKUP_WINDOW_HOURS: tell them, drop them, return the next member."""
    if _QUEUES[tool]:
        lapsed = _QUEUES[tool].popleft()
        send_email(lapsed, f"Your hold on {tool} has lapsed",
                   f"You did not collect {tool} within {PICKUP_WINDOW_HOURS} hours, so it has gone to the next "
                   "member in the queue. You can place a new hold at any time.")
    return _QUEUES[tool][0] if _QUEUES[tool] else None
