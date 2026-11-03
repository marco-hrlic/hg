"""Member notifications (plain-text email)."""

OUTBOX: list[dict] = []
MAX_BODY = 500


def send_email(member: str, subject: str, body: str) -> None:
    if len(body) > MAX_BODY:
        raise ValueError("notification body too long")
    OUTBOX.append({"to": member, "subject": subject, "body": body})


def send_ready_notice(member: str, tool: str, deadline: str) -> None:
    send_email(member, f"{tool} is ready for pickup", f"Your hold on {tool} is ready. Collect it by {deadline}.")
