import pytest

from toolshare import notifications


def test_ready_notice_is_short_plain_text():
    notifications.send_ready_notice("ana", "drill", "Thu 18:00")
    msg = notifications.OUTBOX[-1]
    assert msg["subject"] and len(msg["body"]) <= 500


def test_body_limit_enforced():
    with pytest.raises(ValueError):
        notifications.send_email("ana", "s", "x" * 501)
