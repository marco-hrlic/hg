from toolshare import holds, inventory


def test_queue_order_and_expiry():
    inventory.mark_returned("drill")
    holds.place_hold("ana", "drill")
    holds.place_hold("ben", "drill")
    assert holds.next_in_queue("drill") == "ana"
    assert holds.expire_hold("drill") == "ben"
