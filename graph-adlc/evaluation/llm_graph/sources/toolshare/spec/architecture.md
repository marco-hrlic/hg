# Toolshare — Architecture notes

Author: Sam. Last edited 2026-09-02.

- CMP-HOLDS, the holds service (`toolshare/holds.py`), owns the hold queue and hold expiry. It implements
  REQ-PLACE-HOLD and REQ-HOLD-EXPIRY.
- CMP-INVENTORY, the inventory module (`toolshare/inventory.py`), tracks whether a tool is on the shelf. The holds
  service reads availability from inventory.
- CMP-NOTIFY, the notification service (`toolshare/notifications.py`), sends member emails and implements
  REQ-READY-NOTICE. Every message it sends must follow CON-NOTICE-FORMAT.
- Tests: TEST-HOLD-QUEUE (`tests/test_holds.py`) checks queue order and expiry. TEST-NOTICE-FORMAT
  (`tests/test_notifications.py`) checks messages against CON-NOTICE-FORMAT.
