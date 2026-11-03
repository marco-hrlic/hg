# Toolshare — Requirements

Owner: Priya. Status: draft 1 (2026-09-02).

## Goal

GOAL-FAIR-ACCESS — Members can borrow shared tools fairly and predictably.

## Capability: Holds

CAP-HOLDS — Members can queue for tools that are not currently on the shelf.

REQ-PLACE-HOLD — A member can place a hold on a catalogued tool that is on loan.
Acceptance criteria:
- A member without overdue items can place a hold on any catalogued tool.
- Holds on the same tool are served first come, first served.

REQ-HOLD-EXPIRY — When a held tool becomes available, the member has a pickup window to collect it. If they do not
collect it in time, the hold lapses and the tool passes to the next member in the queue.
The length of the pickup window is not decided yet (see Q-PICKUP-WINDOW).

Q-PICKUP-WINDOW — How long is the pickup window? The desk is staffed three days a week. This blocks REQ-HOLD-EXPIRY.

## Capability: Member notifications

CAP-NOTIFY — Members are told when something they are waiting for changes.

REQ-READY-NOTICE — When a held tool is ready for pickup, the member is notified by email.
Acceptance criteria:
- The email names the tool and the pickup deadline.

CON-NOTICE-FORMAT — Notification messages are plain-text emails with a subject line and at most 500 characters of
body text. Agreed with the desk volunteer coordinator. Status: accepted.
