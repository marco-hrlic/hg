# Toolshare — Requirements

Owner: Priya. Status: draft 4 (2026-10-05).

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
Acceptance criteria:
- The pickup window is 48 hours from the moment the ready notice is sent.
- When the window passes, the hold lapses and the next member in the queue is offered the tool.

Q-PICKUP-WINDOW — How long is the pickup window? Decided by the board on 2026-09-10: 48 hours from the ready
notice. Resolved.

## Capability: Member notifications

CAP-NOTIFY — Members are told when something they are waiting for changes.

REQ-READY-NOTICE — When a held tool is ready to be picked up, we notify the member by email.
Acceptance criteria:
- The email names the tool and the pickup deadline.

CON-NOTICE-FORMAT — Notification messages are plain text. They are sent by email or, if the member prefers, by
SMS. An SMS body is at most 160 characters; an email has a subject line and at most 500 characters of body text.
Every message states the desk opening hours (Tue/Thu/Sat). Changed 2026-10-05 after member feedback. Status:
accepted.
