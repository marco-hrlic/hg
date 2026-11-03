# toolshare specification (after E5, arm G2)

> Generated from the specification graph at revision 9. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-FAIR-ACCESS** — Fair access to shared tools _(source: toolshare:spec/requirements.md L7-7; via P-0001/CHG-2)_
  Members can borrow shared tools fairly and predictably.
  - **CAP-HOLDS** — Holds _(source: toolshare:spec/requirements.md L9-11; via P-0001/CHG-2)_
    Members can queue for tools that are not currently on the shelf.
    - **REQ-HOLD-EXPIRY** — Expire uncollected holds _(source: toolshare:spec/requirements.md L18-20; via P-0001/CHG-2)_
      When a held tool becomes available, the member has a pickup window. If it is not collected in time, the hold lapses and the tool passes to the next member in the queue. The window length is undecided.
      Acceptance criteria: _none recorded yet_
      Depends on: REQ-READY-NOTICE (unconfirmed)
      Blocked by open question(s): Q-P-0002-C1, Q-P-0004-C1, Q-P-0005-C1, Q-P-0006-C1, Q-P-0007-C3, Q-P-0008-C1
    - **REQ-PLACE-HOLD** — Place a hold _(source: toolshare:spec/requirements.md L13-16; via P-0001/CHG-2)_
      A member can place a hold on a catalogued tool that is on loan.
      Acceptance criteria:
      - A member without overdue items can place a hold on any catalogued tool.
      - Holds on the same tool are served first come, first served.
  - **CAP-NOTIFY** — Member notifications _(source: toolshare:spec/requirements.md L24-26; via P-0001/CHG-2)_
    Members are told when something they are waiting for changes.
    - **REQ-READY-NOTICE** — Ready-for-pickup notice _(source: toolshare:spec/requirements.md L28-30; via P-0001/CHG-2)_
      When a held tool is ready for pickup, the member is notified by email.
      Acceptance criteria:
      - The email names the tool and the pickup deadline.
      Blocked by open question(s): Q-P-0007-C2

## Contracts

- **CON-NOTICE-FORMAT** — Notification format [status: accepted] _(source: toolshare:spec/requirements.md L32-33; via P-0001/CHG-2)_
  Notification messages are plain-text emails with a subject line and at most 500 characters of body text.
  Applies to: CMP-NOTIFY

## Components (responsibilities only)

- **CMP-HOLDS** — Holds service _(source: toolshare:spec/architecture.md L5-6; git:toolshare L23-30; git:toolshare L8-9; via P-0004/CHG-5)_
  Owns the hold queue and hold expiry. Its expiry function removes the first queued member, sends that member a lapse email referring to a 48-hour window, and returns the next queued member if any; it does not check elapsed time.
  Implements: REQ-HOLD-EXPIRY, REQ-PLACE-HOLD
  Depends on: CMP-INVENTORY, CMP-NOTIFY
- **CMP-INVENTORY** — Inventory module _(source: toolshare:spec/architecture.md L7-8; via P-0001/CHG-2)_
  Tracks whether a tool is on the shelf.
- **CMP-NOTIFY** — Notification service _(source: toolshare:spec/architecture.md L9-10; via P-0001/CHG-2)_
  Sends member emails.
  Implements: REQ-READY-NOTICE

## Decisions

- **Q-PICKUP-WINDOW** — Pickup window length: How long is the pickup window? The desk is staffed three days a week. The board decided on 2026-09-10 that it is 48 hours from the ready notice. _(source: toolshare:spec/requirements.md L20-22; toolshare:spec/requirements.md L24-25; via P-0002/CHG-3)_

## Open questions

- **Q-P-0002-C1** (blocking) — Resolve: The expiry function removes the first queued member whenever called, without checking the … (blocks REQ-HOLD-EXPIRY)
  The expiry function removes the first queued member whenever called, without checking the ready-notice time or elapsed window. If callers do not enforce the timing, a hold can lapse before the required 48 hours.
- **Q-P-0004-C1** (blocking) — Resolve: The expiry function removes a queued member without checking the ready-notice time or … (blocks REQ-HOLD-EXPIRY)
  The expiry function removes a queued member without checking the ready-notice time or elapsed window. If a caller invokes it early, the hold lapses before the required 48 hours.
- **Q-P-0005-C1** (blocking) — Resolve: expire_hold removes the first queued member without checking when a ready notice was sent … (blocks CMP-HOLDS, REQ-HOLD-EXPIRY)
  expire_hold removes the first queued member without checking when a ready notice was sent or whether 48 hours have elapsed. An early call would lapse the hold before the required window ends.
- **Q-P-0006-C1** (blocking) — Resolve: expire_hold removes the first queued member without checking the ready-notice time or … (blocks CMP-HOLDS, REQ-HOLD-EXPIRY)
  expire_hold removes the first queued member without checking the ready-notice time or elapsed window. An early call would lapse the hold before the required 48 hours; caller-side enforcement is not established by the supplied code.
- **Q-P-0007-C1** (blocking) — Resolve: The contract requires every message to state desk opening hours, but the supplied … (blocks CON-NOTICE-FORMAT)
  The contract requires every message to state desk opening hours, but the supplied ready-notice and lapse-message templates do not include them.
- **Q-P-0007-C2** (blocking) — Resolve: The ready-notice requirement specifies email, while the changed contract permits SMS when … (blocks CON-NOTICE-FORMAT, REQ-READY-NOTICE)
  The ready-notice requirement specifies email, while the changed contract permits SMS when preferred. It is unclear whether an SMS may replace the required email for a ready notice.
- **Q-P-0007-C3** (blocking) — Resolve: The expiry function removes the first queued member without checking the ready-notice … (blocks CMP-HOLDS, REQ-HOLD-EXPIRY)
  The expiry function removes the first queued member without checking the ready-notice time or elapsed window. An early call could lapse a hold before 48 hours; caller-side timing enforcement is not established by the supplied code.
- **Q-P-0008-C1** (blocking) — Resolve: The expiry function removes the first queued member without checking elapsed time. An … (blocks CMP-HOLDS, REQ-HOLD-EXPIRY)
  The expiry function removes the first queued member without checking elapsed time. An early call could lapse the hold before the required 48 hours; caller-side enforcement is not established.
- **Q-P-0008-C2** (blocking) — Resolve: The contract requires desk opening hours in every message, but the supplied ready-notice … (blocks CON-NOTICE-FORMAT)
  The contract requires desk opening hours in every message, but the supplied ready-notice and lapse-message templates omit them.

## Proposed — not approved

- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-HOLDS (LE-GOAL-FAIR-ACCESS--refines--CAP-HOLDS)
- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-NOTIFY (LE-GOAL-FAIR-ACCESS--refines--CAP-NOTIFY)
- Unconfirmed relationship: REQ-HOLD-EXPIRY depends on REQ-READY-NOTICE (LE-REQ-HOLD-EXPIRY--depends_on--REQ-READY-NOTICE)
- Deferred update_node of REQ-HOLD-EXPIRY (P-0002/op-1): Draft 2 decides the window length and adds acceptance criteria.
- Deferred update_node of REQ-HOLD-EXPIRY (P-0004/op-1): The requirements document records the board's decision and acceptance criteria, replacing the undecided window.
- Deferred update_node of REQ-HOLD-EXPIRY (P-0005/op-1): The requirements document specifies the previously undecided window and its acceptance criteria.
- Deferred update_node of REQ-HOLD-EXPIRY (P-0006/op-1): Draft 3 decides the previously undecided window and states acceptance criteria.
- Deferred update_node of CON-NOTICE-FORMAT (P-0007/op-1): The accepted contract was changed to allow SMS and require desk opening hours in every message.
- Deferred update_node of REQ-HOLD-EXPIRY (P-0007/op-2): The requirement now states the decided pickup window and its acceptance criteria.
- Deferred update_node of REQ-HOLD-EXPIRY (P-0008/op-1): Draft 4 replaces the undecided window with an explicit 48-hour window and acceptance criteria.
- Deferred update_node of CON-NOTICE-FORMAT (P-0008/op-2): Draft 4 adds a preferred SMS channel and desk-hours content while retaining the email limits and accepted status.

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
