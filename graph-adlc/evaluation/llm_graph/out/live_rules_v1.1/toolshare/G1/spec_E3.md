# toolshare specification (after E3, arm G1)

> Generated from the specification graph at revision 4. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-FAIR-ACCESS** — Fair access to shared tools _(source: toolshare:spec/requirements.md L5-7; via P-0001/CHG-2)_
  Members can borrow shared tools fairly and predictably.
  - **CAP-HOLDS** — Holds _(source: toolshare:spec/requirements.md L9-11; via P-0001/CHG-2)_
    Members can queue for tools that are not currently on the shelf.
    - **REQ-HOLD-EXPIRY** — Hold expiry _(source: toolshare:spec/requirements.md L18-20; via P-0001/CHG-2)_
      When a held tool becomes available, the member has a pickup window. If the member does not collect it in time, the hold lapses and the tool passes to the next member in the queue. The window length is undecided.
      Acceptance criteria: _none recorded yet_
      Depends on: REQ-READY-NOTICE (unconfirmed)
      Blocked by open question(s): Q-P-0002-C1, Q-P-0003-C1
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
      Constrained by: CON-NOTICE-FORMAT

## Contracts

- **CON-NOTICE-FORMAT** — Notification format [status: accepted] _(source: toolshare:spec/requirements.md L32-33; via P-0001/CHG-2)_
  Notification messages are plain-text emails with a subject line and at most 500 characters of body text.
  Applies to: CMP-NOTIFY, REQ-READY-NOTICE

## Components (responsibilities only)

- **CMP-HOLDS** — Holds service _(source: toolshare:spec/architecture.md L5-6; via P-0001/CHG-2)_
  Owns the hold queue and hold expiry.
  Implements: REQ-HOLD-EXPIRY, REQ-PLACE-HOLD
  Depends on: CMP-INVENTORY, CMP-NOTIFY
- **CMP-INVENTORY** — Inventory module _(source: toolshare:spec/architecture.md L7-8; via P-0001/CHG-2)_
  Tracks whether a tool is on the shelf.
- **CMP-NOTIFY** — Notification service _(source: toolshare:spec/architecture.md L9-10; via P-0001/CHG-2)_
  Sends member emails.
  Implements: REQ-READY-NOTICE

## Decisions

- **Q-PICKUP-WINDOW** — Pickup window length: How long is the pickup window? The desk is staffed three days a week. The board decided on 2026-09-10: 48 hours from the ready notice. _(source: toolshare:spec/requirements.md L20-22; toolshare:spec/requirements.md L24-25; via P-0002/CHG-3)_

## Open questions

- **Q-P-0002-C1** (blocking) — Resolve: The expiry function removes the first queued member without checking the ready-notice … (blocks CMP-HOLDS, REQ-HOLD-EXPIRY)
  The expiry function removes the first queued member without checking the ready-notice time or elapsed window. This could permit premature expiry unless a caller enforces the 48-hour condition.
- **Q-P-0003-C1** (blocking) — Resolve: Expiry removes the first queued member without checking the ready-notice time or elapsed … (blocks CMP-HOLDS, REQ-HOLD-EXPIRY)
  Expiry removes the first queued member without checking the ready-notice time or elapsed window. It could therefore lapse a hold prematurely unless a caller enforces the 48-hour condition.

## Proposed — not approved

- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-HOLDS (LE-GOAL-FAIR-ACCESS--refines--CAP-HOLDS)
- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-NOTIFY (LE-GOAL-FAIR-ACCESS--refines--CAP-NOTIFY)
- Unconfirmed relationship: REQ-HOLD-EXPIRY depends on REQ-READY-NOTICE (LE-REQ-HOLD-EXPIRY--depends_on--REQ-READY-NOTICE)
- Unconfirmed relationship: REQ-READY-NOTICE constrained by CON-NOTICE-FORMAT (LE-REQ-READY-NOTICE--constrained_by--CON-NOTICE-FORMAT)
- Unconfirmed relationship: TEST-HOLD-QUEUE verifies REQ-HOLD-EXPIRY (LE-TEST-HOLD-QUEUE--verifies--REQ-HOLD-EXPIRY)
- Unconfirmed relationship: TEST-HOLD-QUEUE verifies REQ-PLACE-HOLD (LE-TEST-HOLD-QUEUE--verifies--REQ-PLACE-HOLD)
- Deferred update_node of REQ-HOLD-EXPIRY (P-0002/op-1): Draft 2 resolves the window length and adds acceptance criteria.
- Deferred update_node of REQ-HOLD-EXPIRY (P-0003/op-1): The requirements document states the previously undecided window and its acceptance criteria.
- Deferred update_node of CMP-HOLDS (P-0003/op-2): The changed implementation now emails the member whose hold lapses.

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
