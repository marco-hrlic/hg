# toolshare specification (after E2, arm G2)

> Generated from the specification graph at revision 3. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-FAIR-ACCESS** — Fair access to shared tools _(source: toolshare:spec/requirements.md L7-7; via P-0001/CHG-2)_
  Members can borrow shared tools fairly and predictably.
  - **CAP-HOLDS** — Holds _(source: toolshare:spec/requirements.md L9-11; via P-0001/CHG-2)_
    Members can queue for tools that are not currently on the shelf.
    - **REQ-HOLD-EXPIRY** — Expire uncollected holds _(source: toolshare:spec/requirements.md L18-20; via P-0001/CHG-2)_
      When a held tool becomes available, the member has a pickup window. If it is not collected in time, the hold lapses and the tool passes to the next member in the queue. The window length is undecided.
      Acceptance criteria: _none recorded yet_
      Depends on: REQ-READY-NOTICE (unconfirmed)
      Blocked by open question(s): Q-P-0002-C1
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

## Contracts

- **CON-NOTICE-FORMAT** — Notification format [status: accepted] _(source: toolshare:spec/requirements.md L32-33; via P-0001/CHG-2)_
  Notification messages are plain-text emails with a subject line and at most 500 characters of body text.
  Applies to: CMP-NOTIFY

## Components (responsibilities only)

- **CMP-HOLDS** — Holds service _(source: toolshare:spec/architecture.md L5-6; via P-0001/CHG-2)_
  Owns the hold queue and hold expiry.
  Implements: REQ-HOLD-EXPIRY, REQ-PLACE-HOLD
  Depends on: CMP-INVENTORY
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

## Proposed — not approved

- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-HOLDS (LE-GOAL-FAIR-ACCESS--refines--CAP-HOLDS)
- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-NOTIFY (LE-GOAL-FAIR-ACCESS--refines--CAP-NOTIFY)
- Unconfirmed relationship: REQ-HOLD-EXPIRY depends on REQ-READY-NOTICE (LE-REQ-HOLD-EXPIRY--depends_on--REQ-READY-NOTICE)
- Deferred update_node of REQ-HOLD-EXPIRY (P-0002/op-1): Draft 2 decides the window length and adds acceptance criteria.

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
