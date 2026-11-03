# toolshare specification (after E1, arm G0)

> Generated from the specification graph at revision 2. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-FAIR-ACCESS** — Fair, predictable tool borrowing _(source: toolshare:spec/requirements.md L7-7; via P-0001/CHG-2)_
  Members can borrow shared tools fairly and predictably.
  - **CAP-HOLDS** — Holds _(source: toolshare:spec/requirements.md L11-11; via P-0001/CHG-2)_
    Members can queue for tools that are not currently on the shelf.
    - **REQ-HOLD-EXPIRY** — Hold expiry and pickup window _(source: toolshare:spec/requirements.md L18-18; via P-0001/CHG-2)_
      When a held tool becomes available, the member has a pickup window to collect it; otherwise the hold lapses and the tool passes to the next member.
      Acceptance criteria: _none recorded yet_
      Blocked by open question(s): Q-PICKUP-WINDOW
    - **REQ-PLACE-HOLD** — Place a hold _(source: toolshare:spec/requirements.md L13-13; toolshare:spec/requirements.md L16-16; via P-0001/CHG-2)_
      A member can place a hold on a catalogued tool that is on loan.
      Acceptance criteria:
      - A member without overdue items can place a hold on any catalogued tool.
      - Holds on the same tool are served first come, first served.
  - **CAP-NOTIFY** — Member notifications _(source: toolshare:spec/requirements.md L26-26; via P-0001/CHG-2)_
    Members are told when something they are waiting for changes.
    - **REQ-READY-NOTICE** — Ready-for-pickup notice _(source: toolshare:spec/requirements.md L28-28; via P-0001/CHG-2)_
      When a held tool is ready for pickup, the member is notified by email.
      Acceptance criteria:
      - The email names the tool and the pickup deadline.
      Constrained by: CON-NOTICE-FORMAT

## Contracts

- **CON-NOTICE-FORMAT** — Notification message format [status: accepted] _(source: toolshare:spec/requirements.md L32-32; via P-0001/CHG-2)_
  Notification messages are plain-text emails with a subject line and at most 500 characters of body text.
  Applies to: CMP-NOTIFY, REQ-READY-NOTICE

## Components (responsibilities only)

- **CMP-HOLDS** — Holds service _(source: toolshare:spec/architecture.md L5-5; via P-0001/CHG-2)_
  Owns the hold queue and hold expiry.
  Implements: REQ-HOLD-EXPIRY, REQ-PLACE-HOLD
  Depends on: CMP-INVENTORY
- **CMP-INVENTORY** — Inventory module _(source: toolshare:spec/architecture.md L7-7; via P-0001/CHG-2)_
  Tracks whether a tool is on the shelf.
- **CMP-NOTIFY** — Notification service _(source: toolshare:spec/architecture.md L9-9; via P-0001/CHG-2)_
  Sends member emails.
  Implements: REQ-READY-NOTICE

## Decisions

_None recorded._

## Open questions

- **Q-PICKUP-WINDOW** (blocking) — How long is the pickup window? (blocks REQ-HOLD-EXPIRY)
  The desk is staffed three days a week.

## Proposed — not approved

- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-HOLDS (LE-GOAL-FAIR-ACCESS--refines--CAP-HOLDS)
- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-NOTIFY (LE-GOAL-FAIR-ACCESS--refines--CAP-NOTIFY)
- Unconfirmed relationship: REQ-READY-NOTICE constrained by CON-NOTICE-FORMAT (LE-REQ-READY-NOTICE--constrained_by--CON-NOTICE-FORMAT)
- Unconfirmed relationship: TEST-HOLD-QUEUE verifies REQ-PLACE-HOLD (LE-TEST-HOLD-QUEUE--verifies--REQ-PLACE-HOLD)

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
