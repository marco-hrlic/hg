# toolshare specification (after E1, arm G0)

> Generated from the specification graph at revision 2. This file is read-only: propose changes through the graph review process instead of editing it. Implementation code is not included; it lives in the project's Git repository.

## Goals, capabilities and requirements

- **GOAL-FAIR-ACCESS** — Fair access to shared tools _(source: toolshare:spec/requirements.md L5-7; via P-0001/CHG-2)_
  Members can borrow shared tools fairly and predictably.
  - **CAP-HOLDS** — Holds _(source: toolshare:spec/requirements.md L9-11; via P-0001/CHG-2)_
    Members can queue for tools that are not currently on the shelf.
    - **REQ-HOLD-EXPIRY** — Hold expiry _(source: toolshare:spec/requirements.md L18-20; via P-0001/CHG-2)_
      When a held tool becomes available, the member has a pickup window. If they do not collect it in time, the hold lapses and the tool passes to the next member in the queue. The window length is undecided.
      Acceptance criteria: _none recorded yet_
      Blocked by open question(s): Q-PICKUP-WINDOW
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

_None recorded._

## Open questions

- **Q-PICKUP-WINDOW** (blocking) — Pickup window length (blocks REQ-HOLD-EXPIRY)
  How long is the pickup window? The desk is staffed three days a week.

## Proposed — not approved

- Unconfirmed relationship: CAP-HOLDS refines REQ-HOLD-EXPIRY (LE-CAP-HOLDS--refines--REQ-HOLD-EXPIRY)
- Unconfirmed relationship: CAP-HOLDS refines REQ-PLACE-HOLD (LE-CAP-HOLDS--refines--REQ-PLACE-HOLD)
- Unconfirmed relationship: CAP-NOTIFY refines REQ-READY-NOTICE (LE-CAP-NOTIFY--refines--REQ-READY-NOTICE)
- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-HOLDS (LE-GOAL-FAIR-ACCESS--refines--CAP-HOLDS)
- Unconfirmed relationship: GOAL-FAIR-ACCESS refines CAP-NOTIFY (LE-GOAL-FAIR-ACCESS--refines--CAP-NOTIFY)

---
_Completeness note: this projection lists registered relationships only. An absent relationship is not evidence that two items are independent._
