"""Sidecar audit ledger for proposals, model calls and review decisions.

Stored in extra tables of the same SQLite file. The core engine schema is not
modified; graph changes still only happen through ``GraphADLC.commit``.
"""

from __future__ import annotations

import json
from typing import Any

from ..core import GraphADLC
from ..models import canonical_json

LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, record_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_proposals (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, base_graph_rev INTEGER NOT NULL, status TEXT NOT NULL,
    proposal_json TEXT NOT NULL, validation_json TEXT, change_id TEXT, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS llm_decisions (
    proposal_id TEXT NOT NULL, op_id TEXT NOT NULL, decision TEXT NOT NULL, reviewer TEXT NOT NULL,
    rationale TEXT NOT NULL, fingerprint TEXT NOT NULL, evidence_key TEXT NOT NULL, change_id TEXT,
    decided_at TEXT NOT NULL, PRIMARY KEY (proposal_id, op_id)
);
CREATE TABLE IF NOT EXISTS llm_provenance (
    entity_id TEXT NOT NULL, proposal_id TEXT NOT NULL, op_id TEXT NOT NULL, change_id TEXT NOT NULL,
    support TEXT NOT NULL, citations_json TEXT NOT NULL, PRIMARY KEY (entity_id, proposal_id, op_id)
);
"""

PROPOSAL_STATUSES = ("validated", "reviewed", "committed", "rejected", "stale", "invalid")


class Ledger:
    def __init__(self, app: GraphADLC):
        self.app = app
        self.store = app.store
        self.conn = app.store.conn
        self.conn.executescript(LEDGER_SCHEMA)

    # -- ids ------------------------------------------------------------------
    def next_proposal_id(self) -> str:
        with self.store.transaction():
            return f"P-{self.store.next_seq('llm_proposal'):04d}"

    # -- calls ----------------------------------------------------------------
    def record_call(self, record: dict[str, Any]) -> str:
        with self.store.transaction():
            seq = self.store.next_seq("llm_call")
            cid = f"CALL-{seq:04d}"
            self.conn.execute("INSERT INTO llm_calls(id, seq, record_json, created_at) VALUES (?,?,?,?)",
                              (cid, seq, canonical_json(record), self.store.clock()))
        return cid

    def calls(self) -> list[dict[str, Any]]:
        return [{"id": r["id"], **json.loads(r["record_json"])}
                for r in self.conn.execute("SELECT * FROM llm_calls ORDER BY seq")]

    # -- proposals -------------------------------------------------------------
    def save_proposal(self, proposal: dict[str, Any], validation: dict[str, Any] | None, status: str) -> None:
        now = self.store.clock()
        with self.store.transaction():
            row = self.conn.execute("SELECT seq FROM llm_proposals WHERE id=?", (proposal["proposal_id"],)).fetchone()
            if row is None:
                seq = self.store.next_seq("llm_proposal_row")
                self.conn.execute(
                    "INSERT INTO llm_proposals(id, seq, base_graph_rev, status, proposal_json, validation_json,"
                    " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (proposal["proposal_id"], seq, proposal["base_graph_revision"], status,
                     canonical_json(proposal), canonical_json(validation) if validation else None, now, now))
            else:
                self.conn.execute(
                    "UPDATE llm_proposals SET status=?, proposal_json=?, validation_json=?, base_graph_rev=?,"
                    " updated_at=? WHERE id=?",
                    (status, canonical_json(proposal), canonical_json(validation) if validation else None,
                     proposal["base_graph_revision"], now, proposal["proposal_id"]))
            self.store.log_event("llm_proposal_" + status, proposal["proposal_id"],
                                 {"base_graph_revision": proposal["base_graph_revision"],
                                  "operations": len(proposal["operations"])})

    def set_status(self, proposal_id: str, status: str, change_id: str | None = None) -> None:
        with self.store.transaction():
            self.conn.execute("UPDATE llm_proposals SET status=?, change_id=COALESCE(?, change_id), updated_at=?"
                              " WHERE id=?", (status, change_id, self.store.clock(), proposal_id))
            self.store.log_event("llm_proposal_" + status, proposal_id, {"change_id": change_id})

    def proposal(self, proposal_id: str) -> dict[str, Any]:
        r = self.conn.execute("SELECT * FROM llm_proposals WHERE id=?", (proposal_id,)).fetchone()
        if r is None:
            raise KeyError(proposal_id)
        return self._row(r)

    def proposals(self) -> list[dict[str, Any]]:
        return [self._row(r) for r in self.conn.execute("SELECT * FROM llm_proposals ORDER BY seq")]

    @staticmethod
    def _row(r: Any) -> dict[str, Any]:
        return {"id": r["id"], "status": r["status"], "base_graph_revision": r["base_graph_rev"],
                "change_id": r["change_id"], "proposal": json.loads(r["proposal_json"]),
                "validation": json.loads(r["validation_json"]) if r["validation_json"] else None}

    # -- decisions --------------------------------------------------------------
    def record_decisions(self, proposal_id: str, decisions: list[dict[str, Any]], change_id: str | None) -> None:
        now = self.store.clock()
        with self.store.transaction():
            for d in decisions:
                self.conn.execute(
                    "INSERT OR REPLACE INTO llm_decisions(proposal_id, op_id, decision, reviewer, rationale,"
                    " fingerprint, evidence_key, change_id, decided_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (proposal_id, d["op_id"], d["decision"], d["reviewer"], d.get("rationale", ""),
                     d["fingerprint"], d["evidence_key"], change_id if d["decision"] == "accept" else None, now))
            self.store.log_event("llm_review", proposal_id,
                                 {"decisions": {d["op_id"]: d["decision"] for d in decisions},
                                  "change_id": change_id})

    def decisions(self, proposal_id: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM llm_decisions", ()
        if proposal_id:
            q, args = q + " WHERE proposal_id=?", (proposal_id,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY decided_at, proposal_id, op_id", args)]

    def rejected(self, fingerprint: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM llm_decisions WHERE decision='reject' AND fingerprint=?", (fingerprint,))]

    def deferred(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(
            "SELECT * FROM llm_decisions WHERE decision='defer' ORDER BY proposal_id, op_id")]

    # -- provenance ---------------------------------------------------------------
    def record_provenance(self, rows: list[dict[str, Any]]) -> None:
        with self.store.transaction():
            for r in rows:
                self.conn.execute(
                    "INSERT OR REPLACE INTO llm_provenance(entity_id, proposal_id, op_id, change_id, support,"
                    " citations_json) VALUES (?,?,?,?,?,?)",
                    (r["entity_id"], r["proposal_id"], r["op_id"], r["change_id"], r["support"],
                     canonical_json(r["citations"])))

    def provenance(self, entity_id: str | None = None) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM llm_provenance", ()
        if entity_id:
            q, args = q + " WHERE entity_id=?", (entity_id,)
        return [{**{k: r[k] for k in ("entity_id", "proposal_id", "op_id", "change_id", "support")},
                 "citations": json.loads(r["citations_json"])}
                for r in self.conn.execute(q + " ORDER BY entity_id, proposal_id, op_id", args)]
