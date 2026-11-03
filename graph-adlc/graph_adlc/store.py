"""SQLite persistence: schema, transactions, snapshot import/export."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from .models import Edge, Graph, Node, canonical_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL, body TEXT NOT NULL,
    data_json TEXT NOT NULL, rev INTEGER NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    src TEXT NOT NULL REFERENCES nodes(id), dst TEXT NOT NULL REFERENCES nodes(id),
    relation TEXT NOT NULL, status TEXT NOT NULL, rev INTEGER NOT NULL,
    deleted INTEGER NOT NULL DEFAULT 0, evidence_ref TEXT
);
CREATE TABLE IF NOT EXISTS changes (
    id TEXT PRIMARY KEY, graph_rev INTEGER NOT NULL UNIQUE, base_graph_rev INTEGER NOT NULL,
    actor TEXT NOT NULL, rationale TEXT NOT NULL, operations_json TEXT NOT NULL,
    impact_json TEXT NOT NULL, diff_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS work_units (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, kind TEXT NOT NULL, targets_json TEXT NOT NULL,
    baseline_json TEXT NOT NULL, context_json TEXT NOT NULL, status TEXT NOT NULL,
    code_rev TEXT, history_json TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY, seq INTEGER NOT NULL UNIQUE, work_unit_id TEXT REFERENCES work_units(id),
    node_id TEXT NOT NULL, kind TEXT NOT NULL, result TEXT NOT NULL, spec_revs_json TEXT NOT NULL,
    code_rev TEXT, details_json TEXT NOT NULL, invalidated_by TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, graph_rev INTEGER NOT NULL, kind TEXT NOT NULL,
    ref_id TEXT, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    fixed = os.environ.get("GADLC_FIXED_TIME")
    if fixed:
        return fixed
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class Store:
    def __init__(self, path: str, clock: Callable[[], str] = utc_now):
        self.path = path
        self.clock = clock
        self.conn = sqlite3.connect(path, isolation_level=None)  # explicit transactions only
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._depth = 0

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Single write transaction; nested use joins the outer transaction."""
        if self._depth:
            self._depth += 1
            try:
                yield self.conn
            finally:
                self._depth -= 1
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield self.conn
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")
        finally:
            self._depth = 0

    # -- meta -----------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                          (key, str(value)))

    def initialized(self) -> bool:
        return self.get_meta("graph_revision") is not None

    def graph_revision(self) -> int:
        v = self.get_meta("graph_revision")
        if v is None:
            raise RuntimeError("database is not initialized; run `gadlc init`")
        return int(v)

    def next_seq(self, counter: str) -> int:
        n = int(self.get_meta(f"seq:{counter}", "0")) + 1
        self.set_meta(f"seq:{counter}", n)
        return n

    # -- graph ----------------------------------------------------------
    def load_graph(self) -> Graph:
        g = Graph(revision=self.graph_revision())
        for r in self.conn.execute("SELECT * FROM nodes ORDER BY id"):
            g.nodes[r["id"]] = Node(
                id=r["id"], kind=r["kind"], title=r["title"], body=r["body"], data=json.loads(r["data_json"]),
                rev=r["rev"], deleted=bool(r["deleted"]), created_at=r["created_at"], updated_at=r["updated_at"],
            )
        for r in self.conn.execute("SELECT * FROM edges ORDER BY id"):
            g.edges[r["id"]] = Edge(
                id=r["id"], src=r["src"], dst=r["dst"], relation=r["relation"], status=r["status"],
                rev=r["rev"], deleted=bool(r["deleted"]), evidence_ref=r["evidence_ref"],
            )
        return g

    def upsert_node(self, n: Node) -> None:
        now = self.clock()
        self.conn.execute(
            """INSERT INTO nodes(id, kind, title, body, data_json, rev, deleted, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title, body=excluded.body,
                 data_json=excluded.data_json, rev=excluded.rev, deleted=excluded.deleted,
                 updated_at=excluded.updated_at""",
            (n.id, n.kind, n.title, n.body, canonical_json(n.data), n.rev, int(n.deleted),
             n.created_at or now, n.updated_at or now),
        )

    def upsert_edge(self, e: Edge) -> None:
        self.conn.execute(
            """INSERT INTO edges(id, src, dst, relation, status, rev, deleted, evidence_ref)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status, rev=excluded.rev,
                 deleted=excluded.deleted, evidence_ref=excluded.evidence_ref""",
            (e.id, e.src, e.dst, e.relation, e.status, e.rev, int(e.deleted), e.evidence_ref),
        )

    # -- changes / events --------------------------------------------------
    def insert_change(self, row: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT INTO changes(id, graph_rev, base_graph_rev, actor, rationale, operations_json,
                                   impact_json, diff_json, created_at) VALUES(?,?,?,?,?,?,?,?,?)""",
            (row["id"], row["graph_rev"], row["base_graph_rev"], row["actor"], row["rationale"],
             canonical_json(row["operations"]), canonical_json(row["impact"]), canonical_json(row["diff"]),
             row["created_at"]),
        )

    def list_changes(self) -> list[dict[str, Any]]:
        return [self._change_row(r) for r in self.conn.execute("SELECT * FROM changes ORDER BY graph_rev")]

    def get_change(self, change_id: str) -> dict[str, Any] | None:
        r = self.conn.execute("SELECT * FROM changes WHERE id=?", (change_id,)).fetchone()
        return self._change_row(r) if r else None

    @staticmethod
    def _change_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"], "graph_rev": r["graph_rev"], "base_graph_rev": r["base_graph_rev"],
            "actor": r["actor"], "rationale": r["rationale"], "operations": json.loads(r["operations_json"]),
            "impact": json.loads(r["impact_json"]), "diff": json.loads(r["diff_json"]),
            "created_at": r["created_at"],
        }

    def log_event(self, kind: str, ref_id: str | None, payload: dict[str, Any]) -> None:
        rev = int(self.get_meta("graph_revision", "0"))
        self.conn.execute("INSERT INTO events(graph_rev, kind, ref_id, payload_json, created_at) VALUES(?,?,?,?,?)",
                          (rev, kind, ref_id, canonical_json(payload), self.clock()))

    def list_events(self) -> list[dict[str, Any]]:
        return [
            {"seq": r["seq"], "graph_rev": r["graph_rev"], "kind": r["kind"], "ref_id": r["ref_id"],
             "payload": json.loads(r["payload_json"]), "created_at": r["created_at"]}
            for r in self.conn.execute("SELECT * FROM events ORDER BY seq")
        ]

    # -- work units ---------------------------------------------------------
    def save_work_unit(self, w: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT INTO work_units(id, seq, kind, targets_json, baseline_json, context_json, status, code_rev,
                                      history_json, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET baseline_json=excluded.baseline_json,
                 context_json=excluded.context_json, status=excluded.status, code_rev=excluded.code_rev,
                 history_json=excluded.history_json, updated_at=excluded.updated_at""",
            (w["id"], w["seq"], w["kind"], canonical_json(w["target_ids"]), canonical_json(w["baseline"]),
             canonical_json(w["context_node_ids"]), w["status"], w.get("code_rev"), canonical_json(w["history"]),
             w["created_at"], w["updated_at"]),
        )

    def get_work_unit(self, work_id: str) -> dict[str, Any]:
        r = self.conn.execute("SELECT * FROM work_units WHERE id=?", (work_id,)).fetchone()
        if r is None:
            raise KeyError(work_id)
        return self._work_row(r)

    def list_work_units(self) -> list[dict[str, Any]]:
        return [self._work_row(r) for r in self.conn.execute("SELECT * FROM work_units ORDER BY seq")]

    @staticmethod
    def _work_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"], "seq": r["seq"], "kind": r["kind"], "target_ids": json.loads(r["targets_json"]),
            "baseline": json.loads(r["baseline_json"]), "context_node_ids": json.loads(r["context_json"]),
            "status": r["status"], "code_rev": r["code_rev"], "history": json.loads(r["history_json"]),
            "created_at": r["created_at"], "updated_at": r["updated_at"],
        }

    # -- evidence -------------------------------------------------------------
    def insert_evidence(self, ev: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT INTO evidence(id, seq, work_unit_id, node_id, kind, result, spec_revs_json, code_rev,
                                    details_json, invalidated_by, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (ev["id"], ev["seq"], ev.get("work_unit_id"), ev["node_id"], ev["kind"], ev["result"],
             canonical_json(ev["spec_revs"]), ev.get("code_rev"), canonical_json(ev.get("details", {})),
             ev.get("invalidated_by"), ev["created_at"]),
        )

    def invalidate_evidence(self, evidence_id: str, change_id: str) -> None:
        self.conn.execute("UPDATE evidence SET invalidated_by=? WHERE id=? AND invalidated_by IS NULL",
                          (change_id, evidence_id))

    def list_evidence(self, ids: list[str] | None = None) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM evidence ORDER BY seq").fetchall()
        out = [self._evidence_row(r) for r in rows]
        if ids is not None:
            wanted = set(ids)
            out = [e for e in out if e["id"] in wanted]
        return out

    @staticmethod
    def _evidence_row(r: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": r["id"], "seq": r["seq"], "work_unit_id": r["work_unit_id"], "node_id": r["node_id"],
            "kind": r["kind"], "result": r["result"], "spec_revs": json.loads(r["spec_revs_json"]),
            "code_rev": r["code_rev"], "details": json.loads(r["details_json"]),
            "invalidated_by": r["invalidated_by"], "created_at": r["created_at"],
        }
