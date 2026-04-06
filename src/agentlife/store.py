"""SQLite storage layer for sessions and spans."""

from __future__ import annotations

import json
import os
from pathlib import Path

import aiosqlite

from agentlife.models import Session, Span

DEFAULT_DB_PATH = os.path.join(Path.home(), ".agentlife", "traces.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    total_tokens INTEGER DEFAULT 0,
    total_cost REAL DEFAULT 0.0,
    total_duration_ms REAL DEFAULT 0.0,
    span_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    parent_span_id TEXT,
    span_type TEXT NOT NULL DEFAULT 'function',
    name TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_ms REAL,
    input_data TEXT,
    output_data TEXT,
    error TEXT,
    model TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cost REAL,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id);
CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_span_id);
"""


class Store:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.executescript(_SCHEMA)
        return db

    # ── Sessions ──

    async def save_session(self, session: Session) -> None:
        db = await self._get_db()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, name, status, started_at, ended_at,
                    total_tokens, total_cost, total_duration_ms,
                    span_count, error_count, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.id, session.name, session.status.value,
                    session.started_at, session.ended_at,
                    session.total_tokens, session.total_cost,
                    session.total_duration_ms, session.span_count,
                    session.error_count, json.dumps(session.metadata),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[Session]:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            rows = await cursor.fetchall()
            return [self._row_to_session(r) for r in rows]
        finally:
            await db.close()

    async def get_session(self, session_id: str) -> Session | None:
        db = await self._get_db()
        try:
            cursor = await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = await cursor.fetchone()
            return self._row_to_session(row) if row else None
        finally:
            await db.close()

    async def delete_session(self, session_id: str) -> None:
        db = await self._get_db()
        try:
            await db.execute("DELETE FROM spans WHERE session_id = ?", (session_id,))
            await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await db.commit()
        finally:
            await db.close()

    async def clear_all(self) -> None:
        db = await self._get_db()
        try:
            await db.execute("DELETE FROM spans")
            await db.execute("DELETE FROM sessions")
            await db.commit()
        finally:
            await db.close()

    # ── Spans ──

    async def save_span(self, span: Span) -> None:
        db = await self._get_db()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO spans
                   (id, session_id, parent_span_id, span_type, name, status,
                    started_at, ended_at, duration_ms,
                    input_data, output_data, error,
                    model, prompt_tokens, completion_tokens, total_tokens, cost,
                    metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    span.id, span.session_id, span.parent_span_id,
                    span.span_type.value, span.name, span.status.value,
                    span.started_at, span.ended_at, span.duration_ms,
                    _json_safe(span.input_data), _json_safe(span.output_data), span.error,
                    span.model, span.prompt_tokens, span.completion_tokens,
                    span.total_tokens, span.cost,
                    json.dumps(span.metadata),
                ),
            )
            await db.commit()
        finally:
            await db.close()

    async def get_spans(self, session_id: str) -> list[Span]:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                "SELECT * FROM spans WHERE session_id = ? ORDER BY started_at ASC",
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_span(r) for r in rows]
        finally:
            await db.close()

    # ── Helpers ──

    @staticmethod
    def _row_to_session(row: aiosqlite.Row) -> Session:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return Session(**d)

    @staticmethod
    def _row_to_span(row: aiosqlite.Row) -> Span:
        d = dict(row)
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        d["input_data"] = _json_parse(d.get("input_data"))
        d["output_data"] = _json_parse(d.get("output_data"))
        return Span(**d)


def _json_safe(value: object) -> str | None:
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _json_parse(value: str | None) -> object:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value
