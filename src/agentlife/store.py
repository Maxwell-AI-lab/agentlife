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
    group_id TEXT,
    sample_index INTEGER,
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
CREATE INDEX IF NOT EXISTS idx_sessions_group ON sessions(group_id);
"""

_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN group_id TEXT",
    "ALTER TABLE sessions ADD COLUMN sample_index INTEGER",
]


class Store:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    async def _get_db(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.executescript(_SCHEMA)
        for sql in _MIGRATIONS:
            try:
                await db.execute(sql)
                await db.commit()
            except Exception:
                pass
        return db

    # ── Sessions ──

    async def save_session(self, session: Session) -> None:
        db = await self._get_db()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO sessions
                   (id, name, group_id, sample_index, status, started_at, ended_at,
                    total_tokens, total_cost, total_duration_ms,
                    span_count, error_count, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.id, session.name,
                    session.group_id, session.sample_index,
                    session.status.value,
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

    # ── Groups ──

    async def list_groups(self, limit: int = 50) -> list[dict]:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                """SELECT group_id,
                          COUNT(*) as n_samples,
                          SUM(total_tokens) as sum_tokens,
                          AVG(total_tokens) as avg_tokens,
                          SUM(total_cost) as sum_cost,
                          AVG(total_cost) as avg_cost,
                          AVG(total_duration_ms) as avg_duration_ms,
                          MIN(total_duration_ms) as min_duration_ms,
                          MAX(total_duration_ms) as max_duration_ms,
                          SUM(error_count) as total_errors,
                          MIN(started_at) as started_at
                   FROM sessions
                   WHERE group_id IS NOT NULL
                   GROUP BY group_id
                   ORDER BY started_at DESC
                   LIMIT ?""",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        finally:
            await db.close()

    async def get_group_sessions(self, group_id: str) -> list[Session]:
        db = await self._get_db()
        try:
            cursor = await db.execute(
                """SELECT * FROM sessions
                   WHERE group_id = ?
                   ORDER BY sample_index ASC, started_at ASC""",
                (group_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_session(r) for r in rows]
        finally:
            await db.close()

    async def get_group_stats(self, group_id: str) -> dict:
        """Compute detailed comparison stats for a group of rollout samples."""
        sessions = await self.get_group_sessions(group_id)
        if not sessions:
            return {"error": "no sessions in group"}

        samples = []
        for s in sessions:
            spans = await self.get_spans(s.id)
            llm_spans = [sp for sp in spans if sp.span_type.value == "llm"]
            tool_spans = [sp for sp in spans if sp.span_type.value == "tool"]
            func_spans = [sp for sp in spans if sp.span_type.value == "function"]

            final_output = None
            for sp in reversed(spans):
                if sp.output_data and sp.span_type.value in ("llm", "function"):
                    final_output = sp.output_data
                    break

            samples.append({
                "session_id": s.id,
                "sample_index": s.sample_index,
                "name": s.name,
                "status": s.status.value,
                "total_tokens": s.total_tokens,
                "total_cost": s.total_cost,
                "total_duration_ms": s.total_duration_ms,
                "span_count": s.span_count,
                "error_count": s.error_count,
                "llm_calls": len(llm_spans),
                "tool_calls": len(tool_spans),
                "func_calls": len(func_spans),
                "final_output": final_output,
            })

        tokens_list = [s["total_tokens"] for s in samples]
        cost_list = [s["total_cost"] for s in samples]
        dur_list = [s["total_duration_ms"] for s in samples]
        steps_list = [s["llm_calls"] for s in samples]

        def _stats(vals: list) -> dict:
            if not vals:
                return {}
            import statistics
            n = len(vals)
            return {
                "min": min(vals),
                "max": max(vals),
                "mean": round(statistics.mean(vals), 2),
                "std": round(statistics.stdev(vals), 2) if n > 1 else 0,
            }

        diagnostics = []
        error_samples = [s for s in samples if s["status"] == "error"]
        if error_samples:
            diagnostics.append({
                "type": "error", "severity": "high",
                "message": f"{len(error_samples)}/{len(samples)} 次采样失败（session 级别错误）",
                "samples": [s["sample_index"] for s in error_samples],
            })

        span_error_samples = [s for s in samples if s["error_count"] > 0 and s["status"] != "error"]
        if span_error_samples:
            diagnostics.append({
                "type": "span_errors", "severity": "high",
                "message": f"{len(span_error_samples)}/{len(samples)} 次采样包含 span 级别错误（异常被捕获但已记录）",
                "samples": [s["sample_index"] for s in span_error_samples],
            })

        zero_token_samples = [s for s in samples if s["total_tokens"] == 0]
        if zero_token_samples and len(zero_token_samples) < len(samples):
            diagnostics.append({
                "type": "no_output", "severity": "medium",
                "message": f"{len(zero_token_samples)}/{len(samples)} 次采样 token 为 0（可能 API 调用失败）",
                "samples": [s["sample_index"] for s in zero_token_samples],
            })

        if tokens_list and max(tokens_list) > 0:
            mean_tok = sum(tokens_list) / len(tokens_list)
            for s in samples:
                if mean_tok > 0 and s["total_tokens"] > mean_tok * 2:
                    diagnostics.append({
                        "type": "token_outlier", "severity": "medium",
                        "message": f"Sample {s['sample_index']} token 用量异常偏高 ({s['total_tokens']} vs 均值 {mean_tok:.0f})",
                        "samples": [s["sample_index"]],
                    })

        if dur_list and max(dur_list) > 0:
            mean_dur = sum(dur_list) / len(dur_list)
            for s in samples:
                if mean_dur > 0 and s["total_duration_ms"] > mean_dur * 2:
                    diagnostics.append({
                        "type": "slow_outlier", "severity": "medium",
                        "message": f"Sample {s['sample_index']} 耗时异常偏高 ({s['total_duration_ms']:.0f}ms vs 均值 {mean_dur:.0f}ms)",
                        "samples": [s["sample_index"]],
                    })

        ok_samples = [s for s in samples if s["status"] == "ok"]
        if len(ok_samples) >= 2:
            outputs = [str(s.get("final_output", "")) for s in ok_samples]
            unique_outputs = len(set(outputs))
            if unique_outputs > 1:
                diagnostics.append({
                    "type": "inconsistent", "severity": "info",
                    "message": f"{unique_outputs} 种不同的最终输出（共 {len(ok_samples)} 次成功采样）",
                    "samples": list(range(len(ok_samples))),
                })

        return {
            "group_id": group_id,
            "n_samples": len(samples),
            "samples": samples,
            "aggregate": {
                "tokens": _stats(tokens_list),
                "cost": _stats(cost_list),
                "duration_ms": _stats(dur_list),
                "llm_calls": _stats(steps_list),
            },
            "diagnostics": diagnostics,
        }

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
