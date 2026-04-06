"""Central trace collector — manages sessions, spans, and async persistence."""

from __future__ import annotations

import asyncio
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from agentlife.models import Session, Span, SpanStatus, SpanType, estimate_cost
from agentlife.store import Store

_current_session: ContextVar[Session | None] = ContextVar("_current_session", default=None)
_current_span: ContextVar[Span | None] = ContextVar("_current_span", default=None)


class Collector:
    """Singleton that buffers trace data and flushes to SQLite."""

    _instance: Collector | None = None

    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._queue: list[tuple[str, Any]] = []
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> Collector:
        if cls._instance is None:
            cls._instance = Collector()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # ── Background event loop ──

    def _ensure_loop(self) -> None:
        if self._loop is not None and self._loop.is_running():
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _submit(self, coro: Any) -> None:
        self._ensure_loop()
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ── Session lifecycle ──

    def start_session(self, name: str = "unnamed", metadata: dict | None = None) -> Session:
        session = Session(name=name, metadata=metadata or {})
        _current_session.set(session)
        self._submit(self.store.save_session(session))
        return session

    def end_session(self, session: Session, error: str | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        session.ended_at = now
        session.status = SpanStatus.ERROR if error else SpanStatus.OK
        if session.started_at:
            start = datetime.fromisoformat(session.started_at)
            session.total_duration_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        self._submit(self.store.save_session(session))
        _current_session.set(None)

    @staticmethod
    def get_current_session() -> Session | None:
        return _current_session.get()

    # ── Span lifecycle ──

    def start_span(
        self,
        name: str,
        span_type: SpanType = SpanType.FUNCTION,
        input_data: Any = None,
        model: str | None = None,
        metadata: dict | None = None,
    ) -> Span:
        session = _current_session.get()
        if session is None:
            session = self.start_session(name="auto")

        parent = _current_span.get()
        span = Span(
            session_id=session.id,
            parent_span_id=parent.id if parent else None,
            span_type=span_type,
            name=name,
            input_data=input_data,
            model=model,
            metadata=metadata or {},
        )
        _current_span.set(span)
        return span

    def end_span(
        self,
        span: Span,
        output_data: Any = None,
        error: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        span.ended_at = now.isoformat()

        if span.started_at:
            start = datetime.fromisoformat(span.started_at)
            span.duration_ms = (now - start).total_seconds() * 1000

        span.output_data = output_data
        span.status = SpanStatus.ERROR if error else SpanStatus.OK
        span.error = error

        if prompt_tokens is not None:
            span.prompt_tokens = prompt_tokens
        if completion_tokens is not None:
            span.completion_tokens = completion_tokens
        if span.prompt_tokens and span.completion_tokens:
            span.total_tokens = span.prompt_tokens + span.completion_tokens
            span.cost = estimate_cost(span.model, span.prompt_tokens, span.completion_tokens)

        # Update session aggregates
        session = _current_session.get()
        if session:
            session.span_count += 1
            session.total_tokens += span.total_tokens or 0
            session.total_cost += span.cost or 0.0
            if error:
                session.error_count += 1

        self._submit(self.store.save_span(span))

        # Restore parent span
        parent_id = span.parent_span_id
        if parent_id is None:
            _current_span.set(None)
        # Keep parent reference (it was set before this span)

    @staticmethod
    def get_current_span() -> Span | None:
        return _current_span.get()
