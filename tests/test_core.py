"""Basic tests for AgentLife core functionality."""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

from agentlife.collector import Collector
from agentlife.models import Session, Span, SpanStatus, SpanType, estimate_cost
from agentlife.store import Store


@pytest.fixture
def tmp_store():
    with tempfile.TemporaryDirectory() as td:
        yield Store(db_path=os.path.join(td, "test.db"))


# ── Models ──


def test_session_defaults():
    s = Session()
    assert s.status == SpanStatus.RUNNING
    assert s.total_tokens == 0
    assert len(s.id) == 12


def test_span_defaults():
    s = Span(session_id="abc")
    assert s.span_type == SpanType.FUNCTION
    assert s.status == SpanStatus.RUNNING


def test_estimate_cost_known_model():
    cost = estimate_cost("gpt-4o-mini", 1000, 500)
    assert cost > 0


def test_estimate_cost_unknown_model():
    cost = estimate_cost("unknown-model", 1000, 500)
    assert cost == 0.0


# ── Store ──


@pytest.mark.asyncio
async def test_store_save_and_list(tmp_store: Store):
    session = Session(name="test-session")
    await tmp_store.save_session(session)

    sessions = await tmp_store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].name == "test-session"


@pytest.mark.asyncio
async def test_store_save_span(tmp_store: Store):
    session = Session(name="test")
    await tmp_store.save_session(session)

    span = Span(session_id=session.id, name="llm-call", span_type=SpanType.LLM, model="gpt-4o")
    await tmp_store.save_span(span)

    spans = await tmp_store.get_spans(session.id)
    assert len(spans) == 1
    assert spans[0].name == "llm-call"
    assert spans[0].model == "gpt-4o"


@pytest.mark.asyncio
async def test_store_delete_session(tmp_store: Store):
    session = Session(name="to-delete")
    await tmp_store.save_session(session)
    await tmp_store.delete_session(session.id)

    sessions = await tmp_store.list_sessions()
    assert len(sessions) == 0


@pytest.mark.asyncio
async def test_store_clear_all(tmp_store: Store):
    for i in range(3):
        await tmp_store.save_session(Session(name=f"s{i}"))
    await tmp_store.clear_all()

    sessions = await tmp_store.list_sessions()
    assert len(sessions) == 0


# ── Public API ──


def test_init_and_session():
    import agentlife

    agentlife.init(patch_openai=False)

    with agentlife.session("test-api"):
        pass


def test_trace_decorator():
    import agentlife

    agentlife.init(patch_openai=False)

    @agentlife.trace
    def my_func(x: int) -> int:
        return x * 2

    with agentlife.session("test-decorator"):
        result = my_func(5)

    assert result == 10
