"""Tests for AgentLife core functionality."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest

from agentlife.collector import Collector
from agentlife.models import Session, Span, SpanStatus, SpanType, estimate_cost
from agentlife.store import Store, _schema_initialized


@pytest.fixture
def tmp_store():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "test.db")
        _schema_initialized.discard(path)
        yield Store(db_path=path)


@pytest.fixture(autouse=True)
def reset_collector():
    Collector.reset()
    yield
    Collector.reset()


# ── Models ──


def test_session_defaults():
    s = Session()
    assert s.status == SpanStatus.RUNNING
    assert s.total_tokens == 0
    assert len(s.id) == 12


def test_session_group_fields():
    s = Session(group_id="g-123", group_name="my-group", sample_index=2)
    assert s.group_id == "g-123"
    assert s.group_name == "my-group"
    assert s.sample_index == 2


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


def test_estimate_cost_glm_flash_free():
    cost = estimate_cost("glm-4-flash", 10000, 5000)
    assert cost == 0.0


def test_estimate_cost_partial_match():
    cost = estimate_cost("gpt-4o-mini-2024-07-18", 1000, 500)
    assert cost > 0


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


# ── Store: Groups ──


@pytest.mark.asyncio
async def test_store_group_roundtrip(tmp_store: Store):
    for i in range(3):
        s = Session(name=f"s-{i}", group_id="g-test", group_name="test-group", sample_index=i)
        await tmp_store.save_session(s)

    groups = await tmp_store.list_groups()
    assert len(groups) == 1
    assert groups[0]["group_id"] == "g-test"
    assert groups[0]["group_name"] == "test-group"
    assert groups[0]["n_samples"] == 3


@pytest.mark.asyncio
async def test_store_group_sessions(tmp_store: Store):
    for i in range(4):
        await tmp_store.save_session(
            Session(name=f"sample-{i}", group_id="g-abc", sample_index=i)
        )

    sessions = await tmp_store.get_group_sessions("g-abc")
    assert len(sessions) == 4
    assert [s.sample_index for s in sessions] == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_store_group_stats_basic(tmp_store: Store):
    for i in range(2):
        s = Session(
            name=f"s-{i}", group_id="g-stats", group_name="stats-test",
            sample_index=i, status=SpanStatus.OK,
            total_tokens=100 * (i + 1), total_cost=0.001 * (i + 1),
            total_duration_ms=1000.0 * (i + 1), span_count=3,
        )
        await tmp_store.save_session(s)

    stats = await tmp_store.get_group_stats("g-stats")
    assert stats["n_samples"] == 2
    assert stats["group_name"] == "stats-test"
    assert stats["aggregate"]["tokens"]["min"] == 100
    assert stats["aggregate"]["tokens"]["max"] == 200
    assert len(stats["samples"]) == 2


@pytest.mark.asyncio
async def test_store_group_stats_diagnostics_error(tmp_store: Store):
    s1 = Session(name="ok", group_id="g-d", sample_index=0, status=SpanStatus.OK)
    s2 = Session(name="err", group_id="g-d", sample_index=1, status=SpanStatus.ERROR, error_count=1)
    await tmp_store.save_session(s1)
    await tmp_store.save_session(s2)

    stats = await tmp_store.get_group_stats("g-d")
    diag_types = [d["type"] for d in stats["diagnostics"]]
    assert "error" in diag_types


@pytest.mark.asyncio
async def test_store_group_stats_diagnostics_span_errors(tmp_store: Store):
    s = Session(name="caught", group_id="g-se", sample_index=0, status=SpanStatus.OK, error_count=2)
    await tmp_store.save_session(s)

    stats = await tmp_store.get_group_stats("g-se")
    diag_types = [d["type"] for d in stats["diagnostics"]]
    assert "span_errors" in diag_types


# ── Store: Export ──


@pytest.mark.asyncio
async def test_store_export_session(tmp_store: Store):
    session = Session(name="export-test")
    await tmp_store.save_session(session)
    span = Span(session_id=session.id, name="my-span", span_type=SpanType.LLM)
    await tmp_store.save_span(span)

    data = await tmp_store.export_session(session.id)
    assert data is not None
    assert data["session"]["name"] == "export-test"
    assert len(data["spans"]) == 1
    assert data["spans"][0]["name"] == "my-span"


@pytest.mark.asyncio
async def test_store_export_missing(tmp_store: Store):
    data = await tmp_store.export_session("nonexistent")
    assert data is None


# ── Collector: Parent span restoration ──


def test_collector_nested_span_restore():
    """Verify 3+ level nesting restores parent correctly."""
    with tempfile.TemporaryDirectory() as td:
        store = Store(db_path=os.path.join(td, "test.db"))
        collector = Collector(store=store)
        Collector._instance = collector

        import agentlife
        agentlife._initialized = False
        agentlife.init(patch_openai=False)

        with agentlife.session("nested-test"):
            assert Collector.get_current_span() is None

            span_a = collector.start_span("level-1")
            assert Collector.get_current_span() is span_a

            span_b = collector.start_span("level-2")
            assert Collector.get_current_span() is span_b
            assert span_b.parent_span_id == span_a.id

            span_c = collector.start_span("level-3")
            assert Collector.get_current_span() is span_c
            assert span_c.parent_span_id == span_b.id

            collector.end_span(span_c, output_data="c-done")
            assert Collector.get_current_span() is span_b

            collector.end_span(span_b, output_data="b-done")
            assert Collector.get_current_span() is span_a

            collector.end_span(span_a, output_data="a-done")
            assert Collector.get_current_span() is None


# ── Public API ──


def test_init_and_session():
    import agentlife
    agentlife._initialized = False
    agentlife.init(patch_openai=False)

    with agentlife.session("test-api"):
        pass


def test_trace_decorator():
    import agentlife
    agentlife._initialized = False
    agentlife.init(patch_openai=False)

    @agentlife.trace
    def my_func(x: int) -> int:
        return x * 2

    with agentlife.session("test-decorator"):
        result = my_func(5)

    assert result == 10


def test_trace_nested_decorators():
    """@trace decorators at multiple levels build correct parent chain."""
    import agentlife
    agentlife._initialized = False
    agentlife.init(patch_openai=False)

    results = []

    @agentlife.trace
    def outer():
        results.append("outer-start")
        inner()
        results.append("outer-end")

    @agentlife.trace
    def inner():
        results.append("inner")
        deep()

    @agentlife.trace
    def deep():
        results.append("deep")

    with agentlife.session("nested-trace"):
        outer()

    assert results == ["outer-start", "inner", "deep", "outer-end"]


def test_group_context_manager():
    import agentlife
    agentlife._initialized = False
    agentlife.init(patch_openai=False)

    with agentlife.group("my-batch") as gid:
        assert gid.startswith("g-")
        for i in range(3):
            with agentlife.session(f"s-{i}", sample_index=i) as sess:
                assert sess.group_id == gid
                assert sess.group_name == "my-batch"
                assert sess.sample_index == i


# ── Streaming mock ──


def test_traced_stream_wrapper():
    from agentlife.patchers.openai_patcher import _TracedStream

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta = MagicMock(content="Hello")
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta = MagicMock(content=" World")
    chunk2.choices[0].finish_reason = "stop"
    chunk2.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    mock_stream = iter([chunk1, chunk2])
    mock_span = MagicMock()
    mock_collector = MagicMock()

    ts = _TracedStream(mock_stream, mock_span, mock_collector)
    chunks = list(ts)

    assert len(chunks) == 2
    mock_collector.end_span.assert_called_once()
    call_kwargs = mock_collector.end_span.call_args
    assert call_kwargs[1]["output_data"]["content"] == "Hello World"
    assert call_kwargs[1]["output_data"]["streamed"] is True
    assert call_kwargs[1]["prompt_tokens"] == 10
    assert call_kwargs[1]["completion_tokens"] == 5


def test_extract_output_with_tool_calls():
    from agentlife.patchers.openai_patcher import _extract_output

    mock_response = MagicMock()
    mock_tc = MagicMock()
    mock_tc.id = "call_abc123"
    mock_tc.type = "function"
    mock_tc.function.name = "get_weather"
    mock_tc.function.arguments = '{"city": "Tokyo"}'

    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tc]
    mock_response.choices[0].finish_reason = "tool_calls"

    output = _extract_output(mock_response)
    assert output["role"] == "assistant"
    assert output["content"] is None
    assert len(output["tool_calls"]) == 1
    assert output["tool_calls"][0]["function"]["name"] == "get_weather"
    assert output["tool_calls"][0]["function"]["arguments"] == '{"city": "Tokyo"}'


def test_extract_input_with_tools():
    from agentlife.patchers.openai_patcher import _extract_input

    kwargs = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [
            {"type": "function", "function": {"name": "get_weather", "description": "Get weather for a city"}},
        ],
        "tool_choice": "auto",
    }
    inp = _extract_input(kwargs)
    assert inp["tool_choice"] == "auto"
    assert len(inp["tools"]) == 1
    assert inp["tools"][0]["name"] == "get_weather"


def test_stream_accumulator_tool_calls():
    from agentlife.patchers.openai_patcher import _StreamAccumulator

    acc = _StreamAccumulator()

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta = MagicMock(content=None)
    tc_delta1 = MagicMock()
    tc_delta1.index = 0
    tc_delta1.id = "call_xyz"
    tc_delta1.function = MagicMock(name="get_weather", arguments='{"ci')
    tc_delta1.function.name = "get_weather"
    tc_delta1.function.arguments = '{"ci'
    chunk1.choices[0].delta.tool_calls = [tc_delta1]
    chunk1.choices[0].finish_reason = None
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta = MagicMock(content=None)
    tc_delta2 = MagicMock()
    tc_delta2.index = 0
    tc_delta2.id = None
    tc_delta2.function = MagicMock(name=None, arguments='ty":"Tokyo"}')
    tc_delta2.function.name = None
    tc_delta2.function.arguments = 'ty":"Tokyo"}'
    chunk2.choices[0].delta.tool_calls = [tc_delta2]
    chunk2.choices[0].finish_reason = "tool_calls"
    chunk2.usage = None

    acc.process_chunk(chunk1)
    acc.process_chunk(chunk2)

    output = acc.build_output()
    assert len(output["tool_calls"]) == 1
    assert output["tool_calls"][0]["function"]["name"] == "get_weather"
    assert output["tool_calls"][0]["function"]["arguments"] == '{"city":"Tokyo"}'
    assert output["finish_reason"] == "tool_calls"


def test_traced_stream_error():
    from agentlife.patchers.openai_patcher import _TracedStream

    def _error_gen():
        raise ValueError("API error")
        yield  # noqa: unreachable

    mock_span = MagicMock()
    mock_collector = MagicMock()

    ts = _TracedStream(_error_gen(), mock_span, mock_collector)
    with pytest.raises(ValueError, match="API error"):
        list(ts)

    mock_collector.end_span.assert_called_once()
    assert mock_collector.end_span.call_args[1]["error"] == "API error"
