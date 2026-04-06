"""AgentLife — Peek inside your AI agents."""

from __future__ import annotations

import functools
import inspect
import uuid
from contextlib import contextmanager
from typing import Any, Callable, TypeVar

from agentlife.collector import Collector
from agentlife.models import SpanType

__version__ = "0.1.0"
__all__ = ["init", "session", "group", "trace"]

F = TypeVar("F", bound=Callable[..., Any])

_initialized = False


def init(*, patch_openai: bool = True) -> None:
    """Initialize AgentLife. Call once at the top of your script.

    Args:
        patch_openai: If True, automatically patch the OpenAI client to trace
                      all chat.completions calls.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    Collector.get()  # ensure singleton is created

    if patch_openai:
        from agentlife.patchers.openai_patcher import patch_openai as _do_patch
        _do_patch()


@contextmanager
def group(name: str, metadata: dict | None = None):
    """Context manager that groups multiple rollout samples together.

    Usage::

        with agentlife.group("batch-001"):
            for i in range(4):
                with agentlife.session(f"sample-{i}", sample_index=i):
                    result = agent.run(task)
    """
    group_id = f"g-{uuid.uuid4().hex[:10]}"
    collector = Collector.get()
    collector.set_group(group_id)
    try:
        yield group_id
    finally:
        collector.set_group(None)


@contextmanager
def session(
    name: str = "unnamed",
    metadata: dict | None = None,
    group_id: str | None = None,
    sample_index: int | None = None,
):
    """Context manager that wraps an agent run as a traced session.

    Usage::

        import agentlife
        agentlife.init()

        with agentlife.session("my-task"):
            # ... your agent code ...
    """
    collector = Collector.get()
    sess = collector.start_session(
        name=name, metadata=metadata,
        group_id=group_id, sample_index=sample_index,
    )
    try:
        yield sess
        collector.end_session(sess)
    except Exception as exc:
        collector.end_session(sess, error=str(exc))
        raise


def trace(
    fn: F | None = None,
    *,
    name: str | None = None,
    span_type: SpanType = SpanType.FUNCTION,
) -> F | Callable[[F], F]:
    """Decorator that wraps a function as a traced span.

    Usage::

        @agentlife.trace
        def my_step(query: str) -> str:
            ...

        @agentlife.trace(name="search", span_type=SpanType.TOOL)
        def search(query: str) -> str:
            ...
    """

    def _decorator(func: F) -> F:
        span_name = name or func.__name__

        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                collector = Collector.get()
                span = collector.start_span(
                    name=span_name,
                    span_type=span_type,
                    input_data={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)},
                )
                try:
                    result = await func(*args, **kwargs)
                    collector.end_span(span, output_data=_safe_repr(result))
                    return result
                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            return _async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(func)
            def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                collector = Collector.get()
                span = collector.start_span(
                    name=span_name,
                    span_type=span_type,
                    input_data={"args": _safe_repr(args), "kwargs": _safe_repr(kwargs)},
                )
                try:
                    result = func(*args, **kwargs)
                    collector.end_span(span, output_data=_safe_repr(result))
                    return result
                except Exception as exc:
                    collector.end_span(span, error=str(exc))
                    raise

            return _sync_wrapper  # type: ignore[return-value]

    if fn is not None:
        return _decorator(fn)
    return _decorator  # type: ignore[return-value]


def _safe_repr(value: Any) -> Any:
    """Best-effort JSON-safe representation."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_repr(v) for v in value[:20]]
    if isinstance(value, dict):
        return {str(k): _safe_repr(v) for k, v in list(value.items())[:20]}
    try:
        return str(value)[:500]
    except Exception:
        return "<unserializable>"
