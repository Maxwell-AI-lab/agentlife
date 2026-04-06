"""Monkey-patch the OpenAI Python client to auto-trace chat completions."""

from __future__ import annotations

import functools
import logging
from typing import Any

from agentlife.collector import Collector
from agentlife.models import SpanType

logger = logging.getLogger("agentlife")

_patched = False


def patch_openai() -> None:
    """Patch openai.resources.chat.completions.Completions.create (sync & async)."""
    global _patched
    if _patched:
        return

    try:
        from openai.resources.chat import completions as _mod
    except ImportError:
        logger.debug("openai package not installed — skipping OpenAI patcher")
        return

    _original_create = _mod.Completions.create
    _original_async_create = _mod.AsyncCompletions.create

    @functools.wraps(_original_create)
    def _traced_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return _wrap_sync(_original_create, self, *args, **kwargs)

    @functools.wraps(_original_async_create)
    async def _traced_async_create(self: Any, *args: Any, **kwargs: Any) -> Any:
        return await _wrap_async(_original_async_create, self, *args, **kwargs)

    _mod.Completions.create = _traced_create  # type: ignore[assignment]
    _mod.AsyncCompletions.create = _traced_async_create  # type: ignore[assignment]
    _patched = True
    logger.debug("OpenAI client patched successfully")


def _extract_input(kwargs: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "messages": kwargs.get("messages"),
        "model": kwargs.get("model"),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
        "stream": kwargs.get("stream", False),
    }
    if kwargs.get("tools"):
        result["tools"] = _summarize_tools(kwargs["tools"])
    if kwargs.get("tool_choice"):
        result["tool_choice"] = kwargs["tool_choice"]
    return result


def _summarize_tools(tools: list[Any]) -> list[dict]:
    """Extract tool name + description for compact display."""
    out = []
    for t in tools[:20]:
        if isinstance(t, dict):
            fn = t.get("function", {})
            out.append({"name": fn.get("name"), "description": fn.get("description", "")[:80]})
        else:
            try:
                fn = t.function
                out.append({"name": fn.name, "description": (fn.description or "")[:80]})
            except AttributeError:
                out.append({"raw": str(t)[:100]})
    return out


def _extract_output(response: Any) -> dict[str, Any]:
    try:
        choice = response.choices[0]
        result: dict[str, Any] = {
            "role": getattr(choice.message, "role", None),
            "content": getattr(choice.message, "content", None),
            "finish_reason": getattr(choice, "finish_reason", None),
        }
        tool_calls = getattr(choice.message, "tool_calls", None)
        if tool_calls:
            result["tool_calls"] = [
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", "function"),
                    "function": {
                        "name": getattr(tc.function, "name", None),
                        "arguments": getattr(tc.function, "arguments", None),
                    },
                }
                for tc in tool_calls
            ]
        return result
    except (IndexError, AttributeError):
        return {"raw": str(response)}


def _extract_usage(response: Any) -> tuple[int, int]:
    try:
        usage = response.usage
        return (usage.prompt_tokens or 0, usage.completion_tokens or 0)
    except AttributeError:
        return (0, 0)


# ── Sync wrappers ──


def _wrap_sync(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    collector = Collector.get()
    model = kwargs.get("model", "unknown")
    is_stream = kwargs.get("stream", False)

    span = collector.start_span(
        name=f"chat.completions.create({model})",
        span_type=SpanType.LLM,
        input_data=_extract_input(kwargs),
        model=model,
    )
    try:
        response = original(self, *args, **kwargs)
        if is_stream:
            return _TracedStream(response, span, collector)
        prompt_tok, comp_tok = _extract_usage(response)
        collector.end_span(
            span,
            output_data=_extract_output(response),
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
        )
        return response
    except Exception as exc:
        collector.end_span(span, error=str(exc))
        raise


async def _wrap_async(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    collector = Collector.get()
    model = kwargs.get("model", "unknown")
    is_stream = kwargs.get("stream", False)

    span = collector.start_span(
        name=f"chat.completions.create({model})",
        span_type=SpanType.LLM,
        input_data=_extract_input(kwargs),
        model=model,
    )
    try:
        response = await original(self, *args, **kwargs)
        if is_stream:
            return _TracedAsyncStream(response, span, collector)
        prompt_tok, comp_tok = _extract_usage(response)
        collector.end_span(
            span,
            output_data=_extract_output(response),
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
        )
        return response
    except Exception as exc:
        collector.end_span(span, error=str(exc))
        raise


# ── Streaming wrappers ──


class _StreamAccumulator:
    """Shared logic for accumulating streamed chunks (content + tool_calls)."""

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.finish_reason: str | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_calls: dict[int, dict] = {}

    def process_chunk(self, chunk: Any) -> None:
        try:
            if not chunk.choices:
                pass
            else:
                delta = chunk.choices[0].delta

                if hasattr(delta, "content") and delta.content:
                    self.content_parts.append(delta.content)

                tc_deltas = getattr(delta, "tool_calls", None)
                if tc_deltas:
                    for tc in tc_deltas:
                        idx = getattr(tc, "index", 0)
                        if idx not in self.tool_calls:
                            self.tool_calls[idx] = {
                                "id": getattr(tc, "id", None),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        entry = self.tool_calls[idx]
                        if tc.id:
                            entry["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn:
                            if getattr(fn, "name", None):
                                entry["function"]["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                entry["function"]["arguments"] += fn.arguments

                fr = getattr(chunk.choices[0], "finish_reason", None)
                if fr:
                    self.finish_reason = fr

            usage = getattr(chunk, "usage", None)
            if usage:
                self.prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                self.completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        except (IndexError, AttributeError):
            pass

    def build_output(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(self.content_parts),
            "finish_reason": self.finish_reason or "stop",
            "streamed": True,
        }
        if self.tool_calls:
            result["tool_calls"] = [self.tool_calls[k] for k in sorted(self.tool_calls)]
        return result


class _TracedStream:
    """Wraps an OpenAI sync Stream to capture streamed content for tracing."""

    def __init__(self, stream: Any, span: Any, collector: Collector):
        self._stream = stream
        self._span = span
        self._collector = collector
        self._acc = _StreamAccumulator()
        self._finalized = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self) -> _TracedStream:
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._stream)
            self._acc.process_chunk(chunk)
            return chunk
        except StopIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize_error(str(e))
            raise

    def __enter__(self) -> _TracedStream:
        if hasattr(self._stream, "__enter__"):
            self._stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        self._finalize()
        if hasattr(self._stream, "__exit__"):
            self._stream.__exit__(*args)

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._collector.end_span(
            self._span,
            output_data=self._acc.build_output(),
            prompt_tokens=self._acc.prompt_tokens,
            completion_tokens=self._acc.completion_tokens,
        )

    def _finalize_error(self, error: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._collector.end_span(self._span, error=error)


class _TracedAsyncStream:
    """Wraps an OpenAI async Stream to capture streamed content for tracing."""

    def __init__(self, stream: Any, span: Any, collector: Collector):
        self._stream = stream
        self._span = span
        self._collector = collector
        self._acc = _StreamAccumulator()
        self._finalized = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __aiter__(self) -> _TracedAsyncStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._stream.__anext__()
            self._acc.process_chunk(chunk)
            return chunk
        except StopAsyncIteration:
            self._finalize()
            raise
        except Exception as e:
            self._finalize_error(str(e))
            raise

    async def __aenter__(self) -> _TracedAsyncStream:
        if hasattr(self._stream, "__aenter__"):
            await self._stream.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        self._finalize()
        if hasattr(self._stream, "__aexit__"):
            await self._stream.__aexit__(*args)

    def _finalize(self) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._collector.end_span(
            self._span,
            output_data=self._acc.build_output(),
            prompt_tokens=self._acc.prompt_tokens,
            completion_tokens=self._acc.completion_tokens,
        )

    def _finalize_error(self, error: str) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._collector.end_span(self._span, error=error)
