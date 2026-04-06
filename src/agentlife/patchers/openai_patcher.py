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
    return {
        "messages": kwargs.get("messages"),
        "model": kwargs.get("model"),
        "temperature": kwargs.get("temperature"),
        "max_tokens": kwargs.get("max_tokens"),
    }


def _extract_output(response: Any) -> dict[str, Any]:
    try:
        choice = response.choices[0]
        return {
            "role": getattr(choice.message, "role", None),
            "content": getattr(choice.message, "content", None),
            "finish_reason": getattr(choice, "finish_reason", None),
        }
    except (IndexError, AttributeError):
        return {"raw": str(response)}


def _extract_usage(response: Any) -> tuple[int, int]:
    try:
        usage = response.usage
        return (usage.prompt_tokens or 0, usage.completion_tokens or 0)
    except AttributeError:
        return (0, 0)


def _wrap_sync(original: Any, self: Any, *args: Any, **kwargs: Any) -> Any:
    collector = Collector.get()
    model = kwargs.get("model", "unknown")
    span = collector.start_span(
        name=f"chat.completions.create({model})",
        span_type=SpanType.LLM,
        input_data=_extract_input(kwargs),
        model=model,
    )
    try:
        response = original(self, *args, **kwargs)
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
    span = collector.start_span(
        name=f"chat.completions.create({model})",
        span_type=SpanType.LLM,
        input_data=_extract_input(kwargs),
        model=model,
    )
    try:
        response = await original(self, *args, **kwargs)
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
