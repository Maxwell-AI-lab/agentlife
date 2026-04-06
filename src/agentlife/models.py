"""Core data models for AgentLife traces."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    RUNNING = "running"


class SpanType(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    FUNCTION = "function"


class Span(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_id: str
    parent_span_id: str | None = None
    span_type: SpanType = SpanType.FUNCTION
    name: str = ""
    status: SpanStatus = SpanStatus.RUNNING
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str | None = None
    duration_ms: float | None = None

    # IO
    input_data: Any | None = None
    output_data: Any | None = None
    error: str | None = None

    # LLM-specific
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None

    # Arbitrary metadata
    metadata: dict[str, Any] = Field(default_factory=dict)


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = "unnamed"
    group_id: str | None = None
    group_name: str | None = None
    sample_index: int | None = None
    status: SpanStatus = SpanStatus.RUNNING
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: str | None = None

    total_tokens: int = 0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0
    span_count: int = 0
    error_count: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)


# Token cost per million tokens (input / output) for common models
MODEL_COSTS: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-3.5-turbo": (0.50, 1.50),
    "o1": (15.00, 60.00),
    "o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-4-opus": (15.00, 75.00),
    "claude-4-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-opus": (15.00, 75.00),
    # DeepSeek
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
    # GLM (Zhipu AI)
    "glm-4-flash": (0.00, 0.00),
    "glm-4-air": (0.50, 0.50),
    "glm-4-plus": (5.00, 5.00),
    "glm-4": (1.00, 1.00),
    # Qwen (Alibaba)
    "qwen-turbo": (0.30, 0.60),
    "qwen-plus": (0.80, 2.00),
    "qwen-max": (2.40, 9.60),
    # Doubao (ByteDance)
    "doubao-pro": (0.80, 2.00),
    "doubao-lite": (0.30, 0.60),
}


def estimate_cost(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost in USD based on model and token counts."""
    if not model:
        return 0.0
    key = model.lower().strip()
    for name, (inp_cost, out_cost) in MODEL_COSTS.items():
        if name in key:
            return (prompt_tokens * inp_cost + completion_tokens * out_cost) / 1_000_000
    return 0.0
