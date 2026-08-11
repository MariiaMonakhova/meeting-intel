"""Claude API wrapper for structured (tool-use) outputs.

Every extraction/judging call in this project goes through _forced_tool_call:
give Claude a single tool whose input_schema is generated from a Pydantic
model, force it to call that tool, and validate what comes back against the
same model. This is the one place real network calls happen, and the one
place tokens_in/tokens_out/cost_usd/latency_ms get computed from a real
response rather than an estimate.
"""

from __future__ import annotations

import time
from typing import TypeVar

from anthropic import Anthropic
from anthropic.types import Message
from pydantic import BaseModel, ValidationError

# Real Anthropic API pricing, $ per 1K tokens. Two-tier split: a cheap/fast
# model for the map phase (per-chunk extraction), a stronger model for the
# reduce phase (cross-chunk merge/judgment).
PRICING = {
    "map": {"input_per_1k": 0.001, "output_per_1k": 0.005},  # Claude Haiku 4.5
    "reduce": {"input_per_1k": 0.003, "output_per_1k": 0.015},  # Claude Sonnet 5
}

ModelT = TypeVar("ModelT", bound=BaseModel)


class ToolCallError(RuntimeError):
    """Raised when the model's response doesn't contain the forced tool call."""


def _build_tool(schema_model: type[BaseModel], tool_name: str) -> dict:
    schema = schema_model.model_json_schema()
    schema["additionalProperties"] = False
    return {
        "name": tool_name,
        "description": f"Record the extracted {tool_name} as structured data.",
        "input_schema": schema,
    }


def _forced_tool_call(
    client: Anthropic,
    model: str,
    system: str,
    user_content: str,
    schema_model: type[ModelT],
    tool_name: str,
    max_retries: int = 2,
) -> tuple[ModelT, Message]:
    tool = _build_tool(schema_model, tool_name)
    content = user_content
    last_error: ValidationError | None = None

    for attempt in range(max_retries + 1):
        resp = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": content}],
        )
        tool_use = next((b for b in resp.content if b.type == "tool_use"), None)
        if tool_use is None:
            raise ToolCallError(
                f"Model response did not contain a '{tool_name}' tool call "
                f"(stop_reason={resp.stop_reason!r})"
            )
        try:
            return schema_model.model_validate(tool_use.input), resp
        except ValidationError as e:
            last_error = e
            content = f"{user_content}\n\n[Previous attempt failed validation: {e}. Retry with valid data.]"

    assert last_error is not None
    raise last_error


def compute_cost(tokens_in: int, tokens_out: int, phase: str) -> float:
    pricing = PRICING[phase]
    return (tokens_in / 1000) * pricing["input_per_1k"] + (tokens_out / 1000) * pricing["output_per_1k"]


def timed_forced_tool_call(
    client: Anthropic,
    model: str,
    system: str,
    user_content: str,
    schema_model: type[ModelT],
    tool_name: str,
    phase: str,
    max_retries: int = 2,
) -> tuple[ModelT, dict]:
    """Like _forced_tool_call, but also returns timing/cost metadata
    (tokens_in, tokens_out, latency_ms, cost_usd) computed from the real
    response - the shape every caller (map, reduce, insights, eval, quality)
    actually needs.
    """
    start = time.monotonic()
    parsed, resp = _forced_tool_call(
        client, model, system, user_content, schema_model, tool_name, max_retries
    )
    latency_ms = (time.monotonic() - start) * 1000
    meta = {
        "tokens_in": resp.usage.input_tokens,
        "tokens_out": resp.usage.output_tokens,
        "latency_ms": latency_ms,
        "cost_usd": compute_cost(resp.usage.input_tokens, resp.usage.output_tokens, phase),
    }
    return parsed, meta
