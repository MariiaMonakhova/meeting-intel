import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from meetingintel.llm_client import (
    PRICING,
    ToolCallError,
    _build_tool,
    _forced_tool_call,
    compute_cost,
    timed_forced_tool_call,
)


class Dummy(BaseModel):
    value: int


def _fake_response(tool_input: dict | None, input_tokens: int = 100, output_tokens: int = 20, stop_reason: str = "tool_use"):
    blocks = []
    if tool_input is not None:
        blocks.append(SimpleNamespace(type="tool_use", input=tool_input))
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _fake_client(*responses):
    client = MagicMock()
    client.messages.create.side_effect = list(responses)
    return client


def test_successful_call_returns_validated_model():
    client = _fake_client(_fake_response({"value": 42}))
    parsed, resp = _forced_tool_call(client, "model", "sys", "hi", Dummy, "record_dummy")
    assert parsed == Dummy(value=42)
    assert client.messages.create.call_count == 1


def test_validation_error_triggers_retry_then_succeeds():
    client = _fake_client(
        _fake_response({"value": "not-an-int"}),
        _fake_response({"value": 7}),
    )
    parsed, _resp = _forced_tool_call(client, "model", "sys", "hi", Dummy, "record_dummy", max_retries=1)
    assert parsed.value == 7
    assert client.messages.create.call_count == 2


def test_validation_error_exhausts_retries_raises():
    client = _fake_client(
        _fake_response({"value": "bad"}),
        _fake_response({"value": "still-bad"}),
    )
    with pytest.raises(ValidationError):
        _forced_tool_call(client, "model", "sys", "hi", Dummy, "record_dummy", max_retries=1)
    assert client.messages.create.call_count == 2


def test_missing_tool_use_block_raises_tool_call_error():
    client = _fake_client(_fake_response(None, stop_reason="max_tokens"))
    with pytest.raises(ToolCallError):
        _forced_tool_call(client, "model", "sys", "hi", Dummy, "record_dummy")


def test_timed_forced_tool_call_computes_meta():
    client = _fake_client(_fake_response({"value": 1}, input_tokens=1000, output_tokens=1000))
    parsed, meta = timed_forced_tool_call(client, "model", "sys", "hi", Dummy, "record_dummy", phase="map")
    assert parsed.value == 1
    assert meta["tokens_in"] == 1000
    assert meta["tokens_out"] == 1000
    assert meta["latency_ms"] >= 0
    assert meta["cost_usd"] == pytest.approx(0.001 + 0.005)


@pytest.mark.parametrize(
    "tokens_in,tokens_out,phase,expected",
    [
        (1000, 1000, "map", PRICING["map"]["input_per_1k"] + PRICING["map"]["output_per_1k"]),
        (2000, 500, "reduce", 2 * PRICING["reduce"]["input_per_1k"] + 0.5 * PRICING["reduce"]["output_per_1k"]),
        (0, 0, "map", 0.0),
    ],
)
def test_compute_cost_math(tokens_in, tokens_out, phase, expected):
    assert compute_cost(tokens_in, tokens_out, phase) == pytest.approx(expected)


def test_build_tool_forbids_additional_properties():
    tool = _build_tool(Dummy, "record_dummy")
    assert tool["input_schema"]["additionalProperties"] is False
    assert tool["name"] == "record_dummy"


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="requires a real ANTHROPIC_API_KEY")
def test_real_api_call_extracts_structured_value():
    from anthropic import Anthropic

    client = Anthropic()
    parsed, _resp = _forced_tool_call(
        client, "claude-haiku-4-5", "Extract the number mentioned.",
        "The answer is 42.", Dummy, "record_dummy",
    )
    assert parsed.value == 42
