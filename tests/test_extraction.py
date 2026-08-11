from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from meetingintel.extraction import extract_all_chunks, extract_chunk
from meetingintel.llm_client import compute_cost
from meetingintel.models import Chunk, PipelineConfig, Utterance


def _chunk(chunk_id: int, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        utterances=[Utterance(speaker="Alice", text=text, index=chunk_id)],
        start_index=chunk_id,
        end_index=chunk_id,
        token_count=5,
        overlap_tokens=0,
        speakers=["Alice"],
    )


def _tool_use_response(action_items, decisions, sentiment, input_tokens=50, output_tokens=20):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={
            "action_items": action_items,
            "decisions": decisions,
            "sentiment": sentiment,
        })],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_extract_chunk_stamps_chunk_id_and_computes_cost():
    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(
        action_items=[{"description": "ship the report", "owner": "Alice", "due_date": None}],
        decisions=[{"description": "use plan A", "decided_by": "Bob"}],
        sentiment={"label": "positive", "score": 0.5, "rationale": None},
        input_tokens=50,
        output_tokens=20,
    )
    chunk = _chunk(3, "irrelevant text")
    result = extract_chunk(chunk, PipelineConfig(), client=client)

    assert result.chunk_id == 3
    assert result.action_items[0].source_chunk_id == 3
    assert result.action_items[0].owner == "Alice"
    assert result.decisions[0].source_chunk_id == 3
    assert result.tokens_in == 50
    assert result.tokens_out == 20
    assert result.cost_usd == pytest.approx(compute_cost(50, 20, "map"))


def test_extract_chunk_with_no_action_items_or_decisions():
    client = MagicMock()
    client.messages.create.return_value = _tool_use_response(
        action_items=[], decisions=[], sentiment={"label": "neutral", "score": 0.0, "rationale": None},
    )
    result = extract_chunk(_chunk(0, "small talk"), PipelineConfig(), client=client)
    assert result.action_items == []
    assert result.decisions == []


def test_extract_all_chunks_preserves_input_order_regardless_of_completion_order():
    client = MagicMock()

    def fake_create(**kwargs):
        content = kwargs["messages"][0]["content"]
        marker = content.split()[-1]
        # earlier chunks sleep longer, so later chunks would finish first
        # if the implementation didn't preserve order correctly
        import time

        idx = int(marker.replace("MARK", ""))
        time.sleep(0.01 * (5 - idx) / 100)
        return _tool_use_response(
            action_items=[{"description": f"action-for-{marker}", "owner": None, "due_date": None}],
            decisions=[],
            sentiment={"label": "neutral", "score": 0.0, "rationale": None},
        )

    client.messages.create.side_effect = fake_create
    chunks = [_chunk(i, f"MARK{i}") for i in range(5)]
    results = extract_all_chunks(chunks, PipelineConfig(), client=client, parallelism=4)

    assert [r.chunk_id for r in results] == [0, 1, 2, 3, 4]
    for i, r in enumerate(results):
        assert r.action_items[0].description == f"action-for-MARK{i}"
        assert r.action_items[0].source_chunk_id == i
