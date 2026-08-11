from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from meetingintel.extraction import extract_all_chunks, extract_chunk, reduce_extractions, run_extraction
from meetingintel.llm_client import compute_cost
from meetingintel.models import Chunk, ChunkExtraction, PipelineConfig, SentimentScore, Transcript, Utterance


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


def _chunk_extraction(chunk_id: int) -> ChunkExtraction:
    return ChunkExtraction(
        chunk_id=chunk_id,
        action_items=[],
        decisions=[],
        sentiment=SentimentScore(label="neutral", score=0.0),
        tokens_in=10,
        tokens_out=5,
        latency_ms=1.0,
        cost_usd=0.001,
    )


def _reduce_response(input_tokens=200, output_tokens=80):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={
            "executive_summary": "Team discussed the roadmap.",
            "action_items": [{"description": "ship the merged report", "owner": "Alice", "due_date": None}],
            "decisions": [{"description": "adopt plan A", "decided_by": "Bob"}],
            "overall_sentiment": {"label": "positive", "score": 0.4, "rationale": None},
        })],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_reduce_extractions_returns_payload_and_meta():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={
            "executive_summary": "Team discussed the roadmap.",
            "action_items": [{"description": "ship the merged report", "owner": "Alice", "due_date": None}],
            "decisions": [{"description": "adopt plan A", "decided_by": "Bob"}],
            "overall_sentiment": {"label": "positive", "score": 0.4, "rationale": None},
        })],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=200, output_tokens=80),
    )
    payload, meta = reduce_extractions([_chunk_extraction(0), _chunk_extraction(1)], PipelineConfig(), client=client)
    assert payload.executive_summary == "Team discussed the roadmap."
    assert payload.action_items[0].description == "ship the merged report"
    assert meta["cost_usd"] == pytest.approx(compute_cost(200, 80, "reduce"))


def _transcript_n(n: int) -> Transcript:
    utterances = [
        Utterance(speaker="Alice", text=" ".join(f"w{i}_{k}" for k in range(5)), index=i)
        for i in range(n)
    ]
    return Transcript(meeting_id="m1", title="Test", utterances=utterances)


def test_run_extraction_empty_transcript_returns_empty_summary_without_api_calls():
    client = MagicMock()
    empty_transcript = Transcript(meeting_id="m1", title="Test", utterances=[])
    summary = run_extraction(empty_transcript, PipelineConfig(), client=client)
    assert summary.action_items == []
    assert summary.api_calls_map == 0
    assert summary.api_calls_reduce == 0
    assert client.messages.create.call_count == 0


def test_run_extraction_sums_cost_and_stamps_reduce_source():
    config = PipelineConfig()
    client = MagicMock()

    def fake_create(**kwargs):
        if kwargs["model"] == config.map_model:
            return _tool_use_response(
                action_items=[{"description": "task", "owner": "Alice", "due_date": None}],
                decisions=[],
                sentiment={"label": "neutral", "score": 0.0, "rationale": None},
                input_tokens=100,
                output_tokens=50,
            )
        return _reduce_response(input_tokens=200, output_tokens=80)

    client.messages.create.side_effect = fake_create
    # chunk_size (5) is smaller than a single utterance (5 words ~ 7 tokens),
    # so each of the 3 utterances forces its own chunk -> exactly 3 map calls.
    transcript = _transcript_n(3)
    config = PipelineConfig(chunk_size=5, overlap=0)

    summary = run_extraction(transcript, config, client=client, parallelism=2)

    assert summary.api_calls_map == 3
    assert summary.api_calls_reduce == 1
    expected_cost = 3 * compute_cost(100, 50, "map") + compute_cost(200, 80, "reduce")
    assert summary.cost_usd == pytest.approx(expected_cost)
    assert all(a.source_chunk_id == "reduce" for a in summary.action_items)
    assert all(d.source_chunk_id == "reduce" for d in summary.decisions)
