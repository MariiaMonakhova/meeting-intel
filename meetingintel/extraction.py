"""MapReduce extraction pipeline.

Map phase: extract_chunk() runs one forced-tool-call per chunk (Haiku),
extract_all_chunks() parallelizes that across chunks with a ThreadPoolExecutor
- the "MapReduce / parallel chunking" learning goal made concrete.

Reduce phase: reduce_extractions() is a single forced-tool-call (Sonnet) that
merges every chunk's extraction into one deduplicated result plus an
executive summary - a structured merge, not prose re-summarization, since
deduplicating near-identical action items needs model judgment.

run_extraction() orchestrates the whole thing: chunk -> parallel map ->
reduce -> MeetingSummary.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic
from pydantic import BaseModel

from meetingintel.chunking import chunk_speaker_semantic
from meetingintel.llm_client import timed_forced_tool_call
from meetingintel.models import (
    ActionItem,
    Chunk,
    ChunkExtraction,
    Decision,
    MeetingSummary,
    PipelineConfig,
    SentimentScore,
    Transcript,
)

_MAP_SYSTEM_PROMPT = (
    "You extract action items, decisions, and overall sentiment from a "
    "meeting transcript excerpt. Be precise about who owns each action item "
    "- only set an owner when the transcript makes it clear who is "
    "responsible, and use the exact name as it appears in the transcript."
)


class _ExtractedActionItem(BaseModel):
    """What we ask the model for - source_chunk_id is stamped in code, never
    requested from the model, since the model has no reliable way to know it."""

    description: str
    owner: str | None = None
    due_date: str | None = None


class _ExtractedDecision(BaseModel):
    description: str
    decided_by: str | None = None


class _ChunkExtractionPayload(BaseModel):
    action_items: list[_ExtractedActionItem]
    decisions: list[_ExtractedDecision]
    sentiment: SentimentScore


def _format_chunk(chunk: Chunk) -> str:
    return "\n".join(f"{u.speaker}: {u.text}" for u in chunk.utterances)


def extract_chunk(chunk: Chunk, config: PipelineConfig, client: Anthropic | None = None) -> ChunkExtraction:
    client = client or Anthropic()
    payload, meta = timed_forced_tool_call(
        client,
        config.map_model,
        _MAP_SYSTEM_PROMPT,
        _format_chunk(chunk),
        _ChunkExtractionPayload,
        "record_chunk_extraction",
        phase="map",
    )
    action_items = [
        ActionItem(description=a.description, owner=a.owner, due_date=a.due_date, source_chunk_id=chunk.id)
        for a in payload.action_items
    ]
    decisions = [
        Decision(description=d.description, decided_by=d.decided_by, source_chunk_id=chunk.id)
        for d in payload.decisions
    ]
    return ChunkExtraction(
        chunk_id=chunk.id,
        action_items=action_items,
        decisions=decisions,
        sentiment=payload.sentiment,
        tokens_in=meta["tokens_in"],
        tokens_out=meta["tokens_out"],
        latency_ms=meta["latency_ms"],
        cost_usd=meta["cost_usd"],
    )


def extract_all_chunks(
    chunks: list[Chunk],
    config: PipelineConfig,
    client: Anthropic | None = None,
    parallelism: int = 4,
) -> list[ChunkExtraction]:
    client = client or Anthropic()
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        return list(executor.map(lambda c: extract_chunk(c, config, client), chunks))


_REDUCE_SYSTEM_PROMPT = (
    "You merge structured meeting-extraction results from multiple transcript "
    "chunks into one coherent output. Deduplicate action items and decisions "
    "that refer to the same underlying item even if phrased differently, "
    "keeping the clearest phrasing and the most specific owner/due-date. "
    "Write a concise executive summary of the whole meeting."
)


class _ReducePayload(BaseModel):
    executive_summary: str
    action_items: list[_ExtractedActionItem]
    decisions: list[_ExtractedDecision]
    overall_sentiment: SentimentScore


def reduce_extractions(
    chunk_extractions: list[ChunkExtraction], config: PipelineConfig, client: Anthropic | None = None
) -> tuple[_ReducePayload, dict]:
    client = client or Anthropic()
    payload_json = json.dumps([ce.model_dump() for ce in chunk_extractions], default=str)
    return timed_forced_tool_call(
        client,
        config.reduce_model,
        _REDUCE_SYSTEM_PROMPT,
        payload_json,
        _ReducePayload,
        "record_meeting_summary",
        phase="reduce",
    )


def run_extraction(
    transcript: Transcript,
    config: PipelineConfig,
    client: Anthropic | None = None,
    parallelism: int = 4,
) -> MeetingSummary:
    client = client or Anthropic()
    start = time.monotonic()
    chunks = list(chunk_speaker_semantic(transcript, config.chunk_size, config.overlap))

    if not chunks:
        return MeetingSummary(
            meeting_id=transcript.meeting_id,
            executive_summary="",
            action_items=[],
            decisions=[],
            attendee_insights=[],
            overall_sentiment=SentimentScore(label="neutral", score=0.0),
            cost_usd=0.0,
            api_calls_map=0,
            api_calls_reduce=0,
            wall_clock_s=time.monotonic() - start,
        )

    chunk_extractions = extract_all_chunks(chunks, config, client=client, parallelism=parallelism)
    reduced, reduce_meta = reduce_extractions(chunk_extractions, config, client=client)

    action_items = [
        ActionItem(description=a.description, owner=a.owner, due_date=a.due_date, source_chunk_id="reduce")
        for a in reduced.action_items
    ]
    decisions = [
        Decision(description=d.description, decided_by=d.decided_by, source_chunk_id="reduce")
        for d in reduced.decisions
    ]
    map_cost = sum(ce.cost_usd for ce in chunk_extractions)

    return MeetingSummary(
        meeting_id=transcript.meeting_id,
        executive_summary=reduced.executive_summary,
        action_items=action_items,
        decisions=decisions,
        attendee_insights=[],
        overall_sentiment=reduced.overall_sentiment,
        cost_usd=map_cost + reduce_meta["cost_usd"],
        api_calls_map=len(chunks),
        api_calls_reduce=1,
        wall_clock_s=time.monotonic() - start,
    )
