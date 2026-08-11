"""MapReduce extraction pipeline.

Map phase: extract_chunk() runs one forced-tool-call per chunk (Haiku),
extract_all_chunks() parallelizes that across chunks with a ThreadPoolExecutor
- the "MapReduce / parallel chunking" learning goal made concrete. Reduce
phase (structured merge across chunks into a MeetingSummary) is added in the
next step.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from anthropic import Anthropic
from pydantic import BaseModel

from meetingintel.llm_client import timed_forced_tool_call
from meetingintel.models import ActionItem, Chunk, ChunkExtraction, Decision, PipelineConfig, SentimentScore

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
