"""Data models shared across the meeting-intel pipeline.

Pydantic (not plain dataclasses) so that the extraction-facing models can
double as the source of both the JSON schema sent to Claude's tool-use API
(model_json_schema()) and the validator for what comes back (model_validate()).
"""

from __future__ import annotations

from datetime import date as Date
from typing import Literal

from pydantic import BaseModel, Field

ChunkStrategy = Literal["speaker_semantic"]
AggPattern = Literal["mapreduce"]


class Utterance(BaseModel):
    speaker: str
    text: str
    timestamp: str | None = None  # "HH:MM:SS" if present in the source transcript
    index: int  # stable position within the transcript


class Transcript(BaseModel):
    meeting_id: str
    title: str
    date: Date | None = None
    attendees: list[str] = Field(default_factory=list)
    utterances: list[Utterance]


class Chunk(BaseModel):
    id: int
    utterances: list[Utterance]  # whole utterances only, never split mid-utterance
    start_index: int
    end_index: int
    token_count: int
    overlap_tokens: int
    speakers: list[str]


class ActionItem(BaseModel):
    description: str
    owner: str | None = None
    due_date: str | None = None
    source_chunk_id: int | str  # int during map phase, "reduce" after merging


class Decision(BaseModel):
    description: str
    decided_by: str | None = None
    source_chunk_id: int | str


class SentimentScore(BaseModel):
    label: Literal["positive", "neutral", "negative", "mixed"]
    score: float = Field(ge=-1.0, le=1.0)
    rationale: str | None = None


class ChunkExtraction(BaseModel):
    """Map-phase output for a single chunk."""

    chunk_id: int
    action_items: list[ActionItem]
    decisions: list[Decision]
    sentiment: SentimentScore
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float


class AttendeeInsight(BaseModel):
    name: str
    action_item_count: int
    sentiment_summary: str
    notable_quotes: list[str] = Field(default_factory=list)


class MeetingSummary(BaseModel):
    """Final reduce-phase output for a meeting."""

    meeting_id: str
    executive_summary: str
    action_items: list[ActionItem]  # merged/deduped across chunks
    decisions: list[Decision]  # merged/deduped across chunks
    attendee_insights: list[AttendeeInsight]
    overall_sentiment: SentimentScore
    cost_usd: float
    api_calls_map: int
    api_calls_reduce: int
    wall_clock_s: float


class PipelineConfig(BaseModel):
    chunk_strategy: ChunkStrategy = "speaker_semantic"
    chunk_size: int = 800  # tokens
    overlap: int = 100  # tokens
    agg_pattern: AggPattern = "mapreduce"
    map_model: str = "claude-haiku-4-5"
    reduce_model: str = "claude-sonnet-5"


class ActionItemGroundTruth(BaseModel):
    owner: str
    task: str


class EvalResult(BaseModel):
    meeting_id: str
    precision: float
    recall: float
    f1: float
    matched: list[tuple[str, str]]
    missed: list[ActionItemGroundTruth]
    spurious: list[ActionItem]
    decision_judge_score: float | None = None
    sentiment_judge_score: float | None = None
