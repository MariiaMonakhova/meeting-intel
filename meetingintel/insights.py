"""Per-attendee insights.

action_item_count is tallied in pure Python from already-merged action items
(no LLM call, no cost) - it's a mechanical count, not something that needs
model judgment. sentiment_summary and notable_quotes come from one
additional forced-tool-call (Sonnet) covering every attendee in a single
call, rather than one call per attendee.
"""

from __future__ import annotations

from collections import Counter

from anthropic import Anthropic
from pydantic import BaseModel, Field

from meetingintel.llm_client import timed_forced_tool_call
from meetingintel.models import ActionItem, AttendeeInsight, PipelineConfig, Transcript

_INSIGHTS_SYSTEM_PROMPT = (
    "For each attendee listed, summarize their sentiment/tone during the "
    "meeting in one sentence and pick 1-2 notable direct quotes from them, "
    "if any stand out. Cover every listed attendee, even if briefly - do "
    "not skip anyone or invent attendees not in the list."
)


class _AttendeeInsightPayload(BaseModel):
    name: str
    sentiment_summary: str
    notable_quotes: list[str] = Field(default_factory=list)


class _AttendeeInsightsPayload(BaseModel):
    insights: list[_AttendeeInsightPayload]


def _tally_action_items(action_items: list[ActionItem]) -> Counter:
    return Counter(a.owner for a in action_items if a.owner)


def _format_transcript(transcript: Transcript) -> str:
    lines = [f"{u.speaker}: {u.text}" for u in transcript.utterances]
    return "\n".join(lines) + f"\n\nAttendees: {', '.join(transcript.attendees)}"


def build_attendee_insights(
    transcript: Transcript,
    action_items: list[ActionItem],
    config: PipelineConfig,
    client: Anthropic | None = None,
) -> tuple[list[AttendeeInsight], dict]:
    """Returns (insights, meta) - meta mirrors reduce_extractions's shape
    (at least cost_usd) so callers can fold this call's cost into a running
    total the same way they do for the map/reduce calls.

    Note: action_item_count is matched to the model's returned `name` by
    exact string equality against action item owners - no fuzzy matching, so
    a name spelled differently between the two will silently count as 0.
    """
    if not transcript.attendees:
        return [], {"cost_usd": 0.0}

    client = client or Anthropic()
    counts = _tally_action_items(action_items)

    payload, meta = timed_forced_tool_call(
        client,
        config.reduce_model,
        _INSIGHTS_SYSTEM_PROMPT,
        _format_transcript(transcript),
        _AttendeeInsightsPayload,
        "record_attendee_insights",
        phase="reduce",
    )

    insights = [
        AttendeeInsight(
            name=insight.name,
            action_item_count=counts.get(insight.name, 0),
            sentiment_summary=insight.sentiment_summary,
            notable_quotes=insight.notable_quotes,
        )
        for insight in payload.insights
    ]
    return insights, meta
