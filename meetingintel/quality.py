"""Meeting-quality scoring - a bonus, opt-in analysis over an already-
computed MeetingSummary. Its cost is never folded into MeetingSummary.
cost_usd or run_extraction()'s totals; callers invoke this explicitly, on
top of an already-completed extraction, when they want it.
"""

from __future__ import annotations

import json

from anthropic import Anthropic
from pydantic import BaseModel, Field

from meetingintel.llm_client import timed_forced_tool_call
from meetingintel.models import MeetingSummary, PipelineConfig, Transcript

_QUALITY_SYSTEM_PROMPT = (
    "Score this meeting's quality along three dimensions, each 0-10:\n"
    "- engagement_level: how balanced participation was across attendees, "
    "versus one person dominating the conversation.\n"
    "- decision_clarity: whether decisions were explicit and specific, "
    "versus vague or left open.\n"
    "- follow_up_rate: whether action items had concrete owners and due "
    "dates, versus abstract intentions with no accountability.\n"
    "Use both the raw transcript and the extracted summary to judge this, "
    "and give a brief overall rationale."
)


class MeetingQualityScore(BaseModel):
    engagement_level: float = Field(ge=0, le=10)
    decision_clarity: float = Field(ge=0, le=10)
    follow_up_rate: float = Field(ge=0, le=10)
    rationale: str


def score_meeting_quality(
    transcript: Transcript,
    summary: MeetingSummary,
    config: PipelineConfig,
    client: Anthropic | None = None,
) -> MeetingQualityScore:
    client = client or Anthropic()
    payload = json.dumps(
        {
            "transcript": [u.model_dump() for u in transcript.utterances],
            "summary": summary.model_dump(),
        },
        default=str,
    )
    score, _meta = timed_forced_tool_call(
        client,
        config.reduce_model,
        _QUALITY_SYSTEM_PROMPT,
        payload,
        MeetingQualityScore,
        "record_quality_score",
        phase="reduce",
    )
    return score
