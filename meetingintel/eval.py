"""Evaluation harness.

Two different strategies for two different kinds of claims. Action items are
matched deterministically: an (owner, task) pair either matches a
ground-truth pair or it doesn't - an objective fact, decided in code.
Decisions and sentiment are judged by a second model call with a rationale,
since "was this actually a decision" and "is this meeting's tone neutral or
mixed" don't reduce cleanly to string comparison the way an (owner, task)
pair does.
"""

from __future__ import annotations

import json
from difflib import SequenceMatcher

from anthropic import Anthropic
from pydantic import BaseModel, Field

from meetingintel.llm_client import timed_forced_tool_call
from meetingintel.models import (
    ActionItem,
    ActionItemGroundTruth,
    Decision,
    EvalResult,
    MeetingSummary,
    PipelineConfig,
    SentimentScore,
    Transcript,
)


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def match_action_items(
    predicted: list[ActionItem],
    ground_truth: list[ActionItemGroundTruth],
    task_similarity_threshold: float = 0.5,
) -> EvalResult:
    """Greedy matching, not optimal bipartite matching: for each ground-truth
    item (in order), the best-scoring unmatched predicted item whose
    normalized owner matches exactly is claimed, provided its task-text
    similarity is >= threshold. Ties go to whichever predicted item appears
    earlier in `predicted` (strict '>' when comparing scores).
    """
    unmatched_pred = list(predicted)
    matched: list[tuple[str, str]] = []
    missed: list[ActionItemGroundTruth] = []

    for gt in ground_truth:
        best: ActionItem | None = None
        best_score = -1.0
        for p in unmatched_pred:
            if _normalize(p.owner or "") != _normalize(gt.owner):
                continue
            score = SequenceMatcher(None, _normalize(p.description), _normalize(gt.task)).ratio()
            if score > best_score:
                best, best_score = p, score
        if best is not None and best_score >= task_similarity_threshold:
            matched.append((gt.owner, gt.task))
            unmatched_pred.remove(best)
        else:
            missed.append(gt)

    precision = len(matched) / len(predicted) if predicted else 0.0
    recall = len(matched) / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvalResult(
        meeting_id="",
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        matched=matched,
        missed=missed,
        spurious=unmatched_pred,
    )


class JudgeScore(BaseModel):
    score: float = Field(ge=0, le=10)
    rationale: str


_DECISION_JUDGE_SYSTEM_PROMPT = (
    "You judge whether the predicted list of meeting decisions accurately "
    "and completely reflects what was actually decided in the transcript. "
    "Score 0-10, where 10 means the predicted decisions are complete and "
    "accurate, and 0 means they are missing or wrong. Give a brief rationale."
)

_SENTIMENT_JUDGE_SYSTEM_PROMPT = (
    "You judge whether the predicted overall sentiment accurately reflects "
    "the tone of the meeting transcript. Score 0-10, where 10 means the "
    "predicted sentiment label and rationale are an accurate read of the "
    "meeting's tone, and 0 means it's clearly wrong. Give a brief rationale."
)


def judge_decisions(
    transcript: Transcript,
    predicted: list[Decision],
    reference: list[str] | None,
    config: PipelineConfig,
    client: Anthropic | None = None,
) -> JudgeScore:
    client = client or Anthropic()
    payload = json.dumps(
        {
            "transcript": [u.model_dump() for u in transcript.utterances],
            "predicted_decisions": [d.model_dump() for d in predicted],
            "reference_decisions": reference,
        },
        default=str,
    )
    score, _meta = timed_forced_tool_call(
        client, config.reduce_model, _DECISION_JUDGE_SYSTEM_PROMPT, payload, JudgeScore, "record_judge_score", phase="reduce"
    )
    return score


def judge_sentiment(
    transcript: Transcript,
    predicted: SentimentScore,
    config: PipelineConfig,
    client: Anthropic | None = None,
) -> JudgeScore:
    client = client or Anthropic()
    payload = json.dumps(
        {
            "transcript": [u.model_dump() for u in transcript.utterances],
            "predicted_sentiment": predicted.model_dump(),
        },
        default=str,
    )
    score, _meta = timed_forced_tool_call(
        client, config.reduce_model, _SENTIMENT_JUDGE_SYSTEM_PROMPT, payload, JudgeScore, "record_judge_score", phase="reduce"
    )
    return score


def evaluate_meeting(
    transcript: Transcript,
    summary: MeetingSummary,
    ground_truth: list[ActionItemGroundTruth],
    decision_reference: list[str] | None,
    config: PipelineConfig,
    client: Anthropic | None = None,
    run_judge: bool = False,
) -> EvalResult:
    result = match_action_items(summary.action_items, ground_truth)
    result.meeting_id = transcript.meeting_id
    if run_judge:
        result.decision_judge_score = judge_decisions(
            transcript, summary.decisions, decision_reference, config, client=client
        ).score
        result.sentiment_judge_score = judge_sentiment(
            transcript, summary.overall_sentiment, config, client=client
        ).score
    return result
