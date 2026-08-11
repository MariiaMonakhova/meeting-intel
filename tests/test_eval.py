from difflib import SequenceMatcher
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from meetingintel.eval import evaluate_meeting, judge_decisions, judge_sentiment, match_action_items
from meetingintel.models import (
    ActionItem,
    ActionItemGroundTruth,
    Decision,
    MeetingSummary,
    PipelineConfig,
    SentimentScore,
    Transcript,
    Utterance,
)


def _pred(owner: str, description: str) -> ActionItem:
    return ActionItem(description=description, owner=owner, source_chunk_id="reduce")


def _gt(owner: str, task: str) -> ActionItemGroundTruth:
    return ActionItemGroundTruth(owner=owner, task=task)


def test_exact_match_found():
    result = match_action_items([_pred("Alice", "send the docs")], [_gt("Alice", "send the docs")])
    assert result.matched == [("Alice", "send the docs")]
    assert result.missed == []
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_owner_mismatch_not_found_even_if_task_matches():
    result = match_action_items([_pred("Bob", "send the docs")], [_gt("Alice", "send the docs")])
    assert result.matched == []
    assert result.missed == [_gt("Alice", "send the docs")]


def test_paraphrase_above_threshold_matches():
    result = match_action_items(
        [_pred("Alice", "share the API documentation with the team")],
        [_gt("Alice", "send the API docs to the team")],
        task_similarity_threshold=0.5,
    )
    assert len(result.matched) == 1


def test_paraphrase_below_threshold_does_not_match():
    result = match_action_items(
        [_pred("Alice", "completely unrelated task about lunch orders")],
        [_gt("Alice", "send the API docs to the team")],
        task_similarity_threshold=0.5,
    )
    assert result.matched == []
    assert len(result.missed) == 1


def test_boundary_score_exactly_at_threshold_counts_as_match():
    pred_text, gt_text = "send the docs", "send the doc"
    ratio = SequenceMatcher(None, pred_text, gt_text).ratio()
    result = match_action_items(
        [_pred("Alice", pred_text)], [_gt("Alice", gt_text)], task_similarity_threshold=ratio
    )
    assert len(result.matched) == 1


def test_empty_predicted_returns_zero_precision_no_division_error():
    result = match_action_items([], [_gt("Alice", "send the docs")])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.missed == [_gt("Alice", "send the docs")]


def test_empty_ground_truth_returns_zero_recall_no_division_error():
    result = match_action_items([_pred("Alice", "send the docs")], [])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.spurious == [_pred("Alice", "send the docs")]


def test_both_empty_returns_zero_f1_not_nan():
    result = match_action_items([], [])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_spurious_contains_unmatched_predicted_items():
    result = match_action_items(
        [_pred("Alice", "send the docs"), _pred("Bob", "unrelated task")],
        [_gt("Alice", "send the docs")],
    )
    assert result.spurious == [_pred("Bob", "unrelated task")]


def test_greedy_matching_first_predicted_wins_tie():
    p1 = _pred("Alice", "send the docs")
    p2 = _pred("Alice", "send the docs")
    result = match_action_items([p1, p2], [_gt("Alice", "send the docs")])
    assert len(result.matched) == 1
    assert result.spurious == [p2]


def _transcript() -> Transcript:
    return Transcript(
        meeting_id="m1",
        title="Test",
        attendees=["Alice"],
        utterances=[Utterance(speaker="Alice", text="Let's ship this.", index=0)],
    )


def _judge_response(score=8.0, rationale="looks accurate"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={"score": score, "rationale": rationale})],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=80, output_tokens=20),
    )


def test_judge_decisions_returns_score():
    client = MagicMock()
    client.messages.create.return_value = _judge_response(score=7.5)
    score = judge_decisions(_transcript(), [], None, PipelineConfig(), client=client)
    assert score.score == 7.5
    assert score.rationale


def test_judge_sentiment_returns_score():
    client = MagicMock()
    client.messages.create.return_value = _judge_response(score=9.0)
    score = judge_sentiment(_transcript(), SentimentScore(label="positive", score=0.5), PipelineConfig(), client=client)
    assert score.score == 9.0


def _summary() -> MeetingSummary:
    return MeetingSummary(
        meeting_id="m1",
        executive_summary="Shipped the roadmap.",
        action_items=[_pred("Alice", "send the docs")],
        decisions=[Decision(description="ship next week", decided_by="Alice", source_chunk_id="reduce")],
        attendee_insights=[],
        overall_sentiment=SentimentScore(label="positive", score=0.5),
        cost_usd=0.01,
        api_calls_map=1,
        api_calls_reduce=2,
        wall_clock_s=1.0,
    )


def test_evaluate_meeting_default_makes_no_judge_calls():
    client = MagicMock()
    result = evaluate_meeting(_transcript(), _summary(), [_gt("Alice", "send the docs")], None, PipelineConfig(), client=client)
    assert result.meeting_id == "m1"
    assert result.decision_judge_score is None
    assert result.sentiment_judge_score is None
    assert client.messages.create.call_count == 0


def test_evaluate_meeting_with_judge_populates_both_scores():
    client = MagicMock()
    client.messages.create.return_value = _judge_response(score=6.0)
    result = evaluate_meeting(
        _transcript(), _summary(), [_gt("Alice", "send the docs")], None, PipelineConfig(), client=client, run_judge=True
    )
    assert result.decision_judge_score == 6.0
    assert result.sentiment_judge_score == 6.0
    assert client.messages.create.call_count == 2
