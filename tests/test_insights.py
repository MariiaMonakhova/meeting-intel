from types import SimpleNamespace
from unittest.mock import MagicMock

from meetingintel.insights import _tally_action_items, build_attendee_insights
from meetingintel.models import ActionItem, PipelineConfig, Transcript, Utterance


def _action_item(owner: str | None, description: str = "do something") -> ActionItem:
    return ActionItem(description=description, owner=owner, source_chunk_id="reduce")


def test_tally_action_items_counts_by_owner():
    items = [_action_item("Alice"), _action_item("Bob"), _action_item("Alice")]
    counts = _tally_action_items(items)
    assert counts == {"Alice": 2, "Bob": 1}


def test_tally_action_items_ignores_items_without_owner():
    items = [_action_item(None), _action_item("Alice")]
    counts = _tally_action_items(items)
    assert counts == {"Alice": 1}
    assert None not in counts


def test_tally_action_items_empty_list_returns_empty_counter():
    assert _tally_action_items([]) == {}


def test_build_attendee_insights_empty_attendees_returns_empty_without_api_call():
    client = MagicMock()
    transcript = Transcript(meeting_id="m1", title="Test", attendees=[], utterances=[])
    insights, meta = build_attendee_insights(transcript, [], PipelineConfig(), client=client)
    assert insights == []
    assert meta["cost_usd"] == 0.0
    assert client.messages.create.call_count == 0


def _transcript_with_two_speakers() -> Transcript:
    utterances = [
        Utterance(speaker="Alice", text="Let's ship this by Friday.", index=0),
        Utterance(speaker="Bob", text="Sounds good to me.", index=1),
    ]
    return Transcript(meeting_id="m1", title="Test", attendees=["Alice", "Bob"], utterances=utterances)


def test_build_attendee_insights_merges_llm_response_with_action_item_counts():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={
            "insights": [
                {"name": "Alice", "sentiment_summary": "focused and decisive", "notable_quotes": ["Let's ship this by Friday."]},
                {"name": "Bob", "sentiment_summary": "supportive", "notable_quotes": []},
            ],
        })],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=120, output_tokens=40),
    )
    action_items = [_action_item("Alice"), _action_item("Alice"), _action_item("Bob")]
    insights, meta = build_attendee_insights(
        _transcript_with_two_speakers(), action_items, PipelineConfig(), client=client
    )

    by_name = {i.name: i for i in insights}
    assert by_name["Alice"].action_item_count == 2
    assert by_name["Alice"].sentiment_summary == "focused and decisive"
    assert by_name["Bob"].action_item_count == 1
    assert meta["cost_usd"] > 0


def test_build_attendee_insights_attendee_with_no_action_items_gets_zero_count():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={
            "insights": [
                {"name": "Alice", "sentiment_summary": "quiet", "notable_quotes": []},
                {"name": "Bob", "sentiment_summary": "quiet", "notable_quotes": []},
            ],
        })],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=100, output_tokens=30),
    )
    insights, _meta = build_attendee_insights(
        _transcript_with_two_speakers(), [], PipelineConfig(), client=client
    )
    assert all(i.action_item_count == 0 for i in insights)


def test_build_attendee_insights_name_mismatch_yields_zero_not_a_crash():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input={
            "insights": [{"name": "alice", "sentiment_summary": "focused", "notable_quotes": []}],
        })],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=100, output_tokens=30),
    )
    # owner is "Alice" (capitalized) but the model returned "alice" - documented
    # limitation: exact-match only, so this silently counts as 0 rather than crashing.
    insights, _meta = build_attendee_insights(
        _transcript_with_two_speakers(), [_action_item("Alice")], PipelineConfig(), client=client
    )
    assert insights[0].action_item_count == 0
