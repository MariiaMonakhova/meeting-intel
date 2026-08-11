from pathlib import Path

from meetingintel.models import ActionItem, Decision, MeetingSummary, SentimentScore
from meetingintel.rag.retriever import IndexedMeeting, TfidfRetriever
from meetingintel.rag.store import build_indexed_meeting, load_all_indexed_meetings, save_indexed_meeting


def _doc(meeting_id: str, title: str, text: str) -> IndexedMeeting:
    return IndexedMeeting(meeting_id=meeting_id, title=title, text=text)


def test_query_empty_index_returns_empty_list():
    r = TfidfRetriever()
    r.index([])
    assert r.query("anything") == []


def test_query_before_indexing_returns_empty_list():
    r = TfidfRetriever()
    assert r.query("anything") == []


def test_single_document_index_returns_one_result():
    r = TfidfRetriever()
    r.index([_doc("m1", "Standup", "we discussed the roadmap")])
    results = r.query("roadmap")
    assert len(results) == 1
    assert results[0].meeting_id == "m1"


def test_results_ordered_by_descending_score():
    r = TfidfRetriever()
    r.index([
        _doc("m1", "Incident review", "the database outage caused downtime and paging alerts"),
        _doc("m2", "Product standup", "we reviewed the roadmap and shipped the new onboarding flow"),
        _doc("m3", "Budget planning", "quarterly budget allocation and headcount planning"),
    ])
    results = r.query("outage downtime paging")
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_distinctive_keyword_query_ranks_matching_document_first():
    r = TfidfRetriever()
    r.index([
        _doc("incident", "Incident review", "the database outage caused downtime and paging alerts for the oncall team"),
        _doc("standup", "Product standup", "we reviewed the product roadmap and shipped the new onboarding flow"),
    ])
    results = r.query("database outage paging alerts")
    assert results[0].meeting_id == "incident"


def test_top_k_limits_result_count():
    r = TfidfRetriever()
    r.index([_doc(f"m{i}", f"Meeting {i}", f"topic number {i} discussion") for i in range(10)])
    results = r.query("topic discussion", top_k=3)
    assert len(results) == 3


def test_build_indexed_meeting_flattens_summary():
    summary = MeetingSummary(
        meeting_id="m1",
        executive_summary="The team agreed to ship next week.",
        action_items=[ActionItem(description="send the docs", owner="Alice", source_chunk_id="reduce")],
        decisions=[Decision(description="ship next week", decided_by="Bob", source_chunk_id="reduce")],
        attendee_insights=[],
        overall_sentiment=SentimentScore(label="positive", score=0.5),
        cost_usd=0.01,
        api_calls_map=2,
        api_calls_reduce=2,
        wall_clock_s=1.0,
    )
    indexed = build_indexed_meeting(summary, title="Weekly Sync", date="2026-08-11")
    assert indexed.meeting_id == "m1"
    assert indexed.title == "Weekly Sync"
    assert "ship next week" in indexed.text
    assert "send the docs" in indexed.text


def test_save_and_load_round_trip(tmp_path: Path):
    store_dir = tmp_path / "meeting_store"
    meeting = _doc("m1", "Standup", "we discussed the roadmap")
    save_indexed_meeting(meeting, store_dir=store_dir)

    loaded = load_all_indexed_meetings(store_dir=store_dir)
    assert len(loaded) == 1
    assert loaded[0] == meeting


def test_save_overwrites_existing_file_for_same_meeting_id(tmp_path: Path):
    store_dir = tmp_path / "meeting_store"
    save_indexed_meeting(_doc("m1", "Standup v1", "first version"), store_dir=store_dir)
    save_indexed_meeting(_doc("m1", "Standup v2", "second version"), store_dir=store_dir)

    loaded = load_all_indexed_meetings(store_dir=store_dir)
    assert len(loaded) == 1
    assert loaded[0].title == "Standup v2"


def test_load_all_from_nonexistent_directory_returns_empty(tmp_path: Path):
    assert load_all_indexed_meetings(store_dir=tmp_path / "does_not_exist") == []
