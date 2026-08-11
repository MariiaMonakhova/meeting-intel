"""Local JSON persistence for indexed meetings.

One meeting_store/<meeting_id>.json file per meeting, overwritten on
re-index so re-running the pipeline on the same meeting doesn't create
duplicates in search. No vector DB needed at this scale; a real datastore is
future work if durability/scale beyond a local folder is ever needed.
"""

from __future__ import annotations

from pathlib import Path

from meetingintel.models import MeetingSummary
from meetingintel.rag.retriever import IndexedMeeting

DEFAULT_STORE_DIR = Path("meeting_store")


def build_indexed_meeting(summary: MeetingSummary, title: str, date: str | None = None) -> IndexedMeeting:
    parts = [summary.executive_summary]
    parts += [f"Action: {a.description} (owner: {a.owner})" for a in summary.action_items]
    parts += [f"Decision: {d.description}" for d in summary.decisions]
    return IndexedMeeting(meeting_id=summary.meeting_id, title=title, date=date, text="\n".join(parts))


def save_indexed_meeting(meeting: IndexedMeeting, store_dir: Path = DEFAULT_STORE_DIR) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    path = store_dir / f"{meeting.meeting_id}.json"
    path.write_text(meeting.model_dump_json(indent=2))
    return path


def load_all_indexed_meetings(store_dir: Path = DEFAULT_STORE_DIR) -> list[IndexedMeeting]:
    if not store_dir.exists():
        return []
    return [IndexedMeeting.model_validate_json(path.read_text()) for path in sorted(store_dir.glob("*.json"))]
