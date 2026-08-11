"""Interactive CLI.

A thin REPL over the already-tested pipeline modules (ingest, extraction,
insights, eval, rag) - this file only orchestrates calls into already
unit-tested code and prints results, so there's no test_cli.py, matching
doc-pipeline-lab's approach.
"""

from __future__ import annotations

import json
from pathlib import Path

from anthropic import Anthropic
from rich.console import Console
from rich.table import Table

from meetingintel.config import ensure_api_key
from meetingintel.eval import evaluate_meeting
from meetingintel.extraction import run_extraction
from meetingintel.ingest import load_transcript_file
from meetingintel.models import ActionItemGroundTruth, EvalResult, MeetingSummary, PipelineConfig, Transcript
from meetingintel.rag.retriever import TfidfRetriever
from meetingintel.rag.store import build_indexed_meeting, load_all_indexed_meetings, save_indexed_meeting

console = Console()

MENU = [
    "Load transcript from file",
    "Set parameters (chunk_size, overlap, map/reduce model)",
    "Run extraction (map -> reduce)",
    "View meeting summary, action items, and decisions",
    "View attendee insights",
    "Run evaluation against ground truth",
    "Query cross-meeting RAG",
    "Quit",
]


class Session:
    def __init__(self) -> None:
        self.client = Anthropic()
        self.transcript: Transcript | None = None
        self.transcript_path: Path | None = None
        self.config = PipelineConfig()
        self.last_summary: MeetingSummary | None = None
        self.last_eval: EvalResult | None = None
        self.retriever = TfidfRetriever()
        self.retriever.index(load_all_indexed_meetings())


def _choose(prompt: str, options: list[str]) -> str:
    while True:
        console.print(prompt)
        for i, opt in enumerate(options, start=1):
            console.print(f"  {i}. {opt}")
        raw = input("choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        console.print("[red]Invalid choice, try again.[/red]")


def _ask_int(label: str, current: int) -> int:
    raw = input(f"{label} [{current}]: ").strip()
    if not raw:
        return current
    try:
        return int(raw)
    except ValueError:
        console.print("[red]Not a number, keeping current value.[/red]")
        return current


def _load_transcript(session: Session) -> None:
    path = Path(input("Path to transcript file: ").strip())
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return
    meeting_id = input(f"Meeting id [{path.stem}]: ").strip() or path.stem
    title = input(f"Title [{path.stem}]: ").strip() or path.stem
    date = input("Date (YYYY-MM-DD, optional): ").strip() or None
    session.transcript = load_transcript_file(str(path), meeting_id=meeting_id, title=title, date=date)
    session.transcript_path = path
    session.last_summary = None
    session.last_eval = None
    console.print(
        f"[green]Loaded {len(session.transcript.utterances)} utterances, "
        f"{len(session.transcript.attendees)} attendees: {', '.join(session.transcript.attendees)}[/green]"
    )


def _set_params(session: Session) -> None:
    c = session.config
    chunk_size = _ask_int("chunk_size (tokens)", c.chunk_size)
    overlap = _ask_int("overlap (tokens)", c.overlap)
    if overlap >= chunk_size:
        console.print("[red]overlap must be smaller than chunk_size - keeping previous values.[/red]")
        return
    map_model = input(f"map_model [{c.map_model}]: ").strip() or c.map_model
    reduce_model = input(f"reduce_model [{c.reduce_model}]: ").strip() or c.reduce_model
    session.config = PipelineConfig(chunk_size=chunk_size, overlap=overlap, map_model=map_model, reduce_model=reduce_model)
    console.print("[green]Parameters updated.[/green]")


def _run_extraction(session: Session) -> None:
    if session.transcript is None:
        console.print("[red]Load a transcript first (menu option 1).[/red]")
        return
    console.print("Running extraction - this calls the real Claude API...")
    summary = run_extraction(session.transcript, session.config, client=session.client)
    session.last_summary = summary
    console.print(
        f"[green]Done.[/green] {summary.api_calls_map} map calls, {summary.api_calls_reduce} reduce calls, "
        f"cost ${summary.cost_usd:.4f}, {summary.wall_clock_s:.1f}s wall clock."
    )
    date_str = str(session.transcript.date) if session.transcript.date else None
    indexed = build_indexed_meeting(summary, title=session.transcript.title, date=date_str)
    save_indexed_meeting(indexed)
    session.retriever.index(load_all_indexed_meetings())


def _view_summary(session: Session) -> None:
    if session.last_summary is None:
        console.print("[red]Run extraction first (menu option 3).[/red]")
        return
    s = session.last_summary
    console.print(f"\n[bold]Executive summary[/bold]\n{s.executive_summary}\n")

    action_table = Table(title="Action items")
    action_table.add_column("Owner")
    action_table.add_column("Description")
    action_table.add_column("Due")
    for a in s.action_items:
        action_table.add_row(a.owner or "-", a.description, a.due_date or "-")
    console.print(action_table)

    decision_table = Table(title="Decisions")
    decision_table.add_column("Decided by")
    decision_table.add_column("Description")
    for d in s.decisions:
        decision_table.add_row(d.decided_by or "-", d.description)
    console.print(decision_table)

    console.print(f"Overall sentiment: {s.overall_sentiment.label} ({s.overall_sentiment.score:+.2f})")


def _view_attendee_insights(session: Session) -> None:
    if session.last_summary is None:
        console.print("[red]Run extraction first (menu option 3).[/red]")
        return
    table = Table(title="Attendee insights")
    table.add_column("Name")
    table.add_column("Action items")
    table.add_column("Sentiment summary")
    table.add_column("Notable quotes")
    for insight in session.last_summary.attendee_insights:
        table.add_row(
            insight.name, str(insight.action_item_count), insight.sentiment_summary, "; ".join(insight.notable_quotes)
        )
    console.print(table)


def _run_eval(session: Session) -> None:
    if session.transcript is None or session.last_summary is None:
        console.print("[red]Load a transcript and run extraction first.[/red]")
        return
    if session.transcript_path is None:
        console.print("[red]No transcript file path on record - can't locate ground truth.[/red]")
        return

    gt_path = session.transcript_path.parent / f"{session.transcript_path.stem}.actions.json"
    if not gt_path.exists():
        console.print(f"[red]No ground truth file found at {gt_path}.[/red]")
        return
    ground_truth = [ActionItemGroundTruth(**item) for item in json.loads(gt_path.read_text())]

    decisions_path = session.transcript_path.parent / f"{session.transcript_path.stem}.decisions.json"
    decision_reference = json.loads(decisions_path.read_text()) if decisions_path.exists() else None

    result = evaluate_meeting(
        session.transcript,
        session.last_summary,
        ground_truth,
        decision_reference,
        session.config,
        client=session.client,
        run_judge=True,
    )
    session.last_eval = result
    console.print(
        f"[green]precision={result.precision} recall={result.recall} f1={result.f1}[/green]\n"
        f"matched={result.matched}\n"
        f"missed={[m.task for m in result.missed]}\n"
        f"spurious={[a.description for a in result.spurious]}\n"
        f"decision_judge_score={result.decision_judge_score} sentiment_judge_score={result.sentiment_judge_score}"
    )


def _query_rag(session: Session) -> None:
    query = input("Query: ").strip()
    if not query:
        return
    results = session.retriever.query(query, top_k=5)
    if not results:
        console.print("[yellow]No indexed meetings yet - run extraction on at least one meeting first.[/yellow]")
        return
    table = Table(title=f"Results for: {query}")
    table.add_column("Meeting")
    table.add_column("Score")
    table.add_column("Snippet")
    for r in results:
        table.add_row(r.title, f"{r.score:.3f}", r.snippet)
    console.print(table)


_ACTIONS = {
    "Load transcript from file": _load_transcript,
    "Set parameters (chunk_size, overlap, map/reduce model)": _set_params,
    "Run extraction (map -> reduce)": _run_extraction,
    "View meeting summary, action items, and decisions": _view_summary,
    "View attendee insights": _view_attendee_insights,
    "Run evaluation against ground truth": _run_eval,
    "Query cross-meeting RAG": _query_rag,
}


def main() -> None:
    ensure_api_key()
    session = Session()
    console.print("[bold]meeting-intel[/bold] - AI-powered meeting intelligence\n")
    while True:
        try:
            choice = _choose("What would you like to do?", MENU)
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return
        if choice == "Quit":
            console.print("Bye.")
            return
        _ACTIONS[choice](session)
        console.print()


if __name__ == "__main__":
    main()
