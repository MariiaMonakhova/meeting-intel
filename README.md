# meeting-intel: AI-Powered Meeting Intelligence

A portfolio project: turns meeting transcripts into structured, evaluated
intelligence using the real Claude API. Speaker-aware MapReduce chunking,
structured extraction (action items, decisions, sentiment) via forced
tool-use, cross-meeting RAG (local TF-IDF), and an evaluation harness that
scores extraction quality against human-labeled ground truth.

Sibling project to [doc-pipeline-lab](../doc-pipeline-lab), which explored
the same MapReduce/chunking ideas against a deterministic emulator with no
API cost. This project applies the same discipline to a new domain (meeting
transcripts) and a new set of concerns: real API calls, structured outputs,
RAG, and evaluation.

**Running the CLI makes real, billed calls to the Claude API.** Nothing
here is emulated - every extraction, merge, insight, evaluation judge, and
quality score costs real money (a few cents per meeting, tracked and
printed after each run).

## What's inside

- Speaker-aware semantic chunking: packs whole utterances (never split
  mid-speaker) into token-budgeted chunks with trailing overlap
- MapReduce extraction: parallel per-chunk extraction (Claude Haiku) →
  single structured merge/dedupe call (Claude Sonnet)
- Structured outputs via forced tool-use: every LLM call returns
  Pydantic-validated JSON, not free-text parsing
- Per-attendee insights: action-item counts computed in plain Python (free),
  sentiment/quotes from one additional model call
- Cross-meeting RAG: local TF-IDF retrieval over past meetings, no extra
  API key or cost
- Evaluation harness: deterministic precision/recall/F1 for action items
  (owner + task matching), LLM-judge scores for decisions and sentiment
- Bonus: an opt-in meeting-quality scorer (engagement, decision clarity,
  follow-up rate)
- An interactive CLI

## Installation

```bash
git clone <this-repo>
cd meeting-intel
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env
# edit .env and add your ANTHROPIC_API_KEY
```

## Tests

```bash
.venv/bin/python -m pytest -v
```

All LLM calls are mocked in the default test run - no API key needed and no
cost incurred. A handful of real integration tests exist for the API
wrapper itself, gated behind `ANTHROPIC_API_KEY` being set, and are skipped
otherwise.

## Interactive CLI

```bash
.venv/bin/python -m meetingintel.cli
# or, after install:
.venv/bin/meeting-intel
```

Menu:
1. Load a transcript from a file
2. Set parameters (chunk_size, overlap, map/reduce model)
3. Run extraction (map → reduce) - **this is the step that spends money**
4. View meeting summary, action items, and decisions
5. View attendee insights
6. Run evaluation against ground truth
7. Query cross-meeting RAG
8. Run meeting-quality scorer (bonus, also spends money)
9. Quit

Two sample transcripts are included in `sample_meetings/` (with ground
truth for evaluation) - a product standup and an incident postmortem, in
different topics and tones so cross-meeting RAG has something real to
distinguish between.

## Quick run without the CLI

```bash
.venv/bin/python -c "
from meetingintel.ingest import load_transcript_file
from meetingintel.extraction import run_extraction
from meetingintel.models import PipelineConfig

transcript = load_transcript_file(
    'sample_meetings/product_standup.txt',
    meeting_id='product_standup', title='Product Standup', date='2026-08-04',
)
summary = run_extraction(transcript, PipelineConfig())
print(summary.executive_summary)
print(f'cost: \${summary.cost_usd:.4f}')
"
```

## Transcript format

Plain text, one utterance per line: `Speaker: text`, with an optional
leading `[HH:MM:SS]` timestamp. This is a deliberately simple format, not a
general-purpose transcript parser - real Zoom `.vtt` exports and Otter.ai
exports use different structures; adding parsers for those is a natural
extension point (`parse_vtt()` / `parse_otter_export()`), not built here.

## Project structure

```
meetingintel/
  models.py       — Pydantic models shared across the pipeline
  tokens.py        — token-count heuristic, for chunk-size budgeting only
  ingest.py         — "Speaker: text" transcript parsing
  chunking.py        — speaker-aware semantic chunking
  llm_client.py        — forced tool-use wrapper: structured output + cost/latency accounting
  extraction.py          — MapReduce: map (per-chunk) and reduce (merge/dedupe) phases
  insights.py              — per-attendee insights (counts in Python, sentiment via LLM)
  eval.py                    — deterministic action-item matching + LLM-judge scores
  quality.py                   — bonus meeting-quality scorer
  rag/
    retriever.py                — Retriever protocol + TfidfRetriever
    store.py                     — local JSON persistence for indexed meetings
  config.py                       — .env loading and API key validation
  cli.py                            — interactive REPL
sample_meetings/                     — sample transcripts + ground truth for evaluation
tests/                                 — pytest tests, LLM calls mocked by default
```
