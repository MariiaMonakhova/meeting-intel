"""Transcript ingestion.

Parses the simple "Speaker: text" transcript format used throughout this
project, with an optional leading "[HH:MM:SS]" timestamp per line. This is a
deliberately simple format, not a general-purpose transcript parser: real
Zoom exports (.vtt) use timed cue ranges with the speaker on a separate line,
and Otter.ai exports are per-speaker-turn blocks with timestamps and
confidence scores. Adding parse_vtt() / parse_otter_export() functions that
produce the same Transcript shape is the natural extension point if those
formats are needed later; neither is implemented here.
"""

from __future__ import annotations

import re

from meetingintel.models import Transcript, Utterance

_LINE_RE = re.compile(
    r"^(?:\[(?P<ts>\d{1,2}:\d{2}:\d{2})\]\s*)?(?P<speaker>[^:\n]{1,80}):\s*(?P<text>.+)$"
)


def parse_transcript(
    raw_text: str, meeting_id: str, title: str, date: str | None = None
) -> Transcript:
    utterances: list[Utterance] = []
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if match is None:
            if utterances:
                utterances[-1].text = f"{utterances[-1].text} {line}".strip()
            continue
        utterances.append(
            Utterance(
                speaker=match["speaker"].strip(),
                text=match["text"].strip(),
                timestamp=match["ts"],
                index=len(utterances),
            )
        )
    attendees = sorted({u.speaker for u in utterances})
    return Transcript(
        meeting_id=meeting_id,
        title=title,
        date=date,
        attendees=attendees,
        utterances=utterances,
    )


def load_transcript_file(
    path: str, meeting_id: str, title: str, date: str | None = None
) -> Transcript:
    with open(path, encoding="utf-8") as f:
        raw_text = f.read()
    return parse_transcript(raw_text, meeting_id=meeting_id, title=title, date=date)
