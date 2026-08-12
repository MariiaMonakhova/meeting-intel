"""Streamlit UI.

One screen: upload a meeting file, run the MapReduce extraction pipeline,
show the reduce-phase result. A thin presentation layer over already-tested
modules (config, file_extract, ingest, extraction) - no pipeline logic
lives here, same principle as cli.py.

Run with: streamlit run meetingintel/webapp.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from meetingintel.config import ensure_api_key
from meetingintel.extraction import run_extraction
from meetingintel.file_extract import SUPPORTED_EXTENSIONS, extract_text
from meetingintel.ingest import parse_transcript
from meetingintel.models import PipelineConfig

st.set_page_config(page_title="meeting-intel", page_icon="\U0001f5d2️")

try:
    ensure_api_key()
except SystemExit as e:
    st.error(str(e))
    st.stop()

st.title("meeting-intel")
st.caption(
    "Upload a meeting transcript and get a MapReduce-extracted summary. "
    "This calls the real Claude API - each run costs real money."
)

uploaded = st.file_uploader(
    "Meeting file",
    type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
)

if uploaded is not None:
    default_id = Path(uploaded.name).stem
    meeting_id = st.text_input("Meeting id", value=default_id)
    title = st.text_input("Title", value=default_id)

    if st.button("Run extraction", type="primary"):
        try:
            raw_text = extract_text(uploaded, uploaded.name)
        except (ValueError, UnicodeDecodeError) as e:
            st.error(f"Couldn't read this file: {e}")
            st.stop()

        transcript = parse_transcript(raw_text, meeting_id=meeting_id, title=title)
        if not transcript.utterances:
            st.warning(
                "No 'Speaker: text' lines were recognized in this file. "
                "Check that the extracted text follows that format."
            )
            st.stop()
        st.success(
            f"Parsed {len(transcript.utterances)} utterances, "
            f"{len(transcript.attendees)} attendees: {', '.join(transcript.attendees)}"
        )

        with st.spinner("Running extraction (calling Claude)..."):
            summary = run_extraction(transcript, PipelineConfig())

        st.subheader("Executive summary")
        st.write(summary.executive_summary)

        st.subheader("Action items")
        st.table(
            [
                {"Owner": a.owner or "-", "Description": a.description, "Due": a.due_date or "-"}
                for a in summary.action_items
            ]
        )

        st.subheader("Decisions")
        st.table(
            [
                {"Decided by": d.decided_by or "-", "Description": d.description}
                for d in summary.decisions
            ]
        )

        st.subheader("Sentiment")
        st.write(f"{summary.overall_sentiment.label} ({summary.overall_sentiment.score:+.2f})")

        st.caption(
            f"Cost: ${summary.cost_usd:.4f} · {summary.api_calls_map} map calls, "
            f"{summary.api_calls_reduce} reduce calls · {summary.wall_clock_s:.1f}s"
        )
