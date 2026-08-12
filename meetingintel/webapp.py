"""Streamlit UI.

One screen: upload a meeting file, run the MapReduce extraction pipeline,
show the reduce-phase result. A thin presentation layer over already-tested
modules (file_extract, ingest, extraction) - no pipeline logic lives here,
same principle as cli.py.

Bring-your-own-key: each visitor supplies their own ANTHROPIC_API_KEY in the
UI, so a public deployment never spends the app owner's budget. The key is
never written to disk or logged - it only lives in this browser session's
memory, passed straight into the Anthropic client for that session's calls.

Run with: streamlit run meetingintel/webapp.py
"""

from __future__ import annotations

import os
from pathlib import Path

import anthropic
import streamlit as st
from anthropic import Anthropic

from meetingintel.extraction import run_extraction
from meetingintel.file_extract import SUPPORTED_EXTENSIONS, extract_text
from meetingintel.ingest import parse_transcript
from meetingintel.models import PipelineConfig

st.set_page_config(page_title="meeting-intel", page_icon="\U0001f5d2️")

st.title("meeting-intel")
st.caption(
    "Upload a meeting transcript and get a MapReduce-extracted summary. "
    "This calls the real Claude API using YOUR key below - each run costs "
    "real money on your Anthropic account, not the app owner's."
)

api_key = st.text_input(
    "Your Anthropic API key",
    type="password",
    value=os.environ.get("ANTHROPIC_API_KEY", ""),
    help="Never stored or logged - used only for this browser session's requests.",
)

if not api_key.strip():
    st.info("Enter your Anthropic API key above to get started.")
    st.stop()

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

        client = Anthropic(api_key=api_key)
        with st.spinner("Running extraction (calling Claude)..."):
            try:
                summary = run_extraction(transcript, PipelineConfig(), client=client)
            except anthropic.APIError as e:
                st.error(f"Claude API error - check that your key is valid and has credits: {e}")
                st.stop()

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
