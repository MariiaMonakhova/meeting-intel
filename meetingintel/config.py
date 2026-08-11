"""CLI startup configuration.

Loads .env and validates the API key before any pipeline code gets a chance
to construct an Anthropic client, so a missing key fails fast with a clear
message instead of surfacing as an opaque error deep in extraction.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv


def ensure_api_key() -> None:
    load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
