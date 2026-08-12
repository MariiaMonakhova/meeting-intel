"""Extracting plain text from uploaded meeting files.

File-like object in, plain text out - parse_transcript() works on the
result exactly like it works on a .txt file, and never needs to know these
source formats exist. Everything operates on in-memory file-like objects,
no disk I/O, since Streamlit hands uploaded files over as BytesIO-like
objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf"}


def extract_text(file: BinaryIO, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return _extract_txt(file)
    if suffix == ".docx":
        return _extract_docx(file)
    if suffix == ".pdf":
        return _extract_pdf(file)
    raise ValueError(f"Unsupported file type: {suffix!r}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")


def _extract_txt(file: BinaryIO) -> str:
    data = file.read()
    return data.decode("utf-8") if isinstance(data, bytes) else data


def _extract_docx(file: BinaryIO) -> str:
    document = Document(file)
    return "\n".join(p.text for p in document.paragraphs)


def _extract_pdf(file: BinaryIO) -> str:
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)
