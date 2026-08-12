import io
from unittest.mock import MagicMock, patch

import pytest

from meetingintel.file_extract import SUPPORTED_EXTENSIONS, extract_text


def test_extract_txt_from_bytes():
    file = io.BytesIO("Alice: hello there".encode("utf-8"))
    assert extract_text(file, "meeting.txt") == "Alice: hello there"


def test_extract_txt_uppercase_extension():
    file = io.BytesIO(b"Alice: hi")
    assert extract_text(file, "meeting.TXT") == "Alice: hi"


def test_extract_txt_invalid_utf8_raises():
    file = io.BytesIO(b"\xff\xfe not valid utf-8")
    with pytest.raises(UnicodeDecodeError):
        extract_text(file, "meeting.txt")


def test_extract_txt_empty_file_returns_empty_string():
    file = io.BytesIO(b"")
    assert extract_text(file, "meeting.txt") == ""


def test_unsupported_extension_raises_value_error():
    file = io.BytesIO(b"whatever")
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text(file, "meeting.pptx")


def test_unsupported_extension_message_lists_supported_formats():
    file = io.BytesIO(b"whatever")
    with pytest.raises(ValueError, match=r"\.docx.*\.pdf.*\.txt|\.txt.*\.docx.*\.pdf"):
        extract_text(file, "meeting.doc")


def _docx_bytes(paragraphs: list[str]) -> io.BytesIO:
    from docx import Document

    document = Document()
    for p in paragraphs:
        document.add_paragraph(p)
    buf = io.BytesIO()
    document.save(buf)
    buf.seek(0)
    return buf


def test_extract_docx_joins_paragraphs():
    file = _docx_bytes(["Alice: hello there", "Bob: hi Alice"])
    assert extract_text(file, "meeting.docx") == "Alice: hello there\nBob: hi Alice"


def test_extract_docx_empty_document_returns_empty_string():
    file = _docx_bytes([])
    assert extract_text(file, "meeting.docx") == ""


def _fake_pdf_reader(page_texts: list[str | None]) -> MagicMock:
    pages = []
    for text in page_texts:
        page = MagicMock()
        page.extract_text.return_value = text
        pages.append(page)
    reader = MagicMock()
    reader.pages = pages
    return reader


def test_extract_pdf_joins_page_text():
    with patch("meetingintel.file_extract.PdfReader", return_value=_fake_pdf_reader(["page one", "page two"])):
        text = extract_text(io.BytesIO(b"%PDF-fake"), "meeting.pdf")
    assert text == "page one\npage two"


def test_extract_pdf_none_page_text_becomes_empty_string_not_a_crash():
    with patch("meetingintel.file_extract.PdfReader", return_value=_fake_pdf_reader(["page one", None])):
        text = extract_text(io.BytesIO(b"%PDF-fake"), "meeting.pdf")
    assert text == "page one\n"


def test_extract_pdf_no_pages_returns_empty_string():
    with patch("meetingintel.file_extract.PdfReader", return_value=_fake_pdf_reader([])):
        text = extract_text(io.BytesIO(b"%PDF-fake"), "meeting.pdf")
    assert text == ""


def test_supported_extensions_contains_expected_formats():
    assert SUPPORTED_EXTENSIONS == {".txt", ".docx", ".pdf"}
