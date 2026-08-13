"""Cross-meeting retrieval.

Retriever is a narrow protocol so the scoring backend can be swapped
without touching callers. TfidfRetriever is the lexical baseline: cosine
similarity over TF-IDF vectors, no extra API key or cost. VoyageRetriever
(voyage_retriever.py) is the semantic alternative, behind the same
protocol - this is the whole reason the protocol exists.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class IndexedMeeting(BaseModel):
    meeting_id: str
    title: str
    date: str | None = None
    text: str  # flattened executive_summary + action items + decisions
    embedding: list[float] | None = None  # cached, set by VoyageRetriever.index()


class RetrievalResult(BaseModel):
    meeting_id: str
    title: str
    score: float
    snippet: str


class Retriever(Protocol):
    def index(self, documents: list[IndexedMeeting]) -> None: ...
    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]: ...


class TfidfRetriever:
    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._docs: list[IndexedMeeting] = []

    def index(self, documents: list[IndexedMeeting]) -> None:
        self._docs = documents
        if not documents:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform([d.text for d in documents])

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self._docs or self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix)[0]
        ranked = sorted(zip(self._docs, scores), key=lambda pair: -pair[1])[:top_k]
        return [
            RetrievalResult(meeting_id=doc.meeting_id, title=doc.title, score=float(score), snippet=doc.text[:200])
            for doc, score in ranked
        ]
