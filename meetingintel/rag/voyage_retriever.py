"""Semantic retrieval via Voyage AI embeddings - the real-embeddings
alternative to TfidfRetriever, behind the same Retriever protocol.

Calls Voyage's REST API directly via httpx (already a transitive dependency
via anthropic) rather than the official voyageai SDK, which pulls in a heavy
dependency tree (pillow, huggingface_hub, tokenizers, aiohttp,
langchain-text-splitters) for multimodal features this project doesn't use.
"""

from __future__ import annotations

import os

import httpx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from meetingintel.rag.retriever import IndexedMeeting, RetrievalResult, Retriever, TfidfRetriever

VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"
DEFAULT_MODEL = "voyage-3.5"


def select_retriever() -> Retriever:
    """VoyageRetriever if VOYAGE_API_KEY is set, otherwise the free TF-IDF
    baseline - so the app keeps working for anyone without a Voyage key."""
    if os.environ.get("VOYAGE_API_KEY"):
        return VoyageRetriever()
    return TfidfRetriever()


class VoyageRetriever:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "VoyageRetriever needs an API key - pass api_key= or set VOYAGE_API_KEY."
            )
        self._api_key = resolved_key
        self._model = model
        self._client = client or httpx.Client()
        self._docs: list[IndexedMeeting] = []

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        resp = self._client.post(
            VOYAGE_EMBEDDINGS_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model, "input_type": input_type},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

    def index(self, documents: list[IndexedMeeting]) -> None:
        self._docs = documents
        missing = [d for d in documents if d.embedding is None]
        if not missing:
            return
        embeddings = self._embed([d.text for d in missing], input_type="document")
        for doc, embedding in zip(missing, embeddings):
            doc.embedding = embedding

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        embedded_docs = [d for d in self._docs if d.embedding is not None]
        if not embedded_docs:
            return []
        query_embedding = self._embed([text], input_type="query")[0]
        matrix = np.array([d.embedding for d in embedded_docs])
        scores = cosine_similarity([query_embedding], matrix)[0]
        ranked = sorted(zip(embedded_docs, scores), key=lambda pair: -pair[1])[:top_k]
        return [
            RetrievalResult(meeting_id=doc.meeting_id, title=doc.title, score=float(score), snippet=doc.text[:200])
            for doc, score in ranked
        ]
