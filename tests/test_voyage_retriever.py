import httpx
import pytest
from unittest.mock import MagicMock

from meetingintel.rag.retriever import IndexedMeeting, TfidfRetriever
from meetingintel.rag.voyage_retriever import VoyageRetriever, select_retriever


def _doc(meeting_id: str, text: str = "some meeting text", embedding: list[float] | None = None) -> IndexedMeeting:
    return IndexedMeeting(meeting_id=meeting_id, title=meeting_id, text=text, embedding=embedding)


def _fake_response(embeddings: list[list[float]]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "data": [{"embedding": e, "index": i} for i, e in enumerate(embeddings)],
        "usage": {"total_tokens": 10},
    }
    return resp


def test_index_computes_and_caches_embeddings():
    client = MagicMock()
    client.post.return_value = _fake_response([[1.0, 0.0], [0.0, 1.0]])
    retriever = VoyageRetriever(api_key="fake-key", client=client)

    docs = [_doc("m1"), _doc("m2")]
    retriever.index(docs)

    assert docs[0].embedding == [1.0, 0.0]
    assert docs[1].embedding == [0.0, 1.0]
    assert client.post.call_count == 1
    assert client.post.call_args.kwargs["json"]["input_type"] == "document"


def test_index_skips_documents_with_cached_embedding():
    client = MagicMock()
    client.post.return_value = _fake_response([[9.0, 9.0]])
    retriever = VoyageRetriever(api_key="fake-key", client=client)

    already_embedded = _doc("m1", text="already has one", embedding=[1.0, 1.0])
    needs_embedding = _doc("m2", text="needs one")
    retriever.index([already_embedded, needs_embedding])

    assert already_embedded.embedding == [1.0, 1.0]  # untouched
    assert needs_embedding.embedding == [9.0, 9.0]
    sent_texts = client.post.call_args.kwargs["json"]["input"]
    assert sent_texts == ["needs one"]


def test_index_empty_list_does_not_call_api():
    client = MagicMock()
    retriever = VoyageRetriever(api_key="fake-key", client=client)
    retriever.index([])
    assert client.post.call_count == 0


def test_index_all_already_embedded_does_not_call_api():
    client = MagicMock()
    retriever = VoyageRetriever(api_key="fake-key", client=client)
    retriever.index([_doc("m1", embedding=[1.0, 0.0])])
    assert client.post.call_count == 0


def test_query_before_index_returns_empty():
    client = MagicMock()
    retriever = VoyageRetriever(api_key="fake-key", client=client)
    assert retriever.query("anything") == []


def test_query_with_no_embedded_docs_returns_empty():
    client = MagicMock()
    retriever = VoyageRetriever(api_key="fake-key", client=client)
    retriever.index([_doc("m1", embedding=[1.0, 0.0])])
    # simulate embedding failing to attach (shouldn't happen in practice, but guard anyway)
    retriever._docs[0].embedding = None
    assert retriever.query("anything") == []


def test_query_ranks_by_cosine_similarity():
    client = MagicMock()
    retriever = VoyageRetriever(api_key="fake-key", client=client)

    doc_a = _doc("close", embedding=[1.0, 0.0])
    doc_b = _doc("far", embedding=[0.0, 1.0])
    retriever.index([doc_a, doc_b])
    assert client.post.call_count == 0  # both already embedded, index() made no calls

    client.post.return_value = _fake_response([[0.9, 0.1]])
    results = retriever.query("query text")

    assert client.post.call_args.kwargs["json"]["input_type"] == "query"
    assert [r.meeting_id for r in results] == ["close", "far"]
    assert results[0].score > results[1].score


def test_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        VoyageRetriever()


def test_http_error_propagates():
    client = MagicMock()
    resp = MagicMock()
    resp.raise_for_status.side_effect = httpx.HTTPStatusError("bad key", request=MagicMock(), response=MagicMock())
    client.post.return_value = resp
    retriever = VoyageRetriever(api_key="fake-key", client=client)

    with pytest.raises(httpx.HTTPStatusError):
        retriever.index([_doc("m1")])


def test_select_retriever_chooses_voyage_when_key_present(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "fake-key")
    assert isinstance(select_retriever(), VoyageRetriever)


def test_select_retriever_chooses_tfidf_when_absent(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert isinstance(select_retriever(), TfidfRetriever)
