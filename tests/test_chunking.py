import pytest

from meetingintel.chunking import chunk_speaker_semantic
from meetingintel.models import Transcript, Utterance


def _transcript(n: int, words_per_utterance: int = 10, speakers: tuple[str, ...] = ("Alice", "Bob")) -> Transcript:
    utterances = [
        Utterance(
            speaker=speakers[i % len(speakers)],
            text=" ".join(f"word{i}_{w}" for w in range(words_per_utterance)),
            index=i,
        )
        for i in range(n)
    ]
    return Transcript(meeting_id="m1", title="Test", utterances=utterances)


def test_never_splits_an_utterance():
    t = _transcript(20)
    for chunk in chunk_speaker_semantic(t, chunk_size=30, overlap=5):
        for u in chunk.utterances:
            # every utterance in a chunk is one of the original objects, whole
            assert u in t.utterances


def test_offsets_cover_full_range_without_gaps():
    t = _transcript(15)
    chunks = list(chunk_speaker_semantic(t, chunk_size=20, overlap=3))
    assert chunks[0].start_index == 0
    assert chunks[-1].end_index == t.utterances[-1].index
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_index <= prev.end_index + 1


def test_oversized_single_utterance_does_not_hang():
    long_text = " ".join(f"w{i}" for i in range(500))
    utterances = [Utterance(speaker="Alice", text=long_text, index=0), Utterance(speaker="Bob", text="short", index=1)]
    t = Transcript(meeting_id="m1", title="Test", utterances=utterances)
    chunks = list(chunk_speaker_semantic(t, chunk_size=10, overlap=2))
    assert len(chunks) == 2
    assert chunks[0].utterances[0].speaker == "Alice"


def test_overlap_greater_or_equal_chunk_size_raises():
    t = _transcript(5)
    with pytest.raises(ValueError):
        list(chunk_speaker_semantic(t, chunk_size=10, overlap=10))
    with pytest.raises(ValueError):
        list(chunk_speaker_semantic(t, chunk_size=10, overlap=15))


def test_empty_transcript_returns_no_chunks():
    t = Transcript(meeting_id="m1", title="Test", utterances=[])
    assert list(chunk_speaker_semantic(t, chunk_size=100, overlap=10)) == []


def test_single_utterance_transcript_returns_one_chunk():
    t = _transcript(1)
    chunks = list(chunk_speaker_semantic(t, chunk_size=100, overlap=10))
    assert len(chunks) == 1
    assert chunks[0].overlap_tokens == 0


def test_first_chunk_has_no_incoming_overlap():
    t = _transcript(20)
    chunks = list(chunk_speaker_semantic(t, chunk_size=20, overlap=5))
    assert chunks[0].overlap_tokens == 0


def test_zero_overlap_produces_no_overlap_tokens_anywhere():
    t = _transcript(20)
    chunks = list(chunk_speaker_semantic(t, chunk_size=15, overlap=0))
    assert all(c.overlap_tokens == 0 for c in chunks)


def test_speakers_are_sorted_and_deduped_within_chunk():
    t = _transcript(10, speakers=("Bob", "Alice"))
    for chunk in chunk_speaker_semantic(t, chunk_size=30, overlap=3):
        assert chunk.speakers == sorted(set(chunk.speakers))


@pytest.mark.parametrize("chunk_size,overlap", [(10, 0), (20, 3), (50, 10), (15, 14)])
def test_all_utterances_are_reachable_across_chunks(chunk_size, overlap):
    t = _transcript(25)
    chunks = list(chunk_speaker_semantic(t, chunk_size=chunk_size, overlap=overlap))
    covered_indices = {u.index for c in chunks for u in c.utterances}
    assert covered_indices == {u.index for u in t.utterances}
