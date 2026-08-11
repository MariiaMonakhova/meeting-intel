"""Speaker-aware semantic chunking.

Packs whole Utterances (never split mid-utterance) into Chunks up to
chunk_size tokens, carrying a trailing whole-utterance overlap into the next
chunk so context isn't lost at chunk boundaries. Unlike a streaming
document chunker, this operates on an already-parsed Transcript held in
memory - ingest.py has already bounded the input to one meeting's length,
so there's no need for block-based streaming here.
"""

from __future__ import annotations

from typing import Iterator

from meetingintel.models import Chunk, Transcript, Utterance
from meetingintel.tokens import count_tokens


def _format(utterances: list[Utterance]) -> str:
    return "\n".join(f"{u.speaker}: {u.text}" for u in utterances)


def _text_tokens(utterances: list[Utterance]) -> int:
    if not utterances:
        return 0
    return count_tokens(_format(utterances))


def _trailing_overlap(window: list[Utterance], overlap_tokens: int) -> list[Utterance]:
    """Trailing whole utterances from `window` whose combined size fits in overlap_tokens."""
    if overlap_tokens <= 0 or not window:
        return []
    outgoing: list[Utterance] = []
    for u in reversed(window):
        candidate = [u] + outgoing
        if _text_tokens(candidate) > overlap_tokens:
            break
        outgoing = candidate
    return outgoing


def _pack_utterances(
    utterances: list[Utterance], tokens_per_chunk: int, overlap_tokens: int
) -> Iterator[tuple[list[Utterance], list[Utterance]]]:
    """Yields (window, incoming_overlap) pairs.

    incoming_overlap is the trailing slice of the *previous* window that was
    carried forward to seed this one. Every outer iteration adds at least one
    utterance beyond that carry regardless of size - this guarantees `i`
    always advances, so a single utterance longer than tokens_per_chunk gets
    its own chunk instead of causing an infinite loop.
    """
    if not utterances:
        return
    i = 0
    n = len(utterances)
    carry: list[Utterance] = []
    while i < n:
        incoming = carry
        window = list(carry)
        added_new = False
        while i < n:
            candidate = window + [utterances[i]]
            if added_new and _text_tokens(candidate) > tokens_per_chunk:
                break
            window = candidate
            added_new = True
            i += 1
        outgoing = _trailing_overlap(window, overlap_tokens)
        yield window, incoming
        carry = outgoing


def chunk_speaker_semantic(
    transcript: Transcript, chunk_size: int, overlap: int
) -> Iterator[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    for chunk_id, (window, incoming) in enumerate(
        _pack_utterances(transcript.utterances, chunk_size, overlap)
    ):
        yield Chunk(
            id=chunk_id,
            utterances=window,
            start_index=window[0].index,
            end_index=window[-1].index,
            token_count=count_tokens(_format(window)),
            overlap_tokens=_text_tokens(incoming),
            speakers=sorted({u.speaker for u in window}),
        )
