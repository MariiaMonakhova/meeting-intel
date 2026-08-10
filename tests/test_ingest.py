from meetingintel.ingest import parse_transcript


def test_parses_speaker_and_text():
    t = parse_transcript("Alice: Hello there", meeting_id="m1", title="Test")
    assert len(t.utterances) == 1
    assert t.utterances[0].speaker == "Alice"
    assert t.utterances[0].text == "Hello there"
    assert t.utterances[0].timestamp is None


def test_parses_timestamp_prefix():
    t = parse_transcript("[00:01:23] Bob: Hi there", meeting_id="m1", title="Test")
    assert t.utterances[0].timestamp == "00:01:23"
    assert t.utterances[0].speaker == "Bob"
    assert t.utterances[0].text == "Hi there"


def test_continuation_line_appends_to_previous_utterance():
    raw = "Alice: This is a long thought\nthat continues on the next line."
    t = parse_transcript(raw, meeting_id="m1", title="Test")
    assert len(t.utterances) == 1
    assert t.utterances[0].text == "This is a long thought that continues on the next line."


def test_leading_continuation_line_without_prior_utterance_is_dropped():
    raw = "this line has no speaker prefix\nBob: Now this one does"
    t = parse_transcript(raw, meeting_id="m1", title="Test")
    assert len(t.utterances) == 1
    assert t.utterances[0].speaker == "Bob"


def test_blank_lines_are_ignored():
    raw = "Alice: Hi\n\n\nBob: Hello"
    t = parse_transcript(raw, meeting_id="m1", title="Test")
    assert len(t.utterances) == 2


def test_attendees_are_sorted_and_deduped():
    raw = "Bob: Hi\nAlice: Hey\nBob: Again"
    t = parse_transcript(raw, meeting_id="m1", title="Test")
    assert t.attendees == ["Alice", "Bob"]


def test_index_is_sequential():
    raw = "Alice: One\nBob: Two\nAlice: Three"
    t = parse_transcript(raw, meeting_id="m1", title="Test")
    assert [u.index for u in t.utterances] == [0, 1, 2]


def test_empty_transcript_has_no_utterances_and_no_attendees():
    t = parse_transcript("", meeting_id="m1", title="Test")
    assert t.utterances == []
    assert t.attendees == []
