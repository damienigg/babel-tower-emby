from app.pipeline.stt import Cue
from app.pipeline.vtt import _format_timestamp, _wrap, to_webvtt


def test_format_timestamp_short():
    assert _format_timestamp(0) == "00:00:00.000"
    assert _format_timestamp(1.234) == "00:00:01.234"


def test_format_timestamp_hours():
    assert _format_timestamp(3661.5) == "01:01:01.500"


def test_format_timestamp_negative_clamped_to_zero():
    assert _format_timestamp(-1.0) == "00:00:00.000"


def test_wrap_short_text_unchanged():
    # Default max_line_chars=42
    assert _wrap("Bonjour.") == "Bonjour."


def test_wrap_long_text_splits_to_two_lines():
    text = "A " * 50  # ~100 chars
    out = _wrap(text)
    assert "\n" in out


def test_wrap_overflow_merges_into_last_line_no_silent_drop():
    """Regression: previously truncated past max_lines_per_cue. Now must keep all words."""
    very_long = "word " * 80  # forces 3+ lines at default 42-char width
    out = _wrap(very_long)
    # Every input word must still appear in the output somewhere
    for w in very_long.split():
        assert w in out


def test_to_webvtt_minimal_format():
    cues = [Cue(id=0, start=0.0, end=2.5, text="Hello")]
    out = to_webvtt(cues)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:02.500" in out
    assert "Hello" in out


def test_to_webvtt_includes_header_note():
    cues = [Cue(id=0, start=0.0, end=1.0, text="Hi")]
    out = to_webvtt(cues, header_note="auto-generated")
    assert "NOTE auto-generated" in out
