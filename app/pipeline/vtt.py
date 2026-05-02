import textwrap

from app.config import settings
from app.pipeline.stt import Cue


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _wrap(text: str) -> str:
    """Wrap a cue across at most `max_lines_per_cue` lines. If the text won't
    fit, prefer overflowing lines to silently dropping content — readers can
    still see the full translation, just on a slightly busier display."""
    if len(text) <= settings.max_line_chars:
        return text
    lines = textwrap.wrap(text, width=settings.max_line_chars)
    cap = max(1, settings.max_lines_per_cue)
    if len(lines) <= cap:
        return "\n".join(lines)
    # Too many lines for the configured cap. Keep the first (cap-1) lines as-is
    # and merge everything else into the last line so nothing is dropped.
    head = lines[: cap - 1]
    tail = " ".join(lines[cap - 1:])
    return "\n".join([*head, tail])


def to_webvtt(cues: list[Cue], header_note: str | None = None) -> str:
    parts = ["WEBVTT", ""]
    if header_note:
        parts.append(f"NOTE {header_note}")
        parts.append("")
    for cue in cues:
        start = _format_timestamp(cue.start)
        end = _format_timestamp(cue.end)
        parts.append(f"{start} --> {end}")
        parts.append(_wrap(cue.text))
        parts.append("")
    return "\n".join(parts)
