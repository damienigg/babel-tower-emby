import textwrap

from app.config import settings
from app.pipeline.stt import Cue


# Function words that read awkwardly when stranded at the end of a
# subtitle line. Pro caption guidelines say "never break a phrase
# between an article/preposition and its noun", because the eye
# expects what comes next on the same line. We detect by exact
# match (lowercased, punctuation-stripped) on the LAST word of each
# candidate split — if it's in this set, we try to rebalance.
#
# English + French because those are the languages our test cases
# cover; other languages still wrap to width, just without the
# orphan-fixup pass. Adding a language is just a matter of appending
# its function-word set here.
_ORPHAN_TAIL_WORDS = frozenset({
    # English articles + prepositions + conjunctions that orphan
    "a", "an", "the",
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto",
    "and", "or", "but", "so", "if", "as",
    "is", "was", "are", "were", "be", "been",
    # French
    "le", "la", "les", "un", "une", "des",
    "de", "du", "à", "au", "aux", "en", "sur", "dans", "avec", "pour",
    # "car" (French "because") deliberately omitted — collides with
    # English "car" (vehicle), and dropping the orphan-fix on French
    # "car" is the lesser harm vs. mistreating English content words.
    "et", "ou", "mais", "donc", "ni",
    "se", "ce", "cette", "ces",
    "qui", "que", "qu",
})


def _format_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _is_orphan(word: str) -> bool:
    """Lower-cases + strips trailing punctuation, then checks against
    the orphan-tail set. We accept the punctuation strip because pro
    captioners care about the *visual* word, not the punctuated one
    — ``"the,"`` and ``"the"`` read the same at the end of a line."""
    bare = word.strip(",.;:!?\"'()[]…«»").lower()
    return bare in _ORPHAN_TAIL_WORDS


def _wrap_avoiding_orphans(text: str, width: int, *, cap: int) -> list[str]:
    """Wrap ``text`` at ``width`` columns into at most ``cap`` lines,
    trying to avoid leaving an orphan function word (article /
    preposition / conjunction) at the end of any line except the last.

    Strategy: start from textwrap's split (which already balances on
    word boundaries), then for each NON-FINAL line whose last word is
    an orphan, walk one word backward into the previous boundary
    until either the orphan is gone or the line gets too short
    (< 50% of width — at which point we give up; orphan beats a
    radically unbalanced layout).

    Pure wrapping concern; returns the same list of lines textwrap
    would have, possibly with adjusted split points. Total content
    is preserved exactly."""
    if len(text) <= width:
        return [text]

    lines = textwrap.wrap(text, width=width)
    if cap >= 1 and len(lines) > cap:
        # Same overflow-merge as the original _wrap — keep this here
        # so the orphan-fix only operates on the actually-rendered
        # layout, not on the pre-merge form.
        head = lines[: cap - 1]
        tail = " ".join(lines[cap - 1:])
        lines = [*head, tail]

    # Orphan-fix pass: for every non-last line, if it ends with an
    # orphan word, try to move that word to the start of the next
    # line. Only applies when the result is still readable (the
    # source line stays at ≥ width/2 chars).
    min_acceptable = max(8, width // 2)
    for i in range(len(lines) - 1):
        words = lines[i].split()
        if not words or not _is_orphan(words[-1]):
            continue
        # Move the trailing orphan into the next line. Re-check the
        # NEW length on both sides — we accept the shift unless the
        # source line drops below min_acceptable.
        candidate_src = " ".join(words[:-1])
        if len(candidate_src) < min_acceptable:
            continue
        moved = words[-1]
        lines[i] = candidate_src
        lines[i + 1] = moved + " " + lines[i + 1]
        # If the NEXT line now exceeds width, that's acceptable —
        # subtitle renderers handle overflow gracefully and the
        # alternative (dropping content) would be worse.
    return lines


def _wrap(text: str) -> str:
    """Wrap a cue across at most ``max_lines_per_cue`` lines, avoiding
    function-word orphans at line ends where possible. If the text
    can't fit even with rebalancing, prefer overflowing lines over
    silently dropping content."""
    width = int(settings.max_line_chars)
    cap = max(1, int(settings.max_lines_per_cue))
    lines = _wrap_avoiding_orphans(text, width, cap=cap)
    return "\n".join(lines)


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
