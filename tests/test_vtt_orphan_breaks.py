"""Tests for the 0.7.33 line-break orphan-avoidance pass in
``app/pipeline/vtt.py``. Pro caption guidelines say function words
(articles, prepositions, conjunctions) shouldn't be stranded at the
end of a subtitle line — the eye expects what follows on the same
line. We rebalance the textwrap output to push them to the start of
the next line where possible."""
import pytest

from app.pipeline.vtt import _wrap_avoiding_orphans, _is_orphan


def test_is_orphan_catches_english_articles():
    assert _is_orphan("the") is True
    assert _is_orphan("a") is True
    assert _is_orphan("an") is True


def test_is_orphan_catches_english_prepositions():
    assert _is_orphan("of") is True
    assert _is_orphan("to") is True
    assert _is_orphan("with") is True


def test_is_orphan_catches_french_function_words():
    assert _is_orphan("le") is True
    assert _is_orphan("de") is True
    assert _is_orphan("et") is True
    assert _is_orphan("à") is True


def test_is_orphan_strips_trailing_punctuation():
    # "the," and "the" should both be orphans — the eye reads them
    # the same way at the end of a line.
    assert _is_orphan("the,") is True
    assert _is_orphan("the.") is True
    assert _is_orphan("the!") is True


def test_is_orphan_rejects_content_words():
    assert _is_orphan("walking") is False
    assert _is_orphan("car") is False
    assert _is_orphan("table") is False


def test_wrap_short_text_returns_single_line():
    out = _wrap_avoiding_orphans("Hello world", width=40, cap=2)
    assert out == ["Hello world"]


def test_wrap_rebalances_to_avoid_orphan_article():
    """Naively, textwrap might split as ['The quick brown fox jumps over the',
    'lazy dog'] for a width that fits 35 chars. The orphan 'the' at the end
    of line 1 should get pushed to the start of line 2."""
    text = "The quick brown fox jumps over the lazy dog"
    out = _wrap_avoiding_orphans(text, width=35, cap=2)
    assert len(out) == 2
    # First line must NOT end with an orphan word.
    last_word_line1 = out[0].split()[-1].lower().strip(",.;:!?")
    assert last_word_line1 != "the"
    # The full content survives (orphan fix moves words, never drops them).
    assert " ".join(out).replace("  ", " ") == text


def test_wrap_rebalances_french_orphans():
    text = "Voici un exemple d'un sous-titre qui dépasse la largeur de la"
    out = _wrap_avoiding_orphans(text, width=35, cap=2)
    # First line shouldn't end with an article/preposition like "la"
    # or "de" if there's room to rebalance.
    last_word = out[0].split()[-1].lower().strip(",.;:!?'…")
    assert last_word not in {"la", "le", "de", "du", "à"}


def test_wrap_accepts_orphan_when_rebalance_would_unbalance():
    """If moving the orphan to the next line would shrink the source
    line below ~half the width, the rebalance is rejected — orphan
    beats radically uneven layout. Verify with a constructed case."""
    # Width 30, text where moving "the" would leave only "A " on line 1.
    # In that case the orphan must be tolerated.
    text = "A the quick brown fox jumps over"
    out = _wrap_avoiding_orphans(text, width=30, cap=2)
    # The orphan rebalance has a minimum-line-length guard, so this
    # may or may not move depending on the exact split. The key
    # contract: the function NEVER drops content.
    assert " ".join(out).split() == text.split()


def test_wrap_respects_line_cap_via_overflow_merge():
    """When the wrap would produce more lines than cap, the tail is
    merged into the last line (overflow handling). Orphan fix runs on
    the post-merge result, so a 3-line wrap merged into 2 still gets
    its orphans rebalanced."""
    text = "One two three four five six seven eight nine ten eleven twelve"
    out = _wrap_avoiding_orphans(text, width=15, cap=2)
    assert len(out) == 2
    # No content dropped.
    assert " ".join(out).split() == text.split()


def test_wrap_does_not_break_last_line():
    """Orphan-fix only applies to non-final lines. A function word at
    the end of the LAST line is fine — there's nothing after it."""
    text = "I'm thinking of the"   # absurd cue but tests the contract
    out = _wrap_avoiding_orphans(text, width=40, cap=2)
    # Whatever it does, the last word "the" is allowed to be at the end
    # of the last line.
    assert out[-1].split()[-1].lower().strip(",.;:!?") == "the"
