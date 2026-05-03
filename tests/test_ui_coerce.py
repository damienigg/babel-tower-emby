"""Tests for the form-coercion helper that gates every Settings save.

`_coerce(key, raw_string)` reads the type annotation off `_EnvSettings.{key}`
and converts the form-submitted string to whatever pydantic expects. The
implementation is non-trivial: type annotations come back from
`get_type_hints` as `bool | None` / `str | None` / `list[str]` / etc., not
bare classes — hence the `"bool" in target_str` substring checks. Easy to
break without a test pinning down the dispatch.
"""
from app.ui.routes import _coerce


# ── bool ─────────────────────────────────────────────────────────────────────


def test_coerce_bool_truthy_strings():
    """HTML checkboxes submit "on" when checked. Python truthy strings
    ("true"/"True"/"1"/"yes") are also accepted for API parity."""
    for raw in ("on", "true", "True", "1", "yes"):
        assert _coerce("vision_llm_enabled", raw) is True


def test_coerce_bool_falsy_strings():
    """Anything else is False — including empty (unchecked checkbox)."""
    for raw in ("", "off", "false", "False", "0", "no", "anything-else"):
        assert _coerce("vision_llm_enabled", raw) is False


# ── int ──────────────────────────────────────────────────────────────────────


def test_coerce_int_parses_digits():
    assert _coerce("translation_batch_size", "30") == 30
    assert _coerce("max_line_chars", "0") == 0


def test_coerce_int_empty_string_becomes_zero():
    """Empty number input → 0. Tradeoff: avoids a 422 on form save when
    the user clears a number field expecting "default", but means the
    pydantic validator sees 0 and passes (most are 0-tolerant)."""
    assert _coerce("translation_batch_size", "") == 0


# ── float ────────────────────────────────────────────────────────────────────


def test_coerce_float_parses_decimal():
    assert _coerce("scene_detection_threshold", "0.4") == 0.4


def test_coerce_float_empty_becomes_zero():
    assert _coerce("scene_detection_threshold", "") == 0.0


# ── list[str] ────────────────────────────────────────────────────────────────


def test_coerce_list_splits_on_comma_and_strips():
    """default_source_lang_priority: list[str]. Form submits a single
    string — we split on commas and strip whitespace."""
    assert _coerce("default_source_lang_priority", "en,ja,*") == ["en", "ja", "*"]
    assert _coerce("default_source_lang_priority", "en, ja , * ") == ["en", "ja", "*"]


def test_coerce_list_empty_yields_empty_list():
    """All-whitespace or empty input gives an empty list — NOT a list with
    one empty string. Matters because pydantic would accept `[""]` as
    a valid list[str] but it's not what the user meant."""
    assert _coerce("default_source_lang_priority", "") == []
    assert _coerce("default_source_lang_priority", "  ,  ") == []


def test_coerce_list_drops_blank_segments_between_commas():
    """`a,,b` → `["a", "b"]` — defensive against double commas typos."""
    assert _coerce("default_source_lang_priority", "a,,b") == ["a", "b"]


# ── str (default) ────────────────────────────────────────────────────────────


def test_coerce_str_passthrough():
    """Plain text fields just round-trip the raw string."""
    assert _coerce("media_server_url", "http://emby:8096") == "http://emby:8096"
    assert _coerce("translation_llm_model", "claude-opus-4-7") == "claude-opus-4-7"


def test_coerce_optional_str_passthrough():
    """`str | None` fields (the typical optional secret) still come back
    as a string from the form — the empty-secret case is handled at a
    higher level (skipped before _coerce runs in routes.settings_save)."""
    assert _coerce("media_server_api_key", "secret-value") == "secret-value"


def test_coerce_unknown_key_treats_as_str():
    """Defensive default: if the key isn't on _EnvSettings, str passthrough.
    In production this branch is unreachable (settings_save filters by
    valid_keys first) but the helper shouldn't crash if called wrong."""
    assert _coerce("not-a-real-setting", "hello") == "hello"
