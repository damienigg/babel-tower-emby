"""Tests for the LLM-backed translation provider. We don't talk to a real
LLM — the LLMClient protocol is small, so we substitute a fake client that
returns a canned response (or raises) and assert the provider wraps it
correctly.

Coverage:
- length-mismatch detection (LLM returns N != len(input) cues)
- missing-id detection (LLM returns the right count but wrong ids)
- invalid-JSON detection
- batch-size selection (translation_batch_size)
- LLMError → TranslationError translation
- lang-pair system block shape

Pre-0.7.32 this file also covered scene/cinematic-specific payload
shape (scene bible, per-cue image blocks, cue-to-scene annotations)
and the cinematic batch-size + vision-gating paths. With scene and
cinematic modes removed those tests went too — the provider is now
text-only.
"""
from dataclasses import dataclass

import pytest

from app.pipeline.llm.base import (
    ContentBlock, LLMError, SystemBlock, TextContent,
)
from app.pipeline.stt import Cue
from app.pipeline.translate.base import TranslationError


# ── Fake LLM client ──────────────────────────────────────────────────────────


@dataclass
class _FakeChat:
    """One observed call to chat(). The provider's tests assert on these to
    verify the payload it built (batch size, system shape, etc.)."""
    system: list[SystemBlock]
    content: list[ContentBlock]
    max_tokens: int
    response_schema: dict | None


class FakeLLM:
    """Minimal LLMClient implementation that returns canned responses (or
    raises) and records every chat() invocation."""

    def __init__(
        self,
        responses: list[str] | None = None,
        raises: Exception | None = None,
        supports_vision_value: bool = False,
    ) -> None:
        self.responses = list(responses or [])
        self.raises = raises
        self._supports_vision = supports_vision_value
        self.calls: list[_FakeChat] = []

    def supports_vision(self) -> bool:
        return self._supports_vision

    def chat(self, *, system, content, max_tokens=16000, response_schema=None):
        self.calls.append(_FakeChat(system=list(system), content=list(content),
                                     max_tokens=max_tokens,
                                     response_schema=response_schema))
        if self.raises is not None:
            raise self.raises
        if not self.responses:
            raise AssertionError("FakeLLM ran out of canned responses")
        return self.responses.pop(0)


def _make_provider(fake: FakeLLM, monkeypatch):
    """Construct an LLMTranslationProvider that uses the given FakeLLM
    instead of building one via settings.translation_llm_*."""
    from app.pipeline.translate import llm as llm_mod
    monkeypatch.setattr(llm_mod, "get_translation_llm", lambda: fake)
    return llm_mod.LLMTranslationProvider()


def _cues(n: int) -> list[Cue]:
    return [Cue(id=i, start=float(i), end=float(i) + 1.0, text=f"line {i}")
            for i in range(n)]


def _ok_response_for(cues: list[Cue], prefix: str = "tr-") -> str:
    """Build a JSON response that matches the input cues' ids and count —
    the happy-path translation."""
    import json
    return json.dumps({
        "translations": [{"id": c.id, "text": f"{prefix}{c.text}"} for c in cues],
    })


# ── Happy path ───────────────────────────────────────────────────────────────


def test_translate_happy_path_round_trip(monkeypatch):
    cues = _cues(3)
    fake = FakeLLM(responses=[_ok_response_for(cues)])
    provider = _make_provider(fake, monkeypatch)
    out = provider.translate(cues, source_lang="en", target_lang="fr")
    assert [c.text for c in out] == ["tr-line 0", "tr-line 1", "tr-line 2"]
    # Timing must be preserved exactly — the LLM controls text only.
    for orig, translated in zip(cues, out):
        assert (orig.id, orig.start, orig.end) == (translated.id, translated.start, translated.end)


# ── Failure modes ────────────────────────────────────────────────────────────


def test_translate_raises_on_length_mismatch(monkeypatch):
    """The most common silent corruption mode: model drops or duplicates
    cues. Must surface as TranslationError with a clear message."""
    cues = _cues(3)
    bad = '{"translations": [{"id": 0, "text": "a"}, {"id": 1, "text": "b"}]}'
    fake = FakeLLM(responses=[bad])
    provider = _make_provider(fake, monkeypatch)
    with pytest.raises(TranslationError, match="Length mismatch"):
        provider.translate(cues, "en", "fr")


def test_translate_raises_on_missing_id(monkeypatch):
    """Right count, wrong ids — model preserved length but renumbered."""
    cues = _cues(3)
    bad = ('{"translations": ['
           '{"id": 99, "text": "a"}, {"id": 1, "text": "b"}, {"id": 2, "text": "c"}]}')
    fake = FakeLLM(responses=[bad])
    provider = _make_provider(fake, monkeypatch)
    with pytest.raises(TranslationError, match="Missing translation for cue id 0"):
        provider.translate(cues, "en", "fr")


def test_translate_raises_on_invalid_json(monkeypatch):
    cues = _cues(2)
    fake = FakeLLM(responses=["not json {{ at all"])
    provider = _make_provider(fake, monkeypatch)
    with pytest.raises(TranslationError, match="invalid JSON"):
        provider.translate(cues, "en", "fr")


def test_translate_wraps_llm_errors_as_translation_errors(monkeypatch):
    """LLMError raised by the underlying client (network, auth, etc.)
    becomes TranslationError — callers shouldn't have to know about the
    LLM layer's exception hierarchy."""
    cues = _cues(2)
    fake = FakeLLM(raises=LLMError("backend exploded"))
    provider = _make_provider(fake, monkeypatch)
    with pytest.raises(TranslationError, match="backend exploded"):
        provider.translate(cues, "en", "fr")


# ── Batch-size selection ─────────────────────────────────────────────────────


def test_text_only_batches_use_translation_batch_size(monkeypatch):
    """30-cue input with translation_batch_size=10 should hit the LLM 3
    times. Only one batch size knob remains since scene/cinematic was
    removed."""
    from app.config import settings as _settings
    monkeypatch.setattr(_settings, "_overrides",
                        {**_settings._overrides, "translation_batch_size": 10})
    cues = _cues(30)
    fake = FakeLLM(responses=[
        _ok_response_for(cues[:10]),
        _ok_response_for(cues[10:20]),
        _ok_response_for(cues[20:]),
    ])
    provider = _make_provider(fake, monkeypatch)
    out = provider.translate(cues, "en", "fr")
    assert len(out) == 30
    assert len(fake.calls) == 3


# ── Payload shape ────────────────────────────────────────────────────────────


def test_lang_pair_appears_in_system(monkeypatch):
    """The source/target pair lives in its own system block so callers can
    inspect & swap without rewriting the principles block."""
    cues = _cues(1)
    fake = FakeLLM(responses=[_ok_response_for(cues)])
    provider = _make_provider(fake, monkeypatch)
    provider.translate(cues, "ja", "fr")
    call = fake.calls[0]
    lang_blocks = [b for b in call.system
                   if "Source language: ja" in b.text and "Target language: fr" in b.text]
    assert len(lang_blocks) == 1


def test_user_content_is_just_the_cue_json(monkeypatch):
    """Post-0.7.32 the user content is exactly one TextContent block
    carrying the JSON cue list — no image blocks, no per-cue labels.
    Pinning this stops a future refactor from accidentally reintroducing
    multimodal plumbing."""
    import json
    cues = _cues(2)
    fake = FakeLLM(responses=[_ok_response_for(cues)])
    provider = _make_provider(fake, monkeypatch)
    provider.translate(cues, "en", "fr")
    content = fake.calls[0].content
    assert len(content) == 1
    assert isinstance(content[0], TextContent)
    payload = json.loads(content[0].text)
    assert [p["id"] for p in payload] == [0, 1]
    assert [p["text"] for p in payload] == ["line 0", "line 1"]
    # No legacy 'scene' annotation field on any cue.
    assert all("scene" not in p for p in payload)
