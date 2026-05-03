"""Tests for the LLM-backed translation provider. We don't talk to a real
LLM — the LLMClient protocol is small, so we substitute a fake client that
returns a canned response (or raises) and assert the provider wraps it
correctly.

Coverage:
- length-mismatch detection (LLM returns N != len(input) cues)
- missing-id detection (LLM returns the right count but wrong ids)
- invalid-JSON detection
- batch-size selection (text-only vs cinematic)
- cinematic-mode vision-capability gating
- LLMError → TranslationError translation
- scene-bible payload structure (system blocks, cue annotations)
"""
from dataclasses import dataclass

import pytest

from app.pipeline.llm.base import (
    ContentBlock, ImageContent, LLMError, SystemBlock, TextContent,
)
from app.pipeline.stt import Cue
from app.pipeline.translate.base import (
    SceneInfo, TranslationContext, TranslationError,
)


# ── Fake LLM client ──────────────────────────────────────────────────────────


@dataclass
class _FakeChat:
    """One observed call to chat(). The provider's tests assert on these to
    verify the payload it built (batch size, scene bible, frames, etc.)."""
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
        supports_vision_value: bool = True,
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
    times (no cue_frames context → text-only batching)."""
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


def test_cinematic_uses_cinematic_batch_size(monkeypatch):
    """When context carries cue_frames (cinematic mode), the smaller
    cinematic_batch_size kicks in instead of translation_batch_size."""
    from app.config import settings as _settings
    monkeypatch.setattr(_settings, "_overrides",
                        {**_settings._overrides,
                         "translation_batch_size": 30,
                         "cinematic_batch_size": 4})
    cues = _cues(10)
    # 10 cues / batch 4 → 3 calls (4 + 4 + 2)
    fake = FakeLLM(responses=[
        _ok_response_for(cues[:4]),
        _ok_response_for(cues[4:8]),
        _ok_response_for(cues[8:]),
    ])
    provider = _make_provider(fake, monkeypatch)
    ctx = TranslationContext(
        scenes=[],
        cue_to_scene={},
        cue_frames={c.id: b"\xff\xd8\xff" for c in cues},  # fake JPEGs
    )
    out = provider.translate(cues, "en", "fr", context=ctx)
    assert len(out) == 10
    assert len(fake.calls) == 3


# ── Cinematic vision-capability gating ───────────────────────────────────────


def test_cinematic_with_non_vision_llm_raises_clearly(monkeypatch):
    """User's LLM doesn't support vision but they ran cinematic mode —
    we must catch this before paying for a single LLM call."""
    cues = _cues(3)
    fake = FakeLLM(supports_vision_value=False)
    provider = _make_provider(fake, monkeypatch)
    ctx = TranslationContext(
        scenes=[], cue_to_scene={},
        cue_frames={c.id: b"\xff" for c in cues},
    )
    with pytest.raises(TranslationError, match="vision-capable"):
        provider.translate(cues, "en", "fr", context=ctx)
    # Crucially: NO LLM call was made — we'd have wasted user $$$ for nothing.
    assert fake.calls == []


# ── Payload shape ────────────────────────────────────────────────────────────


def test_system_includes_scene_bible_when_provided(monkeypatch):
    """In scene/cinematic, the system prompt should carry both the static
    principles AND the per-film bible — and the bible should be flagged
    cacheable so Anthropic prompt-caching kicks in."""
    cues = _cues(2)
    fake = FakeLLM(responses=[_ok_response_for(cues)])
    provider = _make_provider(fake, monkeypatch)
    ctx = TranslationContext(
        scenes=[SceneInfo(index=0, start=0.0, end=10.0, description="A train station at dusk.")],
        cue_to_scene={0: 0, 1: 0},
        cue_frames={},  # text-only scene mode
    )
    provider.translate(cues, "en", "fr", context=ctx)
    call = fake.calls[0]
    bible_blocks = [b for b in call.system if "Scene bible" in b.text]
    assert len(bible_blocks) == 1
    assert "train station at dusk" in bible_blocks[0].text
    assert bible_blocks[0].cacheable is True


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


def test_cinematic_attaches_per_cue_image_blocks(monkeypatch):
    """Cinematic mode interleaves an ImageContent block per cue alongside
    the JSON payload. The label `Frame for cue N:` keeps the LLM oriented."""
    cues = _cues(2)
    fake = FakeLLM(responses=[_ok_response_for(cues)])
    provider = _make_provider(fake, monkeypatch)
    ctx = TranslationContext(
        scenes=[], cue_to_scene={},
        cue_frames={0: b"\xff\xd8frame0", 1: b"\xff\xd8frame1"},
    )
    provider.translate(cues, "en", "fr", context=ctx)
    content = fake.calls[0].content
    # First two pairs are (TextContent label, ImageContent), then the JSON payload.
    assert isinstance(content[0], TextContent)
    assert "Frame for cue 0" in content[0].text
    assert isinstance(content[1], ImageContent)
    assert content[1].data == b"\xff\xd8frame0"
    assert isinstance(content[2], TextContent)
    assert "Frame for cue 1" in content[2].text
    assert isinstance(content[3], ImageContent)
    # Last block is the JSON cue payload.
    assert isinstance(content[-1], TextContent)
    assert '"line 0"' in content[-1].text


def test_cue_scene_annotation_when_cue_to_scene_maps(monkeypatch):
    """Scene mode (no cue_frames) annotates each cue in the JSON payload
    with the description of the scene it belongs to. The bible itself stays
    in the system; the inline annotation is shorthand for the LLM."""
    import json
    cues = _cues(2)
    fake = FakeLLM(responses=[_ok_response_for(cues)])
    provider = _make_provider(fake, monkeypatch)
    ctx = TranslationContext(
        scenes=[SceneInfo(index=0, start=0.0, end=10.0, description="Café interior")],
        cue_to_scene={0: 0},  # only cue 0 mapped — cue 1 gets no scene annotation
        cue_frames={},
    )
    provider.translate(cues, "en", "fr", context=ctx)
    payload = json.loads(fake.calls[0].content[-1].text)
    assert payload[0]["scene"] == "Café interior"
    assert "scene" not in payload[1]
