"""Processor validation tests. Heavy externals (ffmpeg, Whisper, LLM) are
mocked — we only want to verify the validation gates and error mapping."""
import pytest

from app.config import settings
from app.processor import (
    BadRequest, MediaNotFound, ProcessRequest, SUPPORTED_MODES, process,
)


def _req(**overrides):
    base = dict(
        media_path="/nonexistent/file.mkv",
        target_lang="fr",
        source_lang_priority=["en", "*"],
        translation_provider="llm",
        mode="audio",
    )
    base.update(overrides)
    return ProcessRequest(**base)


def test_unknown_mode_raises_bad_request():
    with pytest.raises(BadRequest, match="unknown mode"):
        process(_req(mode="bogus"))


def test_supported_modes_cover_documented_set():
    assert "audio" in SUPPORTED_MODES
    assert "scene" in SUPPORTED_MODES
    assert "cinematic" in SUPPORTED_MODES


def test_scene_mode_requires_llm_provider(monkeypatch):
    monkeypatch.setattr(settings, "_overrides",
                        {**settings._overrides, "vision_llm_enabled": True})
    with pytest.raises(BadRequest, match="translation_provider='llm'"):
        process(_req(mode="scene", translation_provider="deepl"))


def test_scene_mode_requires_vision_enabled(monkeypatch):
    monkeypatch.setattr(settings, "_overrides",
                        {**settings._overrides, "vision_llm_enabled": False})
    with pytest.raises(BadRequest, match="Vision LLM"):
        process(_req(mode="scene"))


def test_cinematic_mode_requires_translation_vision(monkeypatch):
    monkeypatch.setattr(settings, "_overrides",
                        {**settings._overrides,
                         "vision_llm_enabled": True,
                         "translation_llm_supports_vision": False})
    with pytest.raises(BadRequest, match="cinematic"):
        process(_req(mode="cinematic"))


def test_audio_mode_with_missing_media_raises_media_not_found():
    with pytest.raises(MediaNotFound):
        process(_req(mode="audio", media_path="/no/such/file.mkv"))
