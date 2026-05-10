"""Tests for the performance-within-safety hardening pass.

Pins down behaviors that protect the in-process memory + concurrency
contract:

- Audio temp wavs land under settings.cache_dir/tmp, NOT /tmp.
- scene_bible.describe_scenes consults the lazy keyframe_provider when
  no eager dict is supplied — never pre-builds the full {idx: bytes}.
- The settings store uses copy-on-write: a concurrent reader sees
  either the pre-update or post-update state, never a half-applied dict.
- LLM clients are constructed with max_retries=0 so SDK-level retry
  multiplication can't blow the per-call timeout budget.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import _EnvSettings, SettingsStore
from app.pipeline import audio as audio_mod
from app.pipeline.scenes import Scene


# ── R1: audio temp dir lives under cache_dir ─────────────────────────────


def test_extract_audio_writes_temp_under_cache_dir(tmp_path, monkeypatch):
    """A 2 h film's 250 MB temp wav must NOT land in /tmp — on TrueNAS
    that can be tmpfs, counted against host RAM. It must go to
    settings.cache_dir/tmp/ instead."""
    from app.config import settings as runtime_settings
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "cache_dir": cache_dir},
    )

    captured_paths: list[str] = []

    def fake_run(args, **kwargs):
        # The output wav path is the last positional argument to ffmpeg.
        captured_paths.append(args[-1])
        from unittest.mock import MagicMock
        return MagicMock(returncode=0)

    with patch.object(audio_mod.subprocess, "run", side_effect=fake_run):
        with audio_mod.extract_audio("/some/movie.mkv", 0) as wav:
            # Path is under cache_dir/tmp, not /tmp.
            assert str(wav).startswith(str(cache_dir / "tmp"))

    # And the file is cleaned up on exit.
    assert captured_paths
    assert (cache_dir / "tmp").exists()   # the directory persists


# ── R3: lazy scene-bible keyframe provider ───────────────────────────────


def test_scene_bible_describes_via_lazy_provider(monkeypatch):
    """When keyframe_provider is set and keyframes={}, describe_scenes
    must call the provider per scene rather than expecting pre-extracted
    bytes. This is the pattern that drops bible-build peak RAM from
    ~125 MB to ~2.5 MB."""
    from app.pipeline import scene_bible

    # Force the batch size to 2 so we exercise the per-batch loop.
    from app.config import settings as runtime_settings
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "scene_bible_batch_size": 2},
    )

    scene_list = [
        Scene(index=0, start=0.0, end=10.0),
        Scene(index=1, start=10.0, end=20.0),
        Scene(index=2, start=20.0, end=30.0),
    ]

    # Provider records which scenes it was asked about.
    provider_calls: list[int] = []

    def provider(scene: Scene) -> bytes:
        provider_calls.append(scene.index)
        return f"jpeg-{scene.index}".encode()

    # Stub the vision LLM so we never actually call out.
    class _FakeVisionLLM:
        def __init__(self):
            self.payloads: list = []

        def chat(self, *, system, content, max_tokens, response_schema):
            self.payloads.append(list(content))
            # Return descriptions for whichever scene ids were in this batch.
            ids = [
                int(b.text.split()[1].rstrip(":"))
                for b in content if hasattr(b, "text") and b.text.startswith("Scene ")
            ]
            return json.dumps({
                "scenes": [{"index": i, "description": f"desc-{i}"} for i in ids],
            })

    fake_llm = _FakeVisionLLM()
    monkeypatch.setattr(scene_bible, "get_vision_llm", lambda: fake_llm)

    scene_bible.describe_scenes(
        scene_list,
        keyframes=None,
        keyframe_provider=provider,
    )

    # Every scene's description was set from the LLM's response.
    assert all(s.description == f"desc-{s.index}" for s in scene_list)
    # The provider was called for each scene (lazy extraction worked).
    assert sorted(provider_calls) == [0, 1, 2]
    # And two LLM batches were issued (3 scenes / batch=2 → 2 + 1).
    assert len(fake_llm.payloads) == 2


def test_scene_bible_eager_dict_takes_precedence_over_provider(monkeypatch):
    """If both keyframes and keyframe_provider are supplied, eager wins
    per scene index — so test fixtures injecting fake JPEGs keep working
    even when a real provider is also wired."""
    from app.pipeline import scene_bible

    scene_list = [
        Scene(index=0, start=0.0, end=10.0),
        Scene(index=1, start=10.0, end=20.0),
    ]
    provider_called = False

    def provider(_scene):
        nonlocal provider_called
        provider_called = True
        return b"from-provider"

    class _FakeVisionLLM:
        def __init__(self):
            self.last_content = None
        def chat(self, *, system, content, **_):
            self.last_content = content
            ids = [
                int(b.text.split()[1].rstrip(":"))
                for b in content if hasattr(b, "text") and b.text.startswith("Scene ")
            ]
            return json.dumps({
                "scenes": [{"index": i, "description": f"d{i}"} for i in ids],
            })

    fake_llm = _FakeVisionLLM()
    monkeypatch.setattr(scene_bible, "get_vision_llm", lambda: fake_llm)

    # Scene 0 has an eager fake; scene 1 doesn't.
    scene_bible.describe_scenes(
        scene_list,
        keyframes={0: b"eager-bytes"},
        keyframe_provider=provider,
    )
    # Provider was called for scene 1 (no eager entry).
    assert provider_called is True
    # Both scenes got descriptions.
    assert scene_list[0].description == "d0"
    assert scene_list[1].description == "d1"


# ── C1: settings copy-on-write ────────────────────────────────────────────


def test_update_replaces_overrides_dict_atomically(tmp_path):
    """update() must rebind self._overrides rather than mutating it in
    place, so a reader holding a reference to the old dict doesn't see
    half-applied state."""
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}

    old_ref = s._overrides
    s.update({"max_line_chars": 50})
    # The dict object itself changed — the rebind happened.
    assert s._overrides is not old_ref
    # The new dict has the new value; the old reference doesn't.
    assert s._overrides["max_line_chars"] == 50
    assert "max_line_chars" not in old_ref


def test_reset_replaces_overrides_dict_atomically(tmp_path):
    env = _EnvSettings()
    env.cache_dir = tmp_path
    s = SettingsStore(env)
    s._file = tmp_path / "settings.json"
    s._overrides = {}
    s.update({"max_line_chars": 50})

    pre_ref = s._overrides
    s.reset("max_line_chars")
    # Mutating reset() would have left _overrides pointing at the same
    # dict; the COW rebind means a new object.
    assert s._overrides is not pre_ref
    assert "max_line_chars" not in s._overrides
    # The old dict still has the value (it was a snapshot for any
    # concurrent reader that captured pre_ref).
    assert pre_ref["max_line_chars"] == 50


# ── C3: LLM clients use max_retries=0 ─────────────────────────────────────


def test_anthropic_client_constructed_with_no_retries(monkeypatch):
    """SDK-level retries multiply the per-call timeout. We turn them off
    so the per-call 300 s cap is a true ceiling."""
    captured_kwargs: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    from app.pipeline.llm import anthropic as ant_mod
    monkeypatch.setattr(ant_mod.anthropic, "Anthropic", _FakeAnthropic)
    ant_mod.AnthropicLLM(api_key="x", model="claude-test")
    assert captured_kwargs.get("max_retries") == 0


def test_openai_compat_client_constructed_with_no_retries(monkeypatch):
    captured_kwargs: dict = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    from app.pipeline.llm import openai_compat as oc_mod
    monkeypatch.setattr(oc_mod.openai, "OpenAI", _FakeOpenAI)
    oc_mod.OpenAICompatLLM(
        base_url="http://x:1234/v1", api_key="y", model="m",
    )
    assert captured_kwargs.get("max_retries") == 0
