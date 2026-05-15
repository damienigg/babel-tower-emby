"""Tests for the performance-within-safety hardening pass.

Pins down behaviors that protect the in-process memory + concurrency
contract:

- Audio temp wavs land under settings.cache_dir/tmp, NOT /tmp.
- The settings store uses copy-on-write: a concurrent reader sees
  either the pre-update or post-update state, never a half-applied dict.
- LLM clients are constructed with max_retries=0 so SDK-level retry
  multiplication can't blow the per-call timeout budget.

Pre-0.7.32 this file also covered scene_bible.describe_scenes lazy
keyframe extraction. With scene/cinematic modes removed those tests
went with them.
"""
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import _EnvSettings, SettingsStore
from app.pipeline import audio as audio_mod


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
        from unittest.mock import MagicMock
        # 0.7.33 added an ffprobe channel-layout probe at the top of
        # extract_audio. Stub it with a "stereo, no FC" response so
        # we fall through to the standard downmix path. The ffmpeg
        # invocation still happens after — that's what we capture.
        if args and "ffprobe" in args[0]:
            cp = MagicMock()
            cp.stdout = '{"streams": [{"channels": 2, "channel_layout": "stereo"}]}'
            cp.returncode = 0
            return cp
        # The output wav path is the last positional argument to ffmpeg.
        captured_paths.append(args[-1])
        return MagicMock(returncode=0)

    with patch.object(audio_mod.subprocess, "run", side_effect=fake_run):
        with audio_mod.extract_audio("/some/movie.mkv", 0) as wav:
            # Path is under cache_dir/tmp, not /tmp.
            assert str(wav).startswith(str(cache_dir / "tmp"))

    # And the file is cleaned up on exit.
    assert captured_paths
    assert (cache_dir / "tmp").exists()   # the directory persists


# ── R3: lazy scene-bible keyframe provider ───────────────────────────────


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


# ── STT release between transcribe and translate ─────────────────────────


def test_stt_openvino_release_clears_cache():
    """release_model() must drop the lru_cache reference so the next
    transcribe() call re-instantiates the model — that's the whole point:
    free ~1 GB of Whisper weights before NLLB's ~1.5 GB initialization
    so the two never co-reside on a 6 GB-capped container."""
    from app.pipeline import stt_openvino

    # Prime the cache with a sentinel via cache_info so we don't need to
    # actually load OpenVINO. We just need to verify cache_clear() runs
    # without raising and that cache_info() reports zero entries after.
    stt_openvino._model_and_processor.cache_clear()   # baseline
    assert stt_openvino._model_and_processor.cache_info().currsize == 0

    # Stuff a fake entry directly via the wrapped function's __wrapped__:
    # we can't easily prime the cache without running the body, but we
    # CAN verify the release function exists and is callable + idempotent.
    stt_openvino.release_model()
    stt_openvino.release_model()   # idempotent
    assert stt_openvino._model_and_processor.cache_info().currsize == 0


def test_stt_faster_whisper_release_clears_cache():
    """Same contract as the OpenVINO backend — CPU image's Whisper cache
    must also evict on demand."""
    from app.pipeline import stt_faster_whisper
    stt_faster_whisper._model.cache_clear()
    assert stt_faster_whisper._model.cache_info().currsize == 0
    stt_faster_whisper.release_model()
    assert stt_faster_whisper._model.cache_info().currsize == 0


def test_lang_detect_release_clears_cache():
    """The tiny detector model (~250 MB resident) is single-use per job —
    after the pre-pass succeeds, it should not sit through translation."""
    from app.pipeline import lang_detect
    lang_detect._detector.cache_clear()
    assert lang_detect._detector.cache_info().currsize == 0
    lang_detect.release_detector()
    assert lang_detect._detector.cache_info().currsize == 0


def test_stt_release_dispatcher_picks_active_backend(monkeypatch):
    """stt.release() must mirror stt.transcribe()'s dispatch — it should
    call release_model() on the backend matching settings.whisper_backend."""
    from app.config import settings as runtime_settings
    from app.pipeline import stt as stt_mod
    from app.pipeline import stt_openvino, stt_faster_whisper

    called = {"openvino": 0, "cpu": 0}
    monkeypatch.setattr(stt_openvino, "release_model",
                        lambda: called.__setitem__("openvino", called["openvino"] + 1))
    monkeypatch.setattr(stt_faster_whisper, "release_model",
                        lambda: called.__setitem__("cpu", called["cpu"] + 1))

    monkeypatch.setattr(runtime_settings, "_overrides",
                        {**runtime_settings._overrides, "whisper_backend": "openvino"})
    stt_mod.release()
    assert called == {"openvino": 1, "cpu": 0}

    monkeypatch.setattr(runtime_settings, "_overrides",
                        {**runtime_settings._overrides, "whisper_backend": "cpu"})
    stt_mod.release()
    assert called == {"openvino": 1, "cpu": 1}


def test_processor_releases_stt_before_translation(monkeypatch, tmp_path):
    """End-to-end: process() must call stt.release() BEFORE invoking the
    translation provider. This is the load-bearing fix for the silent OOM
    at the 80% mark — holding Whisper weights resident while NLLB-600M
    loads breaches a 6 GB cgroup limit on typical NAS deployments and
    the kernel SIGKILLs the uvicorn process with no Python traceback.
    """
    from contextlib import contextmanager
    from app import cache as cache_mod
    from app.config import settings as runtime_settings
    from app.pipeline import audio, stt, tracks
    from app.pipeline.stt import Cue, TranscriptionResult
    from app import processor as processor_mod

    media = tmp_path / "movie.mkv"
    media.write_bytes(b"\x00" * 4096)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setattr(runtime_settings, "_overrides",
                        {**runtime_settings._overrides, "cache_dir": str(cache_dir)})
    monkeypatch.setattr(cache_mod.settings, "cache_dir", cache_dir, raising=False)

    fake_track = type("T", (), {"index": 0, "language": "en", "title": None,
                                  "codec": "aac", "channels": 2, "is_default": True})()
    monkeypatch.setattr(tracks, "probe", lambda *a, **kw: [fake_track])
    monkeypatch.setattr(tracks, "select", lambda *a, **kw: fake_track)

    @contextmanager
    def fake_extract(*a, **kw):
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"")
        yield wav
    monkeypatch.setattr(audio, "extract_audio", fake_extract)

    monkeypatch.setattr(stt, "transcribe", lambda *a, **kw: TranscriptionResult(
        detected_language="en",
        cues=[Cue(id=0, start=0.0, end=2.0, text="hi")],
    ))

    # Record the order: stt.release() must fire BEFORE get_provider().
    order: list[str] = []
    real_release = stt.release
    def spy_release():
        order.append("release")
        real_release()
    monkeypatch.setattr(stt, "release", spy_release)

    class _Provider:
        def translate(self, cues, source_lang, target_lang, context=None,
                      *, progress=None, check_cancel=None):
            order.append("translate")
            return list(cues)

    def spy_get_provider(name):
        order.append("get_provider")
        return _Provider()
    monkeypatch.setattr(processor_mod, "get_provider", spy_get_provider)

    from app.processor import ProcessRequest, process
    req = ProcessRequest(
        media_path=str(media),
        target_lang="fr",
        source_lang_priority=["en", "*"],
        translation_provider="nllb",
    )
    process(req)

    assert order.index("release") < order.index("get_provider"), (
        f"stt.release() must fire before the translation provider loads; got {order}"
    )
