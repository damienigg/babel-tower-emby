"""Tests for the intermediate transcript cache (app/transcript_cache.py).

What this layer buys us: a translation-phase crash (OOM, transient
API error, container restart) no longer wastes the 30+ minutes of
Whisper work that came before it. The next retry hits this cache and
jumps straight to translation.

Pinned behaviors:

- Round-trip preserves cue ids / timings / text / detected language.
- Key includes only STT-relevant fields: changing model / backend /
  vad / track invalidates; changing target_lang / provider / mode
  DOES NOT (those don't affect the transcript).
- Empty-cue results are NOT cached (we don't want to memoize a
  "no speech detected" outcome — the user might fix audio routing
  and retry).
- Corrupted cache files quarantine to .corrupt on load, returning
  None so the pipeline cleanly re-transcribes.
- Atomic write — .tmp sidecar cleaned up on success.
"""
import json
from pathlib import Path

import pytest

from app import transcript_cache
from app.pipeline.stt import Cue, TranscriptionResult


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Isolate each test under tmp_path/cache.

    Belt-and-suspenders: also strip any pre-existing ``cache_dir``
    instance attribute on ``settings`` that a prior test (in this
    file or another) may have left behind via
    ``settings.cache_dir = X``. Without that strip, the instance
    attribute permanently shadows ``_overrides`` and the redirect
    here silently fails — every test reads/writes the polluted dir
    and order-dependent cross-test leakage starts to show up
    (an _vi0 file from one test surviving into another test's
    cache_dir, etc.)."""
    from app.config import settings
    if "cache_dir" in settings.__dict__:
        monkeypatch.delattr(settings, "cache_dir", raising=False)
    cdir = tmp_path / "cache"
    cdir.mkdir()
    monkeypatch.setattr(
        settings, "_overrides",
        {**settings._overrides, "cache_dir": str(cdir)},
    )
    return cdir


def _make_result(detected="en", n_cues=3) -> TranscriptionResult:
    cues = [
        Cue(id=i, start=float(i * 2), end=float(i * 2 + 1.5), text=f"hello {i}")
        for i in range(n_cues)
    ]
    return TranscriptionResult(detected_language=detected, cues=cues)


# ── Round-trip ────────────────────────────────────────────────────────────


def test_lookup_returns_none_when_cache_empty(cache_dir):
    """First call ever — no file, return None cleanly."""
    result = transcript_cache.lookup("fp1", "small", "openvino", True, 0)
    assert result is None


def test_store_then_lookup_returns_same_data(cache_dir):
    """The bread-and-butter case: write, then read back, get the same
    cues with the same timings + text + language."""
    src = _make_result(detected="fr", n_cues=5)
    transcript_cache.store("fp1", "small", "openvino", True, 0, src)

    got = transcript_cache.lookup("fp1", "small", "openvino", True, 0)
    assert got is not None
    assert got.detected_language == "fr"
    assert len(got.cues) == 5
    for orig, restored in zip(src.cues, got.cues):
        assert restored.id == orig.id
        assert restored.start == orig.start
        assert restored.end == orig.end
        assert restored.text == orig.text


# ── Key invalidation: STT-relevant fields ─────────────────────────────────


def test_changing_whisper_model_invalidates_cache(cache_dir):
    """small vs medium vs large produce different cue lists — must miss."""
    transcript_cache.store("fp1", "small", "openvino", True, 0, _make_result())
    assert transcript_cache.lookup("fp1", "medium", "openvino", True, 0) is None


def test_changing_backend_invalidates_cache(cache_dir):
    """openvino and faster-whisper can differ on edge cases — must miss."""
    transcript_cache.store("fp1", "small", "openvino", True, 0, _make_result())
    assert transcript_cache.lookup("fp1", "small", "cpu", True, 0) is None


def test_toggling_vad_invalidates_cache(cache_dir):
    """VAD on vs off changes the cue list materially (silence-region
    hallucinations get filtered out when VAD is on)."""
    transcript_cache.store("fp1", "small", "openvino", True, 0, _make_result())
    assert transcript_cache.lookup("fp1", "small", "openvino", False, 0) is None


def test_changing_track_index_invalidates_cache(cache_dir):
    """Different audio track = different transcript."""
    transcript_cache.store("fp1", "small", "openvino", True, 0, _make_result())
    assert transcript_cache.lookup("fp1", "small", "openvino", True, 1) is None


def test_changing_content_fingerprint_invalidates_cache(cache_dir):
    """Different media file (or its audio/video bytes changed) = miss."""
    transcript_cache.store("fp1", "small", "openvino", True, 0, _make_result())
    assert transcript_cache.lookup("fp2", "small", "openvino", True, 0) is None


def test_toggling_vocal_isolation_invalidates_cache(cache_dir):
    """Whisper sees a fundamentally different audio signal when
    Demucs runs first (vocals stem) vs. when the raw mix is fed in.
    A cache hit across the boundary would silently feed back the
    wrong transcript on toggle."""
    transcript_cache.store(
        "fp1", "small", "openvino", True, 0, _make_result(),
        vocal_isolation_enabled=True,
    )
    # vocal_isolation off → miss
    assert transcript_cache.lookup(
        "fp1", "small", "openvino", True, 0,
        vocal_isolation_enabled=False,
    ) is None
    # same flag → hit
    got = transcript_cache.lookup(
        "fp1", "small", "openvino", True, 0,
        vocal_isolation_enabled=True,
    )
    assert got is not None
    assert len(got.cues) == 3


# ── Empty results not cached ──────────────────────────────────────────────


def test_empty_transcription_not_cached(cache_dir):
    """A 'no speech detected' result shouldn't memoize — the user might
    fix the audio source and retry, and we want them to hit Whisper again."""
    empty = TranscriptionResult(detected_language="en", cues=[])
    transcript_cache.store("fp-empty-test", "small", "openvino", True, 0, empty)
    # lookup() for the exact key we just "stored" must miss — proves
    # the store() call short-circuited on the empty cue list.
    assert transcript_cache.lookup("fp-empty-test", "small", "openvino", True, 0) is None


# ── Corrupted file recovery ───────────────────────────────────────────────


def test_corrupted_cache_file_quarantines_and_returns_none(cache_dir):
    """A bad JSON file must NOT crash the pipeline — quarantine it to
    .corrupt and return None so the next run re-transcribes cleanly."""
    store_dir = transcript_cache._store_dir()
    store_dir.mkdir(parents=True, exist_ok=True)
    key = transcript_cache._key("fp1", "small", "openvino", True, 0)
    bad = store_dir / f"{key}.json"
    bad.write_text("not valid json {{{")

    result = transcript_cache.lookup("fp1", "small", "openvino", True, 0)
    assert result is None
    assert not bad.exists()
    assert bad.with_suffix(".corrupt").exists()


def test_missing_field_quarantines(cache_dir):
    """A JSON file missing required fields is also corruption — same
    quarantine behavior."""
    store_dir = transcript_cache._store_dir()
    store_dir.mkdir(parents=True, exist_ok=True)
    key = transcript_cache._key("fp1", "small", "openvino", True, 0)
    bad = store_dir / f"{key}.json"
    bad.write_text(json.dumps({"detected_language": "en"}))   # missing cues

    result = transcript_cache.lookup("fp1", "small", "openvino", True, 0)
    assert result is None
    assert bad.with_suffix(".corrupt").exists()


# ── Atomic write ──────────────────────────────────────────────────────────


def test_store_leaves_no_tmp_file(cache_dir):
    """The .tmp sidecar must be renamed away on success — its presence
    after a write would indicate a half-finished state."""
    transcript_cache.store("fp1", "small", "openvino", True, 0, _make_result())
    store_dir = transcript_cache._store_dir()
    assert list(store_dir.glob("*.tmp")) == []


def test_store_swallows_io_errors(cache_dir, monkeypatch):
    """If json.dump fails mid-write, we don't crash the pipeline.
    Best-effort persistence — the in-process transcription is still
    valid even if we can't write it to disk."""
    def boom(*a, **kw):
        raise RuntimeError("disk fail")
    monkeypatch.setattr(transcript_cache.json, "dump", boom)
    transcript_cache.store("fp1", "small", "openvino", True, 0, _make_result())   # no raise


# ── Key composition stability ─────────────────────────────────────────────


def test_key_includes_all_dimensions(cache_dir):
    """Keys for any pair of distinct (model, backend, vad, vocal_isolation,
    track, fp) must differ — important for correctness, and a regression
    here would silently return wrong transcripts."""
    keys = set()
    for fp in ("a", "b"):
        for model in ("small", "medium"):
            for backend in ("openvino", "cpu"):
                for vad in (True, False):
                    for vi in (True, False):
                        for t in (0, 1):
                            keys.add(transcript_cache._key(fp, model, backend, vad, t, vi))
    # 2 × 2 × 2 × 2 × 2 × 2 = 64 unique
    assert len(keys) == 64


def test_key_normalizes_slashes_in_model_name(cache_dir):
    """Hugging Face model ids contain slashes (e.g. facebook/nllb-…).
    The cache uses the key as a filename, so slashes must be normalized
    so we don't accidentally write into subdirectories."""
    key = transcript_cache._key("fp", "facebook/whisper-large", "openvino", True, 0)
    assert "/" not in key
