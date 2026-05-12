"""Tests for app/cache_explorer.py.

The explorer is read-side + surgical-delete code that the UI calls into.
The load-bearing behaviors it must hold:

- VTT entries with media_path (0.7.4+) surface the film name; older
  entries fall back to parsing the .vtt NOTE header for lang/mode/
  provider/whisper and to a truncated first-cue preview.
- Transcript entries decode the v2 key shape into per-axis fields
  (backend / model / vad / track). v1 keys (no schema prefix) still
  list — they just don't decode.
- Delete refuses anything that isn't a safe cache key (path
  traversal defense). It also refuses to touch runtime files
  (settings.json, jobs.json) even when given a key that would
  resolve to one.
- list_vtt_entries skips the transcripts/ subdir and the runtime
  files, so a stray "Delete" in the UI can't nuke them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app import cache_explorer


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Each test gets a clean cache dir with the same layout the running
    container uses: top-level VTT cache + transcripts/ subdir."""
    from app.config import settings as runtime_settings
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "transcripts").mkdir()
    monkeypatch.setattr(
        runtime_settings, "_overrides",
        {**runtime_settings._overrides, "cache_dir": str(cache_dir)},
    )
    return cache_dir


def _write_vtt_entry(cache_root: Path, key: str, *, media_path: str | None,
                     vtt_body: str, mode: str = "audio",
                     cue_count: int = 100) -> Path:
    payload: dict = {
        "vtt": vtt_body,
        "source_track": {"index": 0, "language": "eng", "title": None},
        "detected_source_language": "en",
        "cue_count": cue_count,
        "mode": mode,
    }
    if media_path is not None:
        payload["media_path"] = media_path
    p = cache_root / f"{key}.json"
    p.write_text(json.dumps(payload))
    return p


def _write_transcript_entry(cache_root: Path, key: str, *,
                            detected_language: str = "en",
                            cue_count: int = 50) -> Path:
    payload = {
        "detected_language": detected_language,
        "cues": [{"id": i, "start": float(i), "end": float(i + 1),
                  "text": f"line {i}"} for i in range(cue_count)],
    }
    p = cache_root / "transcripts" / f"{key}.json"
    p.write_text(json.dumps(payload))
    return p


# ── VTT cache listing ──────────────────────────────────────────────────────


def test_list_vtt_extracts_media_name_from_payload(cache_root):
    _write_vtt_entry(
        cache_root, "abc123def4567890",
        media_path="/mnt/films/Inception (2010).mkv",
        vtt_body=(
            "WEBVTT\n\n"
            "NOTE Subtitle This auto-subs (en -> fr, mode=audio, "
            "whisper=large-v3-turbo, provider=nllb)\n\n"
            "00:01:00.000 --> 00:01:02.000\nHello world\n"
        ),
    )

    entries = cache_explorer.list_vtt_entries()

    assert len(entries) == 1
    e = entries[0]
    assert e.cache_key == "abc123def4567890"
    assert e.media_path == "/mnt/films/Inception (2010).mkv"
    assert e.media_name == "Inception (2010).mkv"
    assert e.source_lang == "en"
    assert e.target_lang == "fr"
    assert e.mode == "audio"
    assert e.whisper_model == "large-v3-turbo"
    assert e.provider == "nllb"
    assert e.cue_count == 100
    assert e.preview == "Hello world"


def test_list_vtt_legacy_entry_without_media_path_falls_back_to_note_line(cache_root):
    """Pre-0.7.4 entries don't carry media_path. The row still needs to
    render with the lang/mode/provider it can parse, plus a text preview
    so the user can recognize the film."""
    _write_vtt_entry(
        cache_root, "legacy0000000000",
        media_path=None,
        mode="scene",   # payload's mode field still present in legacy entries
        vtt_body=(
            "WEBVTT\n\n"
            "NOTE Subtitle This auto-subs (en -> fr, mode=scene, "
            "whisper=small, provider=deepl)\n\n"
            "00:00:10.000 --> 00:00:12.000\nVous m'attendiez longtemps\n"
        ),
    )

    e = cache_explorer.list_vtt_entries()[0]

    assert e.media_path is None
    assert e.media_name is None
    assert e.source_lang == "en"
    assert e.target_lang == "fr"
    assert e.mode == "scene"
    assert e.whisper_model == "small"
    assert e.provider == "deepl"
    assert e.preview == "Vous m'attendiez longtemps"


def test_list_vtt_corrupt_file_still_renders_a_delete_only_row(cache_root):
    """A garbled cache file mustn't crash the page — the row appears so
    the user can delete it from the UI."""
    (cache_root / "broken00000000.json").write_text("not valid json {{{")

    entries = cache_explorer.list_vtt_entries()

    assert len(entries) == 1
    assert entries[0].cache_key == "broken00000000"
    assert entries[0].media_name is None
    assert entries[0].size_bytes > 0


def test_list_vtt_skips_runtime_files_and_subdirs(cache_root):
    """settings.json / jobs.json share the cache_dir but aren't subtitle
    artefacts; transcripts/ has its own section. Both must be invisible
    here so a stray "Delete" in the UI can't nuke runtime state."""
    (cache_root / "settings.json").write_text("{}")
    (cache_root / "jobs.json").write_text("[]")
    _write_transcript_entry(cache_root, "v2_xyz_openvino_small_vad1_t0")
    _write_vtt_entry(cache_root, "real0000", media_path="/m/x.mkv",
                     vtt_body="WEBVTT\n")

    entries = cache_explorer.list_vtt_entries()

    keys = {e.cache_key for e in entries}
    assert keys == {"real0000"}


def test_list_vtt_sorts_newest_first(cache_root):
    p1 = _write_vtt_entry(cache_root, "old00000", media_path="/m/a.mkv",
                          vtt_body="WEBVTT\n")
    p2 = _write_vtt_entry(cache_root, "new00000", media_path="/m/b.mkv",
                          vtt_body="WEBVTT\n")
    # Force a deterministic mtime ordering — file creation order isn't
    # enough on filesystems with 1s mtime resolution.
    import os
    now = time.time()
    os.utime(p1, (now - 3600, now - 3600))
    os.utime(p2, (now, now))

    entries = cache_explorer.list_vtt_entries()

    assert [e.cache_key for e in entries] == ["new00000", "old00000"]


# ── VTT cache delete ───────────────────────────────────────────────────────


def test_delete_vtt_removes_file(cache_root):
    p = _write_vtt_entry(cache_root, "todelete00", media_path="/m/x.mkv",
                         vtt_body="WEBVTT\n")
    assert p.exists()

    assert cache_explorer.delete_vtt_entry("todelete00") is True

    assert not p.exists()


def test_delete_vtt_returns_false_when_not_found(cache_root):
    assert cache_explorer.delete_vtt_entry("nothing00") is False


def test_delete_vtt_rejects_path_traversal(cache_root):
    """A handler that forwards a `..` value mustn't be able to climb out
    of cache_dir. _validate_cache_key catches it at the boundary."""
    with pytest.raises(ValueError):
        cache_explorer.delete_vtt_entry("../escape")
    with pytest.raises(ValueError):
        cache_explorer.delete_vtt_entry("foo/bar")
    with pytest.raises(ValueError):
        cache_explorer.delete_vtt_entry("")


def test_delete_vtt_refuses_runtime_files(cache_root):
    """Even with a syntactically valid key, the bare 'settings' / 'jobs'
    names match real runtime files in cache_dir. Belt-and-suspenders the
    runtime-file guard so the UI can never destroy them."""
    (cache_root / "settings.json").write_text("{}")
    with pytest.raises(ValueError):
        cache_explorer.delete_vtt_entry("settings")
    # And the file is untouched.
    assert (cache_root / "settings.json").exists()


def test_clear_all_vtt_removes_every_entry(cache_root):
    for k in ["a00", "b00", "c00"]:
        _write_vtt_entry(cache_root, k, media_path=f"/m/{k}.mkv",
                         vtt_body="WEBVTT\n")
    # Runtime file must survive.
    (cache_root / "settings.json").write_text("{}")

    n = cache_explorer.clear_all_vtt_entries()

    assert n == 3
    assert cache_explorer.list_vtt_entries() == []
    assert (cache_root / "settings.json").exists()


# ── Transcript cache ──────────────────────────────────────────────────────


def test_list_transcripts_decodes_v2_key_fields(cache_root):
    _write_transcript_entry(
        cache_root,
        "v2_aabbccddeeff_openvino_large-v3-turbo_vad1_t0",
        detected_language="en", cue_count=42,
    )

    e = cache_explorer.list_transcript_entries()[0]

    assert e.whisper_backend == "openvino"
    assert e.whisper_model == "large-v3-turbo"
    assert e.vad_enabled is True
    assert e.track_index == 0
    assert e.detected_language == "en"
    assert e.cue_count == 42
    assert e.parsed is True


def test_list_transcripts_marks_unparseable_v1_key(cache_root):
    """v1 keys (pre-0.7.2, no schema prefix) still list — they just
    don't decode into per-axis fields. The row is delete-only."""
    _write_transcript_entry(
        cache_root,
        "deadbeef_openvino_large-v3-turbo_vad1_t0",   # no v2_ prefix
        detected_language="fr", cue_count=10,
    )

    e = cache_explorer.list_transcript_entries()[0]

    assert e.parsed is False
    assert e.whisper_backend is None
    # The payload is still readable even when the key isn't.
    assert e.cue_count == 10


def test_delete_transcript_removes_file(cache_root):
    _write_transcript_entry(cache_root, "v2_xx_openvino_small_vad1_t0")

    assert cache_explorer.delete_transcript_entry(
        "v2_xx_openvino_small_vad1_t0"
    ) is True

    assert cache_explorer.list_transcript_entries() == []


def test_clear_all_transcripts(cache_root):
    for k in ["v2_a_openvino_small_vad1_t0",
              "v2_b_openvino_small_vad1_t0",
              "v2_c_openvino_small_vad1_t0"]:
        _write_transcript_entry(cache_root, k)

    n = cache_explorer.clear_all_transcript_entries()

    assert n == 3
    assert cache_explorer.list_transcript_entries() == []
