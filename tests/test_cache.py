import json
from pathlib import Path

from app import cache


def test_file_fingerprint_stable(tmp_path):
    p = tmp_path / "movie.mkv"
    p.write_bytes(b"x" * 100)
    f1 = cache.file_fingerprint(p)
    f2 = cache.file_fingerprint(p)
    assert f1 == f2


def test_file_fingerprint_changes_on_mtime(tmp_path):
    import os, time
    p = tmp_path / "movie.mkv"
    p.write_bytes(b"x")
    f1 = cache.file_fingerprint(p)
    # Bump mtime by 1 second
    new_time = p.stat().st_mtime + 1
    os.utime(p, (new_time, new_time))
    f2 = cache.file_fingerprint(p)
    assert f1 != f2


def test_cache_key_is_deterministic():
    args = ("fp123", "fr", "small", "llm", ["en", "ja"], "audio")
    assert cache.cache_key(*args) == cache.cache_key(*args)


def test_cache_key_includes_threshold_only_when_provided():
    base = cache.cache_key("fp", "fr", "small", "llm", ["en"], "audio")
    with_threshold = cache.cache_key("fp", "fr", "small", "llm", ["en"], "scene", scene_threshold=0.4)
    assert base != with_threshold

    same_threshold = cache.cache_key("fp", "fr", "small", "llm", ["en"], "scene", scene_threshold=0.4)
    diff_threshold = cache.cache_key("fp", "fr", "small", "llm", ["en"], "scene", scene_threshold=0.5)
    assert same_threshold != diff_threshold


def test_cache_load_missing_returns_none(tmp_path, monkeypatch):
    from app.config import settings as _settings
    monkeypatch.setattr(_settings._env, "cache_dir", tmp_path)
    assert cache.load("nonexistent-key") is None


def test_cache_load_corrupt_json_returns_none(tmp_path, monkeypatch):
    from app.config import settings as _settings
    monkeypatch.setattr(_settings._env, "cache_dir", tmp_path)
    (tmp_path / "broken.json").write_text("{not valid json")
    assert cache.load("broken") is None


def test_cache_store_and_load_roundtrip(tmp_path, monkeypatch):
    from app.config import settings as _settings
    monkeypatch.setattr(_settings._env, "cache_dir", tmp_path)
    payload = {"vtt": "WEBVTT\n\nfoo", "cue_count": 1}
    cache.store("k1", payload)
    assert cache.load("k1") == payload
