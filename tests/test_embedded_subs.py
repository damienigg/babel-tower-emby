"""Tests for the embedded-subtitle short-circuit (0.10.0+).

These cover the pure logic that doesn't need a real video file:
- ffprobe JSON → SubtitleTrack list
- Track-picking priority (target_lang > source_lang > en > first)
- Forced/SDH/bitmap handling

The end-to-end "STT was skipped because subs were embedded" assertion
lives in test_processor.py — that one mocks the embedded_subs module
directly rather than fighting an actual ffprobe.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.pipeline import embedded_subs


def _make_track(
    *,
    index=0,
    sub_pos=0,
    codec="subrip",
    language=None,
    title=None,
    is_default=False,
    is_forced=False,
    is_hearing_impaired=False,
):
    return embedded_subs.SubtitleTrack(
        index=index,
        track_index_within_subs=sub_pos,
        codec=codec,
        language=language,
        title=title,
        is_default=is_default,
        is_forced=is_forced,
        is_hearing_impaired=is_hearing_impaired,
    )


# ── list_subtitle_tracks: ffprobe JSON parsing ─────────────────────────────


def test_list_tracks_empty_file_returns_empty_list():
    fake_proc = type("P", (), {"stdout": '{"streams": []}'})()
    with patch.object(embedded_subs.subprocess, "run", return_value=fake_proc):
        tracks = embedded_subs.list_subtitle_tracks("/whatever.mkv")
    assert tracks == []


def test_list_tracks_parses_text_and_bitmap_tracks():
    streams = {
        "streams": [
            {
                "index": 2,
                "codec_name": "subrip",
                "tags": {"language": "eng", "title": "English"},
                "disposition": {"default": 1, "forced": 0, "hearing_impaired": 0},
            },
            {
                "index": 3,
                "codec_name": "subrip",
                "tags": {"language": "fre", "title": "Français (SDH)"},
                "disposition": {"default": 0, "forced": 0, "hearing_impaired": 1},
            },
            {
                "index": 4,
                "codec_name": "hdmv_pgs_subtitle",
                "tags": {"language": "eng"},
                "disposition": {"default": 0, "forced": 0, "hearing_impaired": 0},
            },
        ]
    }
    fake_proc = type("P", (), {"stdout": json.dumps(streams)})()
    with patch.object(embedded_subs.subprocess, "run", return_value=fake_proc):
        tracks = embedded_subs.list_subtitle_tracks("/movie.mkv")
    assert len(tracks) == 3
    # 3-letter ISO → 2-letter normalized
    assert tracks[0].language == "en"
    assert tracks[0].is_text is True
    assert tracks[0].is_bitmap is False
    assert tracks[0].is_default is True
    assert tracks[1].language == "fr"
    assert tracks[1].is_hearing_impaired is True
    assert tracks[2].codec == "hdmv_pgs_subtitle"
    assert tracks[2].is_text is False
    assert tracks[2].is_bitmap is True
    # sub_pos must be 0-based, regardless of absolute stream index
    assert [t.track_index_within_subs for t in tracks] == [0, 1, 2]


def test_list_tracks_tolerates_missing_tags_and_disposition():
    streams = {"streams": [{"index": 2, "codec_name": "subrip"}]}
    fake_proc = type("P", (), {"stdout": json.dumps(streams)})()
    with patch.object(embedded_subs.subprocess, "run", return_value=fake_proc):
        tracks = embedded_subs.list_subtitle_tracks("/movie.mkv")
    assert len(tracks) == 1
    assert tracks[0].language is None
    assert tracks[0].title is None
    assert tracks[0].is_default is False


def test_list_tracks_lowercases_codec():
    streams = {
        "streams": [
            {"index": 2, "codec_name": "SubRip", "tags": {"language": "eng"}},
        ]
    }
    fake_proc = type("P", (), {"stdout": json.dumps(streams)})()
    with patch.object(embedded_subs.subprocess, "run", return_value=fake_proc):
        tracks = embedded_subs.list_subtitle_tracks("/movie.mkv")
    assert tracks[0].codec == "subrip"
    assert tracks[0].is_text is True


# ── pick_best_track: priority + filters ────────────────────────────────────


def test_pick_returns_none_when_no_text_tracks():
    tracks = [_make_track(codec="hdmv_pgs_subtitle", language="en")]
    assert embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en") is None


def test_pick_returns_none_when_only_forced_tracks():
    tracks = [_make_track(language="en", is_forced=True)]
    assert embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en") is None


def test_pick_prefers_target_lang_over_source_lang():
    tracks = [
        _make_track(index=2, sub_pos=0, language="en", title="English"),
        _make_track(index=3, sub_pos=1, language="fr", title="Français"),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en")
    assert chosen.language == "fr"


def test_pick_prefers_source_lang_when_target_absent():
    tracks = [
        _make_track(index=2, sub_pos=0, language="es", title="Spanish"),
        _make_track(index=3, sub_pos=1, language="en", title="English"),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en")
    assert chosen.language == "en"


def test_pick_prefers_english_when_target_and_source_absent():
    tracks = [
        _make_track(index=2, sub_pos=0, language="es"),
        _make_track(index=3, sub_pos=1, language="en"),
        _make_track(index=4, sub_pos=2, language="it"),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="ja")
    assert chosen.language == "en"


def test_pick_falls_through_to_first_text_track_when_no_priority_match():
    tracks = [
        _make_track(index=2, sub_pos=0, language="ja"),
        _make_track(index=3, sub_pos=1, language="ko"),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en")
    # Both at tier 3 — break by stream index
    assert chosen.index == 2


def test_pick_prefers_non_sdh_within_same_language():
    tracks = [
        _make_track(index=2, sub_pos=0, language="fr", is_hearing_impaired=True),
        _make_track(index=3, sub_pos=1, language="fr", is_hearing_impaired=False),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en")
    assert chosen.index == 3


def test_pick_skips_forced_track_picks_full():
    tracks = [
        _make_track(index=2, sub_pos=0, language="fr", is_forced=True),
        _make_track(index=3, sub_pos=1, language="en"),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en")
    assert chosen.language == "en"
    assert chosen.index == 3


def test_pick_handles_none_source_lang():
    tracks = [
        _make_track(index=2, sub_pos=0, language="de"),
        _make_track(index=3, sub_pos=1, language="en"),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang=None)
    # source_lang=None → en wins because tier 2 (English) beats tier 3 (other)
    assert chosen.language == "en"


def test_pick_prefers_default_disposition_within_language_tier():
    tracks = [
        _make_track(index=2, sub_pos=0, language="en", is_default=False),
        _make_track(index=3, sub_pos=1, language="en", is_default=True),
    ]
    chosen = embedded_subs.pick_best_track(tracks, target_lang="fr", source_lang="en")
    assert chosen.index == 3
    assert chosen.is_default is True


# ── EmbeddedSubsMetrics: defaults + fields ─────────────────────────────────


def test_metrics_defaults_are_safe():
    from app.pipeline_metrics import EmbeddedSubsMetrics
    m = EmbeddedSubsMetrics()
    assert m.tracks_detected == 0
    assert m.chosen_track_index is None
    assert m.action == "fallback_no_tracks"


def test_metrics_serializes_via_pipeline_metrics_jsonable():
    from app.pipeline_metrics import EmbeddedSubsMetrics, PipelineMetrics, to_jsonable
    m = PipelineMetrics(embedded_subs=EmbeddedSubsMetrics(
        tracks_detected=2,
        text_tracks_count=1,
        bitmap_tracks_count=1,
        chosen_track_index=3,
        chosen_codec="subrip",
        chosen_lang="fr",
        chosen_title="Français",
        action="copy_same_lang",
    ))
    d = to_jsonable(m)
    assert d["embedded_subs"]["action"] == "copy_same_lang"
    assert d["embedded_subs"]["chosen_lang"] == "fr"
    assert d["embedded_subs"]["tracks_detected"] == 2


# ── Processor integration: embedded subs short-circuit STT ─────────────────


def test_processor_copy_mode_skips_stt_and_translation(monkeypatch, tmp_path):
    """Target-language text track in source → STT and translation are both
    skipped. The .vtt comes straight from the embedded track.

    This is the headline value prop: a film with a pro-authored FR sub
    track and target_lang=fr returns near-instantaneously with copied
    cues, no Whisper, no NLLB.
    """
    from app import cache as cache_mod
    from app import processor as processor_mod
    from app.config import settings as runtime_settings
    from app.pipeline import audio, embedded_subs as es_mod, stt, tracks
    from pathlib import Path

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

    fake_sub = es_mod.SubtitleTrack(
        index=2, track_index_within_subs=0, codec="subrip",
        language="fr", title="Français",
        is_default=True, is_forced=False, is_hearing_impaired=False,
    )
    monkeypatch.setattr(es_mod, "list_subtitle_tracks", lambda p: [fake_sub])
    monkeypatch.setattr(es_mod, "pick_best_track", lambda *a, **kw: fake_sub)

    extracted_vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nBonjour.\n\n"
        "00:00:05.000 --> 00:00:07.000\nLe monde.\n"
    )
    def fake_extract(media_path, track, output_dir=None):
        out = tmp_path / "out.vtt"
        out.write_text(extracted_vtt, encoding="utf-8")
        return out
    monkeypatch.setattr(es_mod, "extract_track", fake_extract)
    monkeypatch.setattr(es_mod, "cleanup_extract_dir", lambda p: None)

    # Trap STT + audio extraction + translation — none of them should fire.
    trap = {"audio": 0, "stt": 0, "translate": 0}
    from contextlib import contextmanager
    @contextmanager
    def trap_audio(*a, **kw):
        trap["audio"] += 1
        yield tmp_path / "ignored.wav"
    def trap_stt(*a, **kw):
        trap["stt"] += 1
        raise AssertionError("STT must not run on embedded-copy path")
    monkeypatch.setattr(audio, "extract_audio", trap_audio)
    monkeypatch.setattr(stt, "transcribe", trap_stt)

    class TrapProvider:
        def translate(self, *a, **kw):
            trap["translate"] += 1
            raise AssertionError("translation must not run on embedded-copy path")
    monkeypatch.setattr(processor_mod, "get_provider", lambda name: TrapProvider())

    req = processor_mod.ProcessRequest(
        media_path=str(media), target_lang="fr",
        source_lang_priority=["en", "*"],
        translation_provider="nllb",
        prefer_embedded_subs=True,
    )
    result = processor_mod.process(req)

    assert trap["audio"] == 0, "audio extraction must not run in copy mode"
    assert trap["stt"] == 0,   "STT must not run in copy mode"
    assert trap["translate"] == 0, "translation must not run in copy mode"
    assert result.cue_count == 2
    assert result.detected_source_language == "fr"
    assert result.pipeline_metrics is not None
    es = result.pipeline_metrics.get("embedded_subs") or {}
    assert es.get("action") == "copy_same_lang"
    assert es.get("chosen_lang") == "fr"
    assert "Bonjour" in result.vtt
    assert "Le monde" in result.vtt


def test_processor_translate_mode_skips_stt_only(monkeypatch, tmp_path):
    """Non-target text track (e.g. English sub in source, user wants FR) →
    STT is skipped but translation runs on the extracted cues.

    Asserts: audio extraction + Whisper are NOT called, translation IS
    called, and the metrics record the right action.
    """
    from app import cache as cache_mod
    from app import processor as processor_mod
    from app.config import settings as runtime_settings
    from app.pipeline import audio, embedded_subs as es_mod, stt, tracks

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

    fake_sub = es_mod.SubtitleTrack(
        index=2, track_index_within_subs=0, codec="subrip",
        language="en", title="English",
        is_default=True, is_forced=False, is_hearing_impaired=False,
    )
    monkeypatch.setattr(es_mod, "list_subtitle_tracks", lambda p: [fake_sub])
    monkeypatch.setattr(es_mod, "pick_best_track", lambda *a, **kw: fake_sub)

    extracted_vtt = (
        "WEBVTT\n\n"
        "00:00:01.000 --> 00:00:03.000\nHello.\n\n"
        "00:00:05.000 --> 00:00:07.000\nThe world.\n"
    )
    def fake_extract(media_path, track, output_dir=None):
        out = tmp_path / "out.vtt"
        out.write_text(extracted_vtt, encoding="utf-8")
        return out
    monkeypatch.setattr(es_mod, "extract_track", fake_extract)
    monkeypatch.setattr(es_mod, "cleanup_extract_dir", lambda p: None)

    trap = {"audio": 0, "stt": 0, "translate": 0}
    from contextlib import contextmanager
    @contextmanager
    def trap_audio(*a, **kw):
        trap["audio"] += 1
        yield tmp_path / "ignored.wav"
    def trap_stt(*a, **kw):
        trap["stt"] += 1
        raise AssertionError("STT must not run when embedded text track is available")
    monkeypatch.setattr(audio, "extract_audio", trap_audio)
    monkeypatch.setattr(stt, "transcribe", trap_stt)

    class FakeProvider:
        def translate(self, cues, source_lang, target_lang,
                      *, progress=None, check_cancel=None):
            trap["translate"] += 1
            assert source_lang == "en"
            assert target_lang == "fr"
            # Echo the cues with translated text so the pipeline can
            # proceed.
            from app.pipeline.stt import Cue
            return [Cue(id=c.id, start=c.start, end=c.end, text=f"FR:{c.text}")
                    for c in cues]
    monkeypatch.setattr(processor_mod, "get_provider", lambda name: FakeProvider())

    req = processor_mod.ProcessRequest(
        media_path=str(media), target_lang="fr",
        source_lang_priority=["en", "*"],
        translation_provider="nllb",
        prefer_embedded_subs=True,
    )
    result = processor_mod.process(req)

    assert trap["audio"] == 0, "audio extraction must be skipped"
    assert trap["stt"] == 0,   "STT must be skipped"
    assert trap["translate"] == 1, "translation must run on extracted cues"
    es = (result.pipeline_metrics or {}).get("embedded_subs") or {}
    assert es.get("action") == "translate_other_lang"
    assert es.get("chosen_lang") == "en"
    assert "FR:Hello" in result.vtt
