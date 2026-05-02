from app.emby.client import EmbyError, EmbyItem, _to_iso639_2


def _streams_with_subtitle(lang):
    return [{"Type": "Subtitle", "Language": lang}]


def test_iso_639_2_mapping():
    assert _to_iso639_2("en") == "eng"
    assert _to_iso639_2("fr") == "fra"
    assert _to_iso639_2("ja") == "jpn"


def test_iso_639_2_passthrough_for_unknown():
    assert _to_iso639_2("unknown") == "unknown"


def test_has_subtitle_track_two_letter_match():
    item = EmbyItem(id="1", name="x", path="/x.mkv", type="Movie",
                    media_streams=_streams_with_subtitle("en"))
    assert item.has_subtitle_track("en") is True
    assert item.has_subtitle_track("fr") is False


def test_has_subtitle_track_three_letter_match():
    """Emby commonly tags subs with ISO 639-2 codes ('eng' not 'en')."""
    item = EmbyItem(id="1", name="x", path="/x.mkv", type="Movie",
                    media_streams=_streams_with_subtitle("eng"))
    assert item.has_subtitle_track("en") is True


def test_has_subtitle_track_case_insensitive():
    item = EmbyItem(id="1", name="x", path="/x.mkv", type="Movie",
                    media_streams=_streams_with_subtitle("ENG"))
    assert item.has_subtitle_track("en") is True


def test_has_subtitle_track_no_subs():
    item = EmbyItem(id="1", name="x", path="/x.mkv", type="Movie", media_streams=[])
    assert item.has_subtitle_track("en") is False


def test_has_subtitle_track_only_audio_streams():
    item = EmbyItem(id="1", name="x", path="/x.mkv", type="Movie",
                    media_streams=[{"Type": "Audio", "Language": "en"}])
    assert item.has_subtitle_track("en") is False


def test_emby_error_on_missing_creds():
    from app.emby.client import EmbyClient
    import pytest
    with pytest.raises(EmbyError):
        EmbyClient("", "")
    with pytest.raises(EmbyError):
        EmbyClient("http://x", "")
