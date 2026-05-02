from app.pipeline.tracks import AudioTrack, select


def _t(index, language=None, title=None, **flags):
    base = dict(
        index=index, language=language, title=title,
        is_default=False, is_dub=False, is_original=False,
        is_commentary=False, is_audio_description=False,
        channels=2,
    )
    base.update(flags)
    return AudioTrack(**base)


def test_select_prefers_priority_order():
    tracks = [_t(0, "fr"), _t(1, "en"), _t(2, "ja")]
    chosen = select(tracks, target_lang="es", source_priority=["en", "ja", "fr"], skip_if_target_audio_exists=True)
    assert chosen.index == 1


def test_select_wildcard_in_priority_picks_anything():
    tracks = [_t(0, "ko")]
    chosen = select(tracks, target_lang="fr", source_priority=["*"], skip_if_target_audio_exists=True)
    assert chosen.index == 0


def test_select_skips_when_target_lang_audio_exists():
    tracks = [_t(0, "en"), _t(1, "fr")]
    chosen = select(tracks, target_lang="fr", source_priority=["en"], skip_if_target_audio_exists=True)
    assert chosen is None


def test_select_does_not_skip_when_override():
    tracks = [_t(0, "en"), _t(1, "fr")]
    chosen = select(tracks, target_lang="fr", source_priority=["en"], skip_if_target_audio_exists=False)
    assert chosen.index == 0


def test_select_filters_out_commentary_and_audio_description():
    tracks = [
        _t(0, "en", title="Director Commentary", is_commentary=True),
        _t(1, "en", title="Audio Description", is_audio_description=True),
        _t(2, "en", title="Main"),
    ]
    chosen = select(tracks, target_lang="fr", source_priority=["en"], skip_if_target_audio_exists=True)
    assert chosen.index == 2


def test_select_prefers_original_over_dub():
    tracks = [
        _t(0, "en", is_dub=True),
        _t(1, "en", is_original=True),
    ]
    chosen = select(tracks, target_lang="fr", source_priority=["en"], skip_if_target_audio_exists=True)
    assert chosen.index == 1


def test_select_returns_none_when_only_junk():
    tracks = [
        _t(0, "en", is_commentary=True),
        _t(1, "en", is_audio_description=True),
    ]
    assert select(tracks, target_lang="fr", source_priority=["en"], skip_if_target_audio_exists=True) is None
