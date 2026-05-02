from app.pipeline.scenes import Scene, keyframe_timestamp, map_cues_to_scenes
from app.pipeline.stt import Cue


def test_keyframe_midpoint():
    s = Scene(index=0, start=10.0, end=20.0)
    assert keyframe_timestamp(s, "midpoint") == 15.0


def test_keyframe_start_offsets_past_cut():
    s = Scene(index=0, start=10.0, end=20.0)
    # Nudges past the cut to avoid grabbing the previous shot
    assert keyframe_timestamp(s, "start") > 10.0
    assert keyframe_timestamp(s, "start") <= 11.0


def test_keyframe_end_clamps_to_start():
    s = Scene(index=0, start=10.0, end=10.05)
    # Even a tiny scene must not produce ts < start
    assert keyframe_timestamp(s, "end") >= 10.0


def test_map_cues_to_scenes_basic():
    scenes = [
        Scene(index=0, start=0.0, end=10.0),
        Scene(index=1, start=10.0, end=20.0),
        Scene(index=2, start=20.0, end=30.0),
    ]
    cues = [
        Cue(id=0, start=2.0, end=3.0, text="a"),    # in scene 0
        Cue(id=1, start=11.0, end=12.0, text="b"),  # in scene 1
        Cue(id=2, start=29.5, end=30.5, text="c"),  # last scene
    ]
    mapping = map_cues_to_scenes(cues, scenes)
    assert mapping == {0: 0, 1: 1, 2: 2}


def test_map_cues_falls_back_to_last_scene_for_overrun():
    scenes = [Scene(index=0, start=0.0, end=10.0)]
    cues = [Cue(id=5, start=99.0, end=100.0, text="late")]
    assert map_cues_to_scenes(cues, scenes) == {5: 0}


def test_map_cues_empty_scenes_returns_empty():
    cues = [Cue(id=0, start=1.0, end=2.0, text="x")]
    assert map_cues_to_scenes(cues, []) == {}
