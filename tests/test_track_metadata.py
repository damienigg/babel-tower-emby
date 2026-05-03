"""Unit tests for the language tag write-back. Heavy externals (mkvpropedit,
ffmpeg, ffprobe) are mocked — we test the dispatch logic + error paths."""
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from app.pipeline import track_metadata


def _fake_ffprobe(audio_indices: list[int]):
    """Build a CompletedProcess that mimics ffprobe -select_streams a output."""
    return subprocess.CompletedProcess(
        args=[], returncode=0,
        stdout=json.dumps({"streams": [{"index": i} for i in audio_indices]}),
        stderr="",
    )


def _ok_proc():
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def _failed_proc(stderr: str = "boom"):
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


def test_unknown_language_raises_without_subprocessing(tmp_path):
    """When we have no ISO 639-2 mapping for the detected code, we fail
    fast before shelling out to anything."""
    f = tmp_path / "movie.mkv"
    f.write_bytes(b"\x00")
    with patch("subprocess.run") as run:
        with pytest.raises(track_metadata.MetadataWriteError, match="no ISO 639-2 mapping"):
            track_metadata.write_audio_language(f, 1, "xyz-not-a-real-lang")
        assert run.call_count == 0


def test_mkv_dispatches_to_mkvpropedit_with_audio_track_position(tmp_path):
    """For Matroska files, we should call mkvpropedit with the 1-based audio
    track position (NOT the absolute stream index)."""
    f = tmp_path / "movie.mkv"
    f.write_bytes(b"\x00")
    # File has video at index 0, then 3 audio streams at 1, 2, 3.
    # We want to tag the SECOND audio stream (absolute index 2 → position 2).
    calls = []

    def run_stub(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            return _fake_ffprobe([1, 2, 3])
        if cmd[0] == "mkvpropedit":
            return _ok_proc()
        raise AssertionError(f"unexpected subprocess call: {cmd[0]}")

    with patch("subprocess.run", side_effect=run_stub):
        track_metadata.write_audio_language(f, 2, "fr")

    mkv_call = next(c for c in calls if c[0] == "mkvpropedit")
    assert "track:a2" in mkv_call
    assert "language=fra" in mkv_call


def test_mp4_dispatches_to_ffmpeg_with_zero_based_audio_index(tmp_path):
    """For MP4/MOV/AVI/etc, we use ffmpeg -c copy. The -metadata:s:a:N flag
    uses 0-based indexing within audio streams."""
    f = tmp_path / "movie.mp4"
    f.write_bytes(b"\x00")
    calls = []

    def run_stub(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "ffprobe":
            return _fake_ffprobe([1, 2])
        if cmd[0] == "ffmpeg":
            # Pretend the remux succeeded — write a tmp file so shutil.move works.
            tmp_arg = cmd[-1]
            Path(tmp_arg).write_bytes(b"\x01")
            return _ok_proc()
        raise AssertionError(f"unexpected subprocess call: {cmd[0]}")

    with patch("subprocess.run", side_effect=run_stub):
        # Tag the second audio stream (absolute index 2 → audio_pos=2 → ffmpeg N=1)
        track_metadata.write_audio_language(f, 2, "ja")

    ffmpeg_call = next(c for c in calls if c[0] == "ffmpeg")
    assert "-metadata:s:a:1" in ffmpeg_call
    assert any("language=jpn" in arg for arg in ffmpeg_call)


def test_non_audio_stream_raises(tmp_path):
    """If the caller passes an absolute index that doesn't correspond to an
    audio stream (e.g. they passed a video index by mistake), we should
    refuse rather than tagging the wrong track."""
    f = tmp_path / "movie.mkv"
    f.write_bytes(b"\x00")

    def run_stub(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _fake_ffprobe([1, 2])  # only audio streams 1 and 2
        raise AssertionError(f"unexpected: {cmd[0]}")

    with patch("subprocess.run", side_effect=run_stub):
        with pytest.raises(track_metadata.MetadataWriteError, match="not an audio stream"):
            track_metadata.write_audio_language(f, 0, "fr")


def test_mkvpropedit_failure_raises(tmp_path):
    f = tmp_path / "movie.mkv"
    f.write_bytes(b"\x00")

    def run_stub(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _fake_ffprobe([1])
        if cmd[0] == "mkvpropedit":
            return _failed_proc("permission denied")
        raise AssertionError(cmd[0])

    with patch("subprocess.run", side_effect=run_stub):
        with pytest.raises(track_metadata.MetadataWriteError, match="mkvpropedit exit 1"):
            track_metadata.write_audio_language(f, 1, "fr")


def test_mkvpropedit_missing_raises_clearly(tmp_path):
    """If mkvtoolnix isn't installed we should give a clear message rather
    than letting FileNotFoundError leak."""
    f = tmp_path / "movie.mkv"
    f.write_bytes(b"\x00")

    def run_stub(cmd, **kwargs):
        if cmd[0] == "ffprobe":
            return _fake_ffprobe([1])
        if cmd[0] == "mkvpropedit":
            raise FileNotFoundError(2, "No such file", "mkvpropedit")
        raise AssertionError(cmd[0])

    with patch("subprocess.run", side_effect=run_stub):
        with pytest.raises(track_metadata.MetadataWriteError, match="mkvtoolnix-cli"):
            track_metadata.write_audio_language(f, 1, "fr")
