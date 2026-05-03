"""Write a detected language tag back into the source media's audio track.

When Whisper detects a language for an audio track that ffprobe reported as
untagged, we persist that detection back into the file's container metadata
so Emby's next probe (and any other downstream tool) sees the correct
language. Two paths by container:

- Matroska (.mkv / .mka / .webm): `mkvpropedit` edits the header in place
  without remuxing. Instant on any file size.
- Everything else (.mp4 / .m4v / .mov / .avi / …): `ffmpeg -c copy` remuxes
  into a temp sibling file and atomic-renames over the original. No re-encode
  but I/O bound (~30-90s per film on a typical disk).

ISO 639-2 (3-letter) is the canonical language tag in Matroska/MP4 metadata.
We map from Whisper's ISO 639-1 output via app.pipeline.lang.to_iso6392.

This module is best-effort by design — callers should treat MetadataWriteError
as non-fatal because the .vtt is the user-visible artifact and the tag write
is a polish on the source file.
"""
import json
import shutil
import subprocess
from pathlib import Path

from app.pipeline.lang import to_iso6392


_MATROSKA_EXTS = {".mkv", ".mka", ".webm"}


class MetadataWriteError(Exception):
    pass


def write_audio_language(
    media_path: Path, track_index: int, lang_iso6391: str
) -> None:
    """Tag the audio stream at `track_index` (the absolute ffprobe index) with
    the language `lang_iso6391` (Whisper's short code, e.g. 'fr'). Mutates the
    source file in place. Raises MetadataWriteError on any failure.
    """
    iso6392 = to_iso6392(lang_iso6391)
    if not iso6392:
        raise MetadataWriteError(
            f"no ISO 639-2 mapping for {lang_iso6391!r} — skipping tag write"
        )

    audio_pos = _audio_position(media_path, track_index)
    if audio_pos is None:
        raise MetadataWriteError(
            f"stream {track_index} in {media_path} is not an audio stream"
        )

    if media_path.suffix.lower() in _MATROSKA_EXTS:
        _write_mkv(media_path, audio_pos, iso6392)
    else:
        _write_via_ffmpeg(media_path, audio_pos, iso6392)


def _audio_position(media_path: Path, absolute_index: int) -> int | None:
    """Map an absolute stream index (the one ffprobe reports) to its 1-based
    position within the file's audio streams (the form mkvpropedit's `track:aN`
    expects, and one more than ffmpeg's 0-based `-metadata:s:a:N`).
    """
    try:
        proc = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "json",
                str(media_path),
            ],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    streams = json.loads(proc.stdout or "{}").get("streams", [])
    for i, s in enumerate(streams):
        if s.get("index") == absolute_index:
            return i + 1
    return None


def _write_mkv(media_path: Path, audio_pos: int, iso6392: str) -> None:
    """`mkvpropedit` references audio tracks by their 1-based position within
    the audio streams (track:a1, track:a2, ...). It modifies the EBML header
    in place — no remux, no temp file."""
    try:
        proc = subprocess.run(
            [
                "mkvpropedit", str(media_path),
                "--edit", f"track:a{audio_pos}",
                "--set", f"language={iso6392}",
            ],
            capture_output=True, text=True,
        )
    except FileNotFoundError as e:
        raise MetadataWriteError(
            "mkvpropedit not installed (need mkvtoolnix-cli)"
        ) from e
    if proc.returncode != 0:
        raise MetadataWriteError(
            f"mkvpropedit exit {proc.returncode}: {proc.stderr.strip()[:200]}"
        )


def _write_via_ffmpeg(media_path: Path, audio_pos: int, iso6392: str) -> None:
    """For non-Matroska containers. Remux to a sibling .babel-tmp file with
    the audio language tag set, then atomic-rename over the original. Uses
    `-c copy` so there's no re-encoding."""
    tmp = media_path.with_suffix(media_path.suffix + ".babel-tmp")
    # ffmpeg's -metadata:s:a:N uses 0-based indexing within audio streams.
    audio_idx_zero_based = audio_pos - 1
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                "-i", str(media_path),
                "-map", "0",
                "-c", "copy",
                f"-metadata:s:a:{audio_idx_zero_based}", f"language={iso6392}",
                str(tmp),
            ],
            capture_output=True, text=True,
        )
    except FileNotFoundError as e:
        raise MetadataWriteError("ffmpeg not installed") from e
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise MetadataWriteError(
            f"ffmpeg remux exit {proc.returncode}: {proc.stderr.strip()[:200]}"
        )

    try:
        # On the same filesystem, shutil.move is rename(2) — atomic.
        # Cross-filesystem fallbacks to copy+remove (still safe; original is
        # only deleted after the new file is fully in place).
        shutil.move(str(tmp), str(media_path))
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise MetadataWriteError(f"failed to replace original: {e}") from e
