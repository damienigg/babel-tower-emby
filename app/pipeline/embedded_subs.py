"""Embedded-subtitle short-circuit.

Many media files already carry subtitle tracks — sometimes professionally
authored, often in the user's target language already. Running Whisper STT
+ NLLB on a track when a pro-authored target-lang sub already sits in the
container is wasted compute AND wasted accuracy: STT introduces timing
drift and translation introduces lexical drift, neither of which can
match human-authored output.

This module gives the processor the option to:

1. Detect text-based subtitle tracks via ffprobe.
2. Pick the best candidate by language priority
   (target_lang > source_lang > en > first text track).
3. Extract the chosen track to a tmp .vtt via ffmpeg.

The processor then either copies the result as-is (when chosen.lang ==
target_lang) or feeds the parsed cues straight into the translation
phase (skipping STT entirely).

Image-based subtitle codecs (PGS / DVD bitmap) are detected and reported
but never extracted — without OCR they can't be turned into text. The
processor falls back to STT for those. OCR support is a Phase 2 concern.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.pipeline.lang import normalize

_log = logging.getLogger("subtitle_this")


# Codec families ffmpeg can directly convert to WebVTT without OCR.
# subrip = .srt, ass/ssa = SubStation Alpha, mov_text = MP4 text track,
# webvtt = pass-through. text = generic plain text (rare).
_TEXT_CODECS = frozenset({
    "subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text", "tx3g",
})

# Image-based subtitle codecs we can identify but not extract as text
# without an OCR pass. Reported so the UI can explain why a Bluray
# rip with only PGS tracks still falls back to STT.
_BITMAP_CODECS = frozenset({
    "hdmv_pgs_subtitle", "pgs", "dvd_subtitle", "dvdsub", "xsub", "vobsub",
})


@dataclass
class SubtitleTrack:
    """One subtitle stream as reported by ffprobe.

    `index` is the absolute ffprobe stream index (the one you pass to
    `ffmpeg -map 0:N`). `track_index_within_subs` is the 0-based position
    within the subtitle-only stream list (the one ffmpeg's `-map 0:s:N`
    syntax uses); the extractor uses the latter because it's stable
    across audio/video stream reordering.
    """
    index: int
    track_index_within_subs: int
    codec: str
    language: str | None
    title: str | None
    is_default: bool
    is_forced: bool
    is_hearing_impaired: bool

    @property
    def is_text(self) -> bool:
        return self.codec in _TEXT_CODECS

    @property
    def is_bitmap(self) -> bool:
        return self.codec in _BITMAP_CODECS


def list_subtitle_tracks(media_path: str) -> list[SubtitleTrack]:
    """Probe `media_path` for subtitle streams via ffprobe.

    Returns a list of SubtitleTrack — empty if the file has no subtitle
    streams. Raises subprocess.SubprocessError / TimeoutExpired only on
    ffprobe failures the caller hasn't already accepted; the caller is
    expected to fall back to the STT path on any error.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "s",
            "-show_entries",
            "stream=index,codec_name:"
            "stream_tags=language,title:"
            "stream_disposition=default,forced,hearing_impaired",
            "-of", "json",
            media_path,
        ],
        capture_output=True, text=True, check=True,
        timeout=30,
    )
    data = json.loads(result.stdout or "{}")
    tracks: list[SubtitleTrack] = []
    for sub_pos, s in enumerate(data.get("streams", [])):
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        tracks.append(SubtitleTrack(
            index=s["index"],
            track_index_within_subs=sub_pos,
            codec=(s.get("codec_name") or "").lower(),
            language=normalize(tags.get("language")),
            title=tags.get("title"),
            is_default=bool(disp.get("default")),
            is_forced=bool(disp.get("forced")),
            is_hearing_impaired=bool(disp.get("hearing_impaired")),
        ))
    return tracks


def pick_best_track(
    tracks: list[SubtitleTrack],
    target_lang: str,
    source_lang: str | None,
) -> SubtitleTrack | None:
    """Choose the best text subtitle track for the job.

    Priority:
    1. target_lang text track (lets caller copy-as-is, zero translation)
    2. source_lang text track (if known)
    3. English text track (most common pivot lang for translation)
    4. First text track of any language

    Forced-only tracks are skipped — they cover hardcoded foreign
    dialog snippets (~5 % of a film) not the full subtitle set. SDH
    tracks (hearing_impaired) are kept; they're full transcripts plus
    sound-effect annotations, which is strictly more content.

    Returns None when no text track is available. The caller falls
    back to STT in that case.
    """
    text_tracks = [t for t in tracks if t.is_text and not t.is_forced]
    if not text_tracks:
        return None

    def rank(t: SubtitleTrack) -> tuple:
        # Lower is better — sort ascending.
        if t.language == target_lang:
            lang_rank = 0
        elif source_lang and t.language == source_lang:
            lang_rank = 1
        elif t.language == "en":
            lang_rank = 2
        else:
            lang_rank = 3
        # Within a language tier, prefer non-SDH (cleaner text, fewer
        # bracketed sound effects). Then default-disposition. Then
        # stream index for stable ordering.
        sdh_penalty = 1 if t.is_hearing_impaired else 0
        default_bonus = 0 if t.is_default else 1
        return (lang_rank, sdh_penalty, default_bonus, t.index)

    return min(text_tracks, key=rank)


def extract_track(
    media_path: str,
    track: SubtitleTrack,
    output_dir: Path | None = None,
) -> Path:
    """Extract `track` to a WebVTT file via ffmpeg.

    Returns the path to the extracted .vtt. Raises subprocess errors on
    ffmpeg failures — the caller catches and falls back to STT.

    `output_dir` defaults to a fresh tempdir; pass a specific dir to
    keep the artefact around for debugging.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="embedded_subs_"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"track_{track.index}.vtt"

    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", media_path,
        # -map 0:s:N selects the Nth subtitle stream (0-based within
        # the subtitle stream group). Using this rather than the
        # absolute stream index makes the command resilient to files
        # where the subtitle streams aren't contiguous.
        "-map", f"0:s:{track.track_index_within_subs}",
        # -c:s webvtt converts srt/ass/mov_text/etc. → WebVTT
        # natively. ffmpeg's subtitle decoders strip ASS styling
        # (\an8 positions, color tags) on conversion, which is
        # exactly what we want — our translation pipeline operates
        # on plain text.
        "-c:s", "webvtt",
        str(out_path),
    ]
    subprocess.run(
        cmd,
        capture_output=True, text=True, check=True,
        # 60 s is generous for subtitle extraction — even a 3 h film
        # has <2 MB of subtitle text and the conversion is near
        # instantaneous. A timeout that bites means something is
        # very wrong (file corruption, ffmpeg deadlock); failing
        # fast lets the caller drop back to STT.
        timeout=60,
    )
    return out_path


def cleanup_extract_dir(path: Path) -> None:
    """Best-effort cleanup of the tempdir created by extract_track.

    Used by the processor in a try/finally so a crashed pipeline
    doesn't leak GBs of tmpdirs. Safe to call on a path that has
    already been removed.
    """
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        _log.warning("embedded_subs: cleanup of %s failed", path, exc_info=True)
