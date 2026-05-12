"""On-disk cache for Whisper transcription results.

Whisper is the long pole of the pipeline — 8-80% of the progress
budget, often 30+ minutes of CPU/iGPU work for a 2 h film at
large-v3-turbo. Translation by comparison is fast (10-15 min for
NLLB-1.3B). If translation crashes (OOM, transient API error, the
container getting restarted) the user has to re-run from zero today
— a fresh half-hour of Whisper before they even get back to the
phase that failed. That's the gap this module closes.

The cache stores the `TranscriptionResult` to disk immediately after
`stt.transcribe()` returns, BEFORE the translation phase touches
anything. On retry, processor.py looks here first; a hit skips
audio extraction AND Whisper entirely and jumps straight to
translation.

Cache key dimensions — only the inputs that materially change
Whisper's output:

- **content_fingerprint** (the same one the main VTT cache uses) —
  bytes-stable across mtime bumps and path moves.
- **whisper_model** — small vs. medium vs. large-v3-turbo etc.
- **whisper_backend** — openvino and faster-whisper can produce
  slightly different cue boundaries.
- **vad_enabled** — toggles silence pre-filtering; materially
  changes the cue list (silent-region hallucinations on vs. off).
- **track_index** — which audio track was selected.

NOT in the key (deliberately):

- target_lang, provider, mode, LLM settings, scene/cinematic knobs
  — these are downstream of transcription. The whole point is to
  let those change between runs without invalidating the transcript.
- language_hint — derived deterministically from the audio + track
  metadata, both of which are captured by the fingerprint + track_index.

Storage is one JSON file per key under
``cache_dir/transcripts/{key}.json``. Atomic via tmp + os.replace.
Corrupted files are renamed to ``.corrupt`` on load rather than
crashing the pipeline.

Cleanup policy: none, for now. A 2 h film with ~1500 cues serializes
to ~200 KB. Users with disk pressure can ``rm -rf cache_dir/transcripts/``
at any time — the next run just re-transcribes.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

from app.config import settings
from app.pipeline.stt import Cue, TranscriptionResult


_log = logging.getLogger(__name__)


def _store_dir() -> Path:
    return Path(settings.cache_dir) / "transcripts"


def _key(
    content_fp: str,
    whisper_model: str,
    whisper_backend: str,
    vad_enabled: bool,
    track_index: int,
) -> str:
    """Stable composite key. Order matters only for readability — these
    fields are concatenated, not hashed, so changing order would break
    existing cache files. Don't reorder unless you bump the schema
    version prefix (and accept the one-time miss across upgrade).

    Schema versions:
    - v2 (current): cues carry source-audio-absolute timestamps.
    - v1 (pre-0.7.2): cues from multi-segment runs had timestamps stamped
      segment-relative because the region-packing remap dropped the
      additive seg_offset_seconds. Files with v1-shaped timestamps look
      structurally valid but collapse every cue into the first 600 s of
      the timeline on long media — invalidating the prefix forces a
      re-transcribe so users don't silently inherit broken caches.
    """
    return (
        f"v2"
        f"_{content_fp}"
        f"_{whisper_backend}"
        f"_{whisper_model.replace('/', '-')}"
        f"_vad{int(bool(vad_enabled))}"
        f"_t{track_index}"
    )


def lookup(
    content_fp: str,
    whisper_model: str,
    whisper_backend: str,
    vad_enabled: bool,
    track_index: int,
) -> TranscriptionResult | None:
    """Returns the cached transcription for these inputs, or None on miss
    or corrupted file. Never raises — failures are logged + the file is
    quarantined so the next run can re-transcribe cleanly."""
    path = _store_dir() / f"{_key(content_fp, whisper_model, whisper_backend, vad_enabled, track_index)}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        cues = [Cue(**c) for c in data["cues"]]
        return TranscriptionResult(
            detected_language=data["detected_language"],
            cues=cues,
        )
    except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
        _log.warning("transcript_cache: %s unreadable (%s) — quarantining", path, e)
        try:
            path.rename(path.with_suffix(".corrupt"))
        except OSError:
            pass
        return None


def store(
    content_fp: str,
    whisper_model: str,
    whisper_backend: str,
    vad_enabled: bool,
    track_index: int,
    result: TranscriptionResult,
) -> None:
    """Persist the transcription. Atomic — writes to ``.tmp`` and renames.
    Best-effort: any IO error is logged and swallowed, since persistence
    is a retry-resume optimization, not a correctness requirement."""
    if not result.cues:
        return   # don't cache empty transcriptions
    store_dir = _store_dir()
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        path = store_dir / f"{_key(content_fp, whisper_model, whisper_backend, vad_enabled, track_index)}.json"
        tmp = path.with_suffix(".tmp")
        payload = {
            "detected_language": result.detected_language,
            "cues": [asdict(c) for c in result.cues],
        }
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        _log.warning("transcript_cache: failed to save", exc_info=True)
