"""Demucs-based vocal isolation phase.

What this phase does
====================
Runs Demucs to split the source audio into stems (drums / bass / other /
vocals) and keeps only the **vocals** stem, which is then handed to
Whisper instead of the full mix. The rest of the pipeline doesn't care:
both ``audio.extract_audio`` and ``isolate_vocals`` yield the same shape
of artifact (a 16 kHz mono WAV at a ``Path``).

Why this exists
===============
Whisper-large on a full cinema mix routinely loses dialogue under score
and SFX. Silero-VAD compounds the problem by silencing whispered/quiet
lines that *do* exist but sit ≤ 12 dB above the music bed. The Inception
diagnostic run showed the climax + ending (130-145 min) had ~33-0 %
dialog coverage relative to the pro reference — almost certainly because
Hans Zimmer's score dominates the mix and the dialog is buried.

Isolating the vocals stem before STT closes most of that gap. The
silence between phrases in the isolated track is *real* silence, so VAD
becomes nearly redundant (it still runs, but rejects almost nothing).

Phase-level RAM lifecycle
=========================
Demucs htdemucs weights load to ~1-2 GB of resident PyTorch state during
the separate_audio_file() call. Holding that alongside a freshly-loaded
Whisper (~1.5 GB) + NLLB (~3 GB) blows past the typical 12 GB cgroup on
TrueNAS deployments.

The context manager below loads → runs → **explicitly releases** the
model BEFORE yielding the vocals WAV. By the time STT enters with a
``with`` block on the yielded path, Demucs occupies zero resident
memory. The vocals WAV file persists on disk through STT (where the
file is mmap'd by soundfile) and gets unlinked when the context exits.

This is the same pattern stt_openvino.release_model uses between STT
and translation — see app/pipeline/stt.py:try_malloc_trim for why
gc.collect() alone is insufficient on glibc.

Dependency
==========
Demucs is an **opt-in** dependency installed via the ``vocal-isolation``
extra (``pip install subtitle-this[vocal-isolation]``) or directly in
the Dockerfile when building the image. The import is lazy so the rest
of the app never pays for Demucs unless the feature is actually used.

Caching note
============
NOT cached on disk in this iteration — re-running the isolation costs
2-10 min of CPU per film. The transcript cache covers the common
recovery case (STT succeeded, translation failed → next run skips both
isolation and STT). The narrow window where isolation is wasted is
"isolation succeeded, STT crashed before completion" which is rare.
The transcript cache key now also includes ``vocal_isolation_enabled``
so toggling the feature ON/OFF properly invalidates.
"""
from __future__ import annotations

import gc
import logging
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from app.config import settings
from app.pipeline.stt import try_malloc_trim


_log = logging.getLogger("subtitle_this")


def _noop_progress(frac: float) -> None: ...
def _noop_cancel() -> None: ...


# Module-level state — lets release_model() find what to free without
# the caller threading a model handle through. Mirrors the per-backend
# cache pattern in stt_openvino / stt_faster_whisper.
_separator = None
_model_name_cached: str | None = None


def is_available() -> tuple[bool, str | None]:
    """Probe the import without raising. Returns ``(ok, error_message)``.
    Used by the Settings UI to render an inline warning if the user
    toggles ``vocal_isolation_enabled`` on an image that doesn't ship
    the ``demucs`` package."""
    try:
        import demucs.api  # noqa: F401
        return True, None
    except ImportError as e:
        return False, str(e)


def _load_separator(model_name: str):
    """Lazy-load and cache the Demucs separator. Reuses the cached
    instance when the model_name matches; otherwise releases the old
    one first so we never hold two model weight tensors simultaneously.

    Raises ImportError with an actionable message when the demucs
    package isn't installed on this image."""
    global _separator, _model_name_cached
    if _separator is not None and _model_name_cached == model_name:
        return _separator

    try:
        from demucs.api import Separator
    except ImportError as e:
        raise ImportError(
            "demucs is not installed in this image. Either disable "
            "vocal_isolation_enabled in Settings, or rebuild the image "
            "with the vocal-isolation extra "
            "(`pip install demucs>=4.0` in the Dockerfile)."
        ) from e

    # Drop any previously cached separator before loading the new one —
    # keeps peak memory at one model's worth.
    if _separator is not None:
        release_model()

    # device="cpu" because we don't bind Demucs to CUDA/iGPU yet. On a
    # 4-core capped TrueNAS deployment this runs ~3-8x realtime — a 2 h
    # film isolates in ~15-30 min. Acceptable as a quality-vs-time
    # trade for users who turn the feature on.
    #
    # segment=7.8 is Demucs's default chunk size in seconds. Smaller
    # = lower peak RAM (each chunk's activation tensor scales linearly)
    # but more overhead per chunk; we keep the default.
    _separator = Separator(
        model=model_name,
        device="cpu",
        progress=False,
    )
    _model_name_cached = model_name
    return _separator


def release_model() -> None:
    """Evict the cached Demucs model + run gc.collect + malloc_trim.

    Called from inside ``isolate_vocals`` AFTER separation completes
    and BEFORE the context manager yields the vocals WAV path — so
    Whisper's load doesn't pile on top of an idle Demucs.

    Safe to call when no separator is cached (no-op then). Cheap to
    call repeatedly."""
    global _separator, _model_name_cached
    _separator = None
    _model_name_cached = None
    gc.collect()
    try_malloc_trim()


def _ffmpeg_extract_for_demucs(media_path: str, track_index: int) -> Path:
    """Extract the chosen audio track as a 44.1 kHz stereo WAV that
    Demucs's pretrained models expect. Demucs *can* resample internally
    but doing it upfront via ffmpeg is faster and predictable.

    Lives under ``<cache_dir>/tmp/`` for the same reason audio.extract_audio
    does — keeps host /tmp clean and stays on the user's chosen volume."""
    tmp_dir = Path(settings.cache_dir) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".demucs_in.wav", delete=False, dir=str(tmp_dir),
    ) as tmp:
        out = Path(tmp.name)
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
            "-i", media_path,
            "-map", f"0:{track_index}",
            "-ac", "2",          # stereo (Demucs trained on stereo)
            "-ar", "44100",      # 44.1 kHz (Demucs's source rate)
            "-c:a", "pcm_s16le",
            str(out),
        ],
        check=True,
        timeout=3600,
    )
    return out


def _save_vocals_as_whisper_wav(vocals_tensor, src_sr: int, out_path: Path) -> None:
    """Mix-down stereo vocals to mono, resample to 16 kHz, write as a
    16-bit PCM WAV at the path Whisper will read. Done via torchaudio
    so we don't shell out to ffmpeg again — the tensor is already in
    memory at this point."""
    import torch
    import torchaudio
    import soundfile as sf

    # vocals_tensor shape: (channels, samples). Mono-mix by averaging.
    if vocals_tensor.dim() == 2 and vocals_tensor.shape[0] > 1:
        mono = vocals_tensor.mean(dim=0, keepdim=True)
    else:
        mono = vocals_tensor
    # Resample 44.1k → 16k using torchaudio's high-quality kaiser_window.
    if src_sr != 16000:
        resampler = torchaudio.transforms.Resample(
            orig_freq=src_sr, new_freq=16000,
            resampling_method="sinc_interp_kaiser",
        )
        mono = resampler(mono)
    # Squeeze to 1D, scale to int16 range, write with soundfile.
    samples = mono.squeeze(0).clamp(-1.0, 1.0).numpy()
    sf.write(str(out_path), samples, 16000, subtype="PCM_16")


@contextmanager
def isolate_vocals(
    media_path: str,
    track_index: int,
    *,
    model_name: str | None = None,
    progress: Callable[[float], None] = _noop_progress,
    check_cancel: Callable[[], None] = _noop_cancel,
) -> Iterator["IsolationResult"]:
    """Context manager that runs Demucs and yields a result handle whose
    ``wav_path`` points to a 16 kHz mono WAV of the vocals stem, ready
    for the STT phase to consume.

    Model lifecycle: loaded inside this block, **released before yield**.
    File lifecycle: vocals WAV created on enter, unlinked on exit.

    Raises ImportError if the demucs package isn't installed.
    Raises subprocess.CalledProcessError if ffmpeg fails to extract.
    Cancel propagation: ``check_cancel`` is called before and after the
    Demucs run; mid-run cancellation isn't supported (Demucs is one
    monolithic call). A canceled job will still finish the current
    isolation and abort at the next checkpoint."""
    model_name = model_name or settings.vocal_isolation_model
    started = time.monotonic()

    progress(0.0)
    check_cancel()
    raw_wav = _ffmpeg_extract_for_demucs(media_path, track_index)
    progress(0.1)

    # Output WAV lives in the same tmp dir so cleanup is uniform.
    tmp_dir = Path(settings.cache_dir) / "tmp"
    with tempfile.NamedTemporaryFile(
        suffix=".vocals.wav", delete=False, dir=str(tmp_dir),
    ) as tmp:
        vocals_wav = Path(tmp.name)

    audio_seconds_processed = 0.0
    try:
        check_cancel()
        sep = _load_separator(model_name)
        progress(0.2)

        # separate_audio_file returns (origin_tensor, stems_dict).
        # stems_dict keys: 'drums', 'bass', 'other', 'vocals' for htdemucs.
        # The "two-stems" variants (mdx_q, mdx_extra_q) return
        # {'vocals', 'no_vocals'} — we tolerate both by indexing by name.
        try:
            origin, stems = sep.separate_audio_file(raw_wav)
        except Exception as e:
            _log.error("Demucs separation failed: %s", e, exc_info=True)
            raise

        if "vocals" not in stems:
            raise RuntimeError(
                f"Demucs model {model_name!r} produced no 'vocals' stem "
                f"(found: {sorted(stems.keys())})"
            )
        vocals = stems["vocals"]
        progress(0.8)

        # ``origin`` is a tensor (channels, samples) at the model's SR.
        # samples/SR gives the audio length we actually processed.
        try:
            audio_seconds_processed = float(origin.shape[-1]) / float(sep.samplerate)
        except Exception:
            audio_seconds_processed = 0.0

        _save_vocals_as_whisper_wav(vocals, sep.samplerate, vocals_wav)
        progress(0.95)

        # ── Critical: release Demucs RAM BEFORE yielding ──────────────
        # Whisper / NLLB will load inside the yielded with-block. We
        # don't want them piling on top of an idle Demucs.
        release_model()
        # Free the source 44.1 kHz WAV too — STT only needs the 16 kHz
        # vocals file from here on.
        raw_wav.unlink(missing_ok=True)
        raw_wav = None  # type: ignore[assignment]

        took = time.monotonic() - started
        progress(1.0)
        yield IsolationResult(
            wav_path=vocals_wav,
            model=model_name,
            took_seconds=round(took, 2),
            audio_seconds_processed=round(audio_seconds_processed, 2),
        )
    finally:
        # Belt-and-suspenders cleanup. The release above already nulled
        # raw_wav; this handles the abort-before-release path.
        if raw_wav is not None:
            raw_wav.unlink(missing_ok=True)
        vocals_wav.unlink(missing_ok=True)
        # If we abort between _load_separator and the explicit release,
        # make sure Demucs RAM doesn't survive the context. Idempotent.
        release_model()


from dataclasses import dataclass


@dataclass
class IsolationResult:
    """Returned to the caller via the context manager. Carries both the
    artifact path (consumed by STT) and the telemetry (folded into
    PipelineMetrics for the stats page)."""
    wav_path: Path
    model: str
    took_seconds: float
    audio_seconds_processed: float
