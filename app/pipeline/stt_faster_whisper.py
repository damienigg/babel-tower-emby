from functools import lru_cache
from pathlib import Path
from typing import Callable

from faster_whisper import WhisperModel

from app.config import settings
from app.pipeline.stt import Cue, TranscriptionResult


@lru_cache(maxsize=1)
def _model(name: str, device: str, compute_type: str) -> WhisperModel:
    """Cache keyed by config so settings changes (UI or env) reload the model.
    maxsize=1 — toggling whisper_model in the UI evicts the previous one
    rather than keeping both resident. Whisper-large weights are ~3 GB;
    holding a spare doubles RAM for no real workflow benefit."""
    return WhisperModel(name, device=device, compute_type=compute_type)


def release_model() -> None:
    """Evict the cached CPU Whisper model. Mirror of stt_openvino.release_model
    — called between transcribe and translate so the local NLLB / vision-LLM
    state can load without piling on top of an idle Whisper still resident.
    try_malloc_trim() returns the freed glibc arenas to the kernel; see
    its docstring in stt.py for why gc.collect() alone isn't enough."""
    import gc
    from app.pipeline.stt import try_malloc_trim
    _model.cache_clear()
    gc.collect()
    try_malloc_trim()


def _noop_progress(frac: float) -> None: ...
def _noop_cancel() -> None: ...


def transcribe(
    audio_path: Path,
    language_hint: str | None = None,
    *,
    progress: Callable[[float], None] = _noop_progress,
    check_cancel: Callable[[], None] = _noop_cancel,
) -> TranscriptionResult:
    model = _model(settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language=language_hint,
        vad_filter=True,
        beam_size=5,
        # ``condition_on_previous_text=False`` is the recommended
        # setting for long-form transcription per the Whisper paper
        # (Section 4.5) and faster-whisper's own README: with it
        # enabled (the library default), the model conditions each
        # 30 s window on the previous window's TEXT — which causes
        # cascading hallucinations after a silent gap (Whisper
        # generates "Thank you. Thanks for watching." repeatedly,
        # then conditions the next window on that, and the loop
        # continues for minutes). On dialog-heavy films with score-
        # bedded scenes, this is THE main source of nonsense cues.
        # Disabling it costs a bit of cross-window context but
        # eliminates the cascading-hallucination class entirely.
        condition_on_previous_text=False,
        # Filter out segments with very low average log-probability —
        # those are the model's "I'm not sure but here's a guess"
        # outputs, which on silence become exactly the signature
        # YouTube-style hallucinations we want to drop. -1.0 is the
        # OpenAI Whisper default; faster-whisper exposes it.
        log_prob_threshold=-1.0,
        # And drop segments where the no-speech probability is high —
        # Whisper's own gate against transcribing silence as if it
        # were speech. 0.6 is the OpenAI default.
        no_speech_threshold=0.6,
    )
    # info.duration is the audio length in seconds (post-VAD when applicable).
    # Each yielded segment has .end (audio timestamp), so segment.end /
    # duration is a fair fractional progress estimate.
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    cues: list[Cue] = []
    for i, seg in enumerate(segments):
        check_cancel()
        text = seg.text.strip()
        if text:
            cues.append(Cue(id=i, start=float(seg.start), end=float(seg.end), text=text))
        if duration > 0:
            progress(float(seg.end) / duration)
    progress(1.0)
    return TranscriptionResult(detected_language=info.language, cues=cues)
