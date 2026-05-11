"""STT dispatcher. Concrete backends live in sibling modules and are loaded lazily
so we never import a backend's heavy deps unless it's actually selected."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Cue:
    id: int
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    detected_language: str
    cues: list[Cue]


def _noop_progress(frac: float) -> None: ...
def _noop_cancel() -> None: ...


def transcribe(
    audio_path: Path,
    language_hint: str | None = None,
    *,
    progress: Callable[[float], None] = _noop_progress,
    check_cancel: Callable[[], None] = _noop_cancel,
) -> TranscriptionResult:
    """`progress` reports fractional completion in [0,1] within transcription
    (the outer pipeline maps it onto its own 0-100 budget). `check_cancel`
    raises JobCanceled if the user has clicked cancel — backends call it
    between segments / chunks so cancel takes effect within seconds, not
    minutes."""
    from app.config import settings
    backend = settings.whisper_backend.lower()
    if backend == "openvino":
        from app.pipeline.stt_openvino import transcribe as run
    elif backend == "cpu":
        from app.pipeline.stt_faster_whisper import transcribe as run
    else:
        raise ValueError(
            f"Unknown BABEL_WHISPER_BACKEND={settings.whisper_backend!r} (expected 'cpu' or 'openvino')"
        )
    return run(audio_path, language_hint=language_hint, progress=progress, check_cancel=check_cancel)


def release() -> None:
    """Evict the active backend's cached Whisper model. Dispatcher mirror of
    transcribe().

    Called by processor.py between the STT and translation phases so the
    ~1-1.5 GB Whisper weights don't sit resident while NLLB / vision-LLM
    state loads — the two together exceed the default 6 GB cgroup limit on
    typical NAS deployments and trigger a silent kernel OOM-kill at the
    80% mark of the pipeline. Reloading on the next job costs ~10-30s,
    which is dwarfed by the transcription cost itself.

    Safe to call when no model is cached — cache_clear() is a no-op then.
    """
    from app.config import settings
    backend = settings.whisper_backend.lower()
    if backend == "openvino":
        from app.pipeline.stt_openvino import release_model
    elif backend == "cpu":
        from app.pipeline.stt_faster_whisper import release_model
    else:
        return
    release_model()
