"""STT dispatcher. Concrete backends live in sibling modules and are loaded lazily
so we never import a backend's heavy deps unless it's actually selected."""
from dataclasses import dataclass
from pathlib import Path

from app.config import settings


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


def transcribe(audio_path: Path, language_hint: str | None = None) -> TranscriptionResult:
    backend = settings.whisper_backend.lower()
    if backend == "openvino":
        from app.pipeline.stt_openvino import transcribe as run
    elif backend == "cpu":
        from app.pipeline.stt_faster_whisper import transcribe as run
    else:
        raise ValueError(
            f"Unknown BABEL_WHISPER_BACKEND={settings.whisper_backend!r} (expected 'cpu' or 'openvino')"
        )
    return run(audio_path, language_hint=language_hint)
