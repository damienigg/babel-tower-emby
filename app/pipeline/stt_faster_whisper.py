from functools import lru_cache
from pathlib import Path

from faster_whisper import WhisperModel

from app.config import settings
from app.pipeline.stt import Cue, TranscriptionResult


@lru_cache(maxsize=2)
def _model(name: str, device: str, compute_type: str) -> WhisperModel:
    """Cache keyed by config so settings changes (UI or env) reload the model.
    maxsize=2 keeps one fallback warm when the user toggles between two models."""
    return WhisperModel(name, device=device, compute_type=compute_type)


def transcribe(audio_path: Path, language_hint: str | None = None) -> TranscriptionResult:
    model = _model(settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)
    segments, info = model.transcribe(
        str(audio_path),
        language=language_hint,
        vad_filter=True,
        beam_size=5,
    )
    cues: list[Cue] = []
    for i, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        cues.append(Cue(id=i, start=float(seg.start), end=float(seg.end), text=text))
    return TranscriptionResult(detected_language=info.language, cues=cues)
