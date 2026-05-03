"""Language detection pre-pass for the OpenVINO Whisper backend.

The OpenVINO HF pipeline doesn't surface Whisper's auto-detected language
(see app/pipeline/stt_openvino.py). When the source audio track is untagged,
that bug propagates downstream: NLLB and DeepL get told the wrong source
language and produce garbage. This module fixes that by running a quick
language-detection pass with `faster-whisper` (tiny model, ~75 MB on disk)
on the first ~30 seconds of the extracted WAV.

Cheap (2-3s on CPU after the model is warmed) and only triggered when the
ffprobe track tag is missing AND the configured Whisper backend is openvino.
The CPU backend (`faster-whisper`) does its own detection during the main
transcribe call, so we don't run this pre-pass there.
"""
from functools import lru_cache
from pathlib import Path


_DETECTOR_MODEL = "tiny"
_DETECTION_SECONDS = 30


@lru_cache(maxsize=1)
def _detector():
    """One process-wide tiny Whisper instance for language detection."""
    from faster_whisper import WhisperModel
    return WhisperModel(_DETECTOR_MODEL, device="cpu", compute_type="int8")


def detect(wav_path: Path) -> str | None:
    """Run Whisper's language detection on the first ~30s of audio. Returns
    the ISO 639-1 code (e.g. 'fr', 'ja') or None if detection failed.
    """
    try:
        import soundfile as sf
    except ImportError:
        return None

    try:
        audio, sr = sf.read(str(wav_path))
    except Exception:
        return None
    if sr != 16000:
        return None

    sample = audio[: _DETECTION_SECONDS * sr]
    if len(sample) == 0:
        return None

    try:
        model = _detector()
        # beam_size=1 + condition_on_previous_text=False = the cheapest pass.
        # We don't care about the transcribed text, only info.language.
        segments, info = model.transcribe(
            sample,
            language=None,
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=True,
        )
        # Force the generator far enough that faster-whisper has finalized
        # info.language regardless of internal lazy evaluation.
        next(iter(segments), None)
    except Exception:
        return None

    return info.language or None
