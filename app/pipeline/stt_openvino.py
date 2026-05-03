"""OpenVINO STT backend via optimum-intel.

Uses Hugging Face Whisper exported to OpenVINO IR. The first call for a given
model triggers download + IR conversion (slow, 5-30 min depending on size);
subsequent calls hit the cached IR and run on the configured device.
"""
import logging
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import soundfile as sf

from app.config import settings
from app.pipeline.openvino_introspect import log_selected_device
from app.pipeline.stt import Cue, TranscriptionResult


def _noop_progress(frac: float) -> None: ...
def _noop_cancel() -> None: ...


# Empirical real-time factors for OpenVINO Whisper on Intel iGPU (N100/N305
# class hardware), per model. Used only to drive the cosmetic heartbeat —
# the bar's slope is purely visual feedback, the actual finish time is
# reported when the blocking pipe() call returns. If the estimate is too
# fast the bar plateaus at HEARTBEAT_CAP; if too slow it never quite reaches
# the cap and jumps from wherever to 1.0 at the end. Model-keyed so the
# slope roughly matches reality for each whisper size — without this,
# small/base would visibly plateau too early and large would crawl.
_OPENVINO_RTF_BY_MODEL: dict[str, float] = {
    "tiny": 0.04,
    "base": 0.05,
    "small": 0.07,
    "medium": 0.12,
    "large-v3-turbo": 0.12,
    "large-v3": 0.20,
}
_DEFAULT_RTF = 0.10
_HEARTBEAT_CAP = 0.95
_HEARTBEAT_INITIAL = 0.02
_HEARTBEAT_TICK_SECONDS = 4.0


_log = logging.getLogger("subtitle_this")
_HF_PREFIX = "openai/whisper-"


@lru_cache(maxsize=2)
def _pipeline(model_name: str, device: str, cache_root: str):
    """Cache keyed by config so settings changes reload the pipeline.
    Heavy imports stay inside so the CPU backend doesn't pay them at import time."""
    from optimum.intel import OVModelForSpeechSeq2Seq
    from transformers import AutoProcessor, pipeline as hf_pipeline

    model_id = _HF_PREFIX + model_name
    cache_dir = Path(cache_root) / "openvino-models"
    cache_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
    model = OVModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        export=True,
        device=device,
        cache_dir=str(cache_dir),
    )
    log_selected_device("whisper:" + model_name, requested=device, model=model)

    return hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30,
        return_timestamps=True,
    )


def transcribe(
    audio_path: Path,
    language_hint: str | None = None,
    *,
    progress: Callable[[float], None] = _noop_progress,
    check_cancel: Callable[[], None] = _noop_cancel,
) -> TranscriptionResult:
    audio, sr = sf.read(str(audio_path))
    if sr != 16000:
        raise RuntimeError(f"expected 16 kHz audio, got {sr} Hz")

    check_cancel()
    pipe = _pipeline(settings.whisper_model, settings.openvino_device, str(settings.cache_dir))
    check_cancel()
    generate_kwargs: dict = {"task": "transcribe"}
    if language_hint:
        generate_kwargs["language"] = language_hint

    # The HF pipeline takes the entire audio array and chunks internally —
    # no per-chunk callback is exposed. To avoid a UI bar that sits frozen
    # at 12% for 10+ minutes, run a daemon "heartbeat" thread alongside the
    # blocking pipe() call: it bumps progress every few seconds based on
    # elapsed time vs. an estimated transcribe duration (audio_duration *
    # RTF). Caps at HEARTBEAT_CAP so we never overshoot the real "done"
    # signal. Pure cosmetic — the actual transcribe is unaffected.
    audio_duration_s = len(audio) / sr
    rtf = _OPENVINO_RTF_BY_MODEL.get(settings.whisper_model, _DEFAULT_RTF)
    estimated_total_s = max(30.0, audio_duration_s * rtf)
    started = time.monotonic()

    progress(_HEARTBEAT_INITIAL)
    done = threading.Event()
    def _heartbeat():
        while not done.wait(timeout=_HEARTBEAT_TICK_SECONDS):
            elapsed = time.monotonic() - started
            frac = _HEARTBEAT_INITIAL + (elapsed / estimated_total_s) * (_HEARTBEAT_CAP - _HEARTBEAT_INITIAL)
            progress(min(_HEARTBEAT_CAP, frac))
    heartbeat = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat.start()
    try:
        result = pipe(audio, return_timestamps=True, generate_kwargs=generate_kwargs)
    finally:
        done.set()
        heartbeat.join(timeout=2.0)
    check_cancel()
    progress(1.0)

    cues: list[Cue] = []
    for i, chunk in enumerate(result.get("chunks", [])):
        ts = chunk.get("timestamp") or (None, None)
        if ts[0] is None or ts[1] is None:
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        cues.append(Cue(id=i, start=float(ts[0]), end=float(ts[1]), text=text))

    # The HF pipeline doesn't surface the language detected by Whisper. Two
    # sources of truth for `language_hint` upstream:
    # 1. ffprobe track tag (when the file is properly tagged)
    # 2. faster-whisper-tiny language-detection pre-pass run by processor.py
    #    when the track has no tag (see app/pipeline/lang_detect.py)
    # The "en" fallback only triggers if BOTH the file is untagged AND the
    # pre-pass returned nothing (e.g. silent or extremely noisy first 30s).
    detected = language_hint or "en"
    return TranscriptionResult(detected_language=detected, cues=cues)
