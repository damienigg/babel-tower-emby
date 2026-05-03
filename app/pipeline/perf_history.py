"""Per-model real-time-factor (RTF) history for the openvino transcribe
heartbeat.

The progress bar's heartbeat needs an estimate of how long the transcribe
will take. Baked-in defaults are necessarily wrong for everyone — a beefy
N305 iGPU is faster than a passive N100, and Whisper's own internal speed
varies with audio characteristics. Rather than hard-coding one number per
model size, we measure the actual RTF on each successful run and persist
the last 10 samples per model. The median is robust to one weird-fast
cached-IR run or one weird-slow cold-cache run.

This makes the bar self-correcting: the FIRST transcribe of a given model
uses the baked-in table. The SECOND uses last run's actual measurement.
By the third-fifth run the estimate is calibrated to the user's hardware.

State lives at /cache/rtf-history.json. Wiped on container restart only if
/cache itself is wiped. Format:

    {"small": [0.072, 0.069, 0.075], "large-v3-turbo": [0.118, 0.121]}

Each entry is the last N RTF samples for that whisper_model. RTF =
elapsed_seconds / audio_duration_seconds; lower is faster.
"""
import json
import statistics
from pathlib import Path

from app.config import settings


_HISTORY_FILENAME = "rtf-history.json"
_MAX_SAMPLES = 10


def _path() -> Path:
    return settings.cache_dir / _HISTORY_FILENAME


def _load() -> dict[str, list[float]]:
    p = _path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    # Best-effort coercion. A corrupt entry is dropped silently rather than
    # throwing — the heartbeat will just fall back to the baked-in default
    # for that model on this run.
    out: dict[str, list[float]] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not isinstance(v, list):
            continue
        try:
            out[k] = [float(x) for x in v if isinstance(x, (int, float))]
        except (TypeError, ValueError):
            continue
    return out


def _save(history: dict[str, list[float]]) -> None:
    try:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _path().write_text(json.dumps(history))
    except OSError:
        # Cache disk full or read-only — heartbeat continues working with
        # baked-in defaults, just no learning across runs.
        pass


def estimated_rtf(model: str, *, default: float) -> float:
    """Median of recent RTF samples for `model`, or `default` if we have
    none. Returns a strictly-positive float."""
    samples = _load().get(model, [])
    if not samples:
        return default
    return max(0.001, statistics.median(samples))


def record_rtf(model: str, rtf: float) -> None:
    """Append a fresh measurement for `model`, keeping the most recent
    _MAX_SAMPLES entries. Discards garbage values defensively (an audio
    file with rounding-zero duration would give NaN/inf)."""
    if not (0.0 < rtf < 100.0):
        return
    history = _load()
    samples = history.get(model, [])
    samples.append(float(rtf))
    history[model] = samples[-_MAX_SAMPLES:]
    _save(history)
