"""Silero-VAD pre-filter for the OpenVINO STT backend.

Whisper hallucinates on silent or speechless audio: fed a window of
ambience, breathing, music, or a long pause, the autoregressive decoder
still emits something — drawn from its language prior. The classic
artefacts ("Thank you.", "Thanks for watching.", "♪", repeats of a
recently-heard line) come from training data dominated by YouTube-style
transcripts.

The reference Whisper pipeline guards against this with a no-speech
threshold + log-prob threshold applied to each segment after decoding,
plus a temperature-fallback retry. We can't use any of that because we
call OVModel.generate() directly — see stt_openvino.py for why. So we
pre-filter the audio with Silero-VAD and only feed Whisper the regions
that contain speech. That:

- removes hallucinations: silent regions never reach the decoder.
- speeds the run: typical films are 30–50 % silence/music/ambience by
  audio runtime — work we now skip outright.

Silero-VAD is a tiny ONNX model that runs ~100× real-time on CPU.
Overhead on a 2 h film: under a minute. The model loads once per process
via @lru_cache.

Per-region chunking (never crossing region boundaries) keeps the
timestamp mapping trivial: every chunk has a single original-audio
offset, which we add to each cue's chunk-relative timestamp. Packing
non-contiguous regions into one chunk would force us to handle cues that
straddle the join — splitting text or re-anchoring — and that complexity
isn't worth the small compute saving.
"""
import logging
import math
from functools import lru_cache
from typing import NamedTuple


_log = logging.getLogger("subtitle_this")
_SAMPLE_RATE = 16000


class Chunk(NamedTuple):
    """One Whisper input window. start_sample/end_sample index into the
    original audio buffer; orig_offset_s is added to every cue timestamp
    the chunk produces so cue times reflect original-audio coordinates."""
    start_sample: int
    end_sample: int
    orig_offset_s: float


@lru_cache(maxsize=1)
def _silero_model():
    """Heavy import + small model load — cached so repeated jobs in the
    same worker don't re-pay it."""
    from silero_vad import load_silero_vad
    return load_silero_vad()


def detect_speech(audio, sample_rate: int) -> list[tuple[int, int]]:
    """Run Silero-VAD over a 1-D float32 mono buffer and return
    [(start_sample, end_sample), ...] of speech regions. End is exclusive.
    Empty list if Silero finds no speech (very quiet or pure-music files)."""
    if sample_rate != _SAMPLE_RATE:
        raise ValueError(f"VAD requires {_SAMPLE_RATE} Hz audio, got {sample_rate}")
    import torch
    from silero_vad import get_speech_timestamps

    model = _silero_model()
    audio_t = torch.from_numpy(audio).float()
    timestamps = get_speech_timestamps(audio_t, model, sampling_rate=sample_rate)
    return [(int(t["start"]), int(t["end"])) for t in timestamps]


def plan_chunks(
    speech_regions: list[tuple[int, int]],
    chunk_samples: int,
    sample_rate: int = _SAMPLE_RATE,
) -> list[Chunk]:
    """Walk each speech region in `chunk_samples`-sized strides, never
    crossing region boundaries. The final chunk of each region may be
    short — the caller zero-pads it to the model's expected window size.

    Pure function: no audio data, no torch — only sample-index arithmetic.
    Unit-tested without the heavy deps."""
    out: list[Chunk] = []
    for region_start, region_end in speech_regions:
        region_len = region_end - region_start
        if region_len <= 0:
            continue
        n = max(1, math.ceil(region_len / chunk_samples))
        for j in range(n):
            cs = region_start + j * chunk_samples
            ce = min(region_end, cs + chunk_samples)
            out.append(Chunk(
                start_sample=cs,
                end_sample=ce,
                orig_offset_s=cs / sample_rate,
            ))
    return out
