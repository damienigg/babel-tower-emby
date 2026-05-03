"""Tests for the VAD chunk planner used by the OpenVINO STT backend.

`plan_chunks` is the pure-Python bookkeeping layer that turns Silero's
list of speech regions into Whisper input windows. The actual VAD inference
needs torch + silero-vad and runs at job time; here we exercise only the
sample-arithmetic so the openvino-flavor heavy deps aren't required.
"""
from app.pipeline.vad import Chunk, plan_chunks


_SR = 16000
_CHUNK_SAMPLES = 30 * _SR  # 480000 samples = 30s


def test_plan_chunks_empty_input_yields_nothing():
    assert plan_chunks([], _CHUNK_SAMPLES, _SR) == []


def test_plan_chunks_short_region_yields_one_chunk():
    """A 5s speech region produces a single chunk anchored at 0s."""
    region = (0, 5 * _SR)
    chunks = plan_chunks([region], _CHUNK_SAMPLES, _SR)
    assert chunks == [Chunk(start_sample=0, end_sample=5 * _SR, orig_offset_s=0.0)]


def test_plan_chunks_long_region_strides_in_30s_windows():
    """A 70s region needs 3 chunks: [0,30), [30,60), [60,70)."""
    region = (0, 70 * _SR)
    chunks = plan_chunks([region], _CHUNK_SAMPLES, _SR)
    assert chunks == [
        Chunk(start_sample=0,             end_sample=30 * _SR, orig_offset_s=0.0),
        Chunk(start_sample=30 * _SR,      end_sample=60 * _SR, orig_offset_s=30.0),
        Chunk(start_sample=60 * _SR,      end_sample=70 * _SR, orig_offset_s=60.0),
    ]


def test_plan_chunks_offset_propagates_for_non_zero_region_start():
    """Region starting at 100s produces a chunk with orig_offset_s=100.0
    so cue timestamps land on the original audio timeline, not the
    region-relative one."""
    region = (100 * _SR, 105 * _SR)
    chunks = plan_chunks([region], _CHUNK_SAMPLES, _SR)
    assert chunks == [Chunk(start_sample=100 * _SR, end_sample=105 * _SR, orig_offset_s=100.0)]


def test_plan_chunks_never_crosses_region_boundaries():
    """Two short regions far apart yield two chunks, not one packed chunk —
    crossing boundaries would produce cues stretched through silence."""
    regions = [(0, 5 * _SR), (200 * _SR, 210 * _SR)]
    chunks = plan_chunks(regions, _CHUNK_SAMPLES, _SR)
    assert chunks == [
        Chunk(start_sample=0,         end_sample=5 * _SR,   orig_offset_s=0.0),
        Chunk(start_sample=200 * _SR, end_sample=210 * _SR, orig_offset_s=200.0),
    ]


def test_plan_chunks_skips_zero_length_regions():
    """A degenerate (start == end) region from a misbehaving VAD must not
    produce a phantom chunk."""
    regions = [(0, 0), (10 * _SR, 12 * _SR)]
    chunks = plan_chunks(regions, _CHUNK_SAMPLES, _SR)
    assert chunks == [Chunk(start_sample=10 * _SR, end_sample=12 * _SR, orig_offset_s=10.0)]


def test_plan_chunks_long_region_with_offset_combines_correctly():
    """Region [50s, 145s] = 95s long → 4 chunks at offsets 50, 80, 110, 140."""
    region = (50 * _SR, 145 * _SR)
    chunks = plan_chunks([region], _CHUNK_SAMPLES, _SR)
    offsets = [c.orig_offset_s for c in chunks]
    assert offsets == [50.0, 80.0, 110.0, 140.0]
    # Last chunk is short (5s of real audio), not padded — the caller pads
    # to chunk_samples before feeding Whisper.
    assert chunks[-1].end_sample - chunks[-1].start_sample == 5 * _SR
