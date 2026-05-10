"""Tests for the STT region packer.

The packer is the hot performance lever: it concatenates multiple short
speech regions into a single 30 s decoder window so the iGPU does ~2-3×
less work on dialog-heavy films. The arithmetic has to be exact —
window-relative timestamps must demultiplex back to original-audio
seconds, and pad-zone cues must drop cleanly. This suite pins down the
edge cases.
"""
from app.pipeline.packing import (
    RegionEntry, Window, plan_packed_windows, remap_cue_to_original,
)


_SR = 16000
_CHUNK = 30 * _SR    # 480 000 samples
_PAD = 8000          # default 0.5 s


# ── Packing strategy ─────────────────────────────────────────────────────────


def test_empty_input_yields_no_windows():
    assert plan_packed_windows([], _CHUNK) == []


def test_single_short_region_makes_one_window_no_pad():
    """A solitary region produces one window with the region at offset 0
    and no pad (pad is only inserted between packed regions)."""
    region = (0, 5 * _SR)   # 5 s
    [win] = plan_packed_windows([region], _CHUNK, pad_samples=_PAD)
    assert win.audio_slices == [(0, 5 * _SR)]
    assert win.region_map == [RegionEntry(
        window_offset_samples=0,
        original_start_samples=0,
        length_samples=5 * _SR,
    )]


def test_two_short_regions_pack_into_one_window_with_pad():
    """Two 5 s regions fit comfortably in 30 s with a 0.5 s pad between.
    Second region's window offset = 5 s + pad."""
    regions = [(0, 5 * _SR), (10 * _SR, 17 * _SR)]   # 5 s + 7 s
    [win] = plan_packed_windows(regions, _CHUNK, pad_samples=_PAD)
    assert len(win.audio_slices) == 2
    # Region 0 sits at window offset 0.
    assert win.region_map[0].window_offset_samples == 0
    assert win.region_map[0].original_start_samples == 0
    # Region 1 sits at offset (region0_length + pad) = 5*SR + _PAD.
    assert win.region_map[1].window_offset_samples == 5 * _SR + _PAD
    assert win.region_map[1].original_start_samples == 10 * _SR
    assert win.region_map[1].length_samples == 7 * _SR


def test_packing_flushes_when_next_region_overflows():
    """A second region that wouldn't fit in the current window starts a new one."""
    # 25 s + 7 s + pad = 32 s > 30 s, so the second must start a new window.
    regions = [(0, 25 * _SR), (30 * _SR, 37 * _SR)]
    windows = plan_packed_windows(regions, _CHUNK, pad_samples=_PAD)
    assert len(windows) == 2
    assert windows[0].audio_slices == [(0, 25 * _SR)]
    assert windows[1].audio_slices == [(30 * _SR, 37 * _SR)]


def test_long_region_gets_standalone_window_per_chunk():
    """A region longer than chunk_samples - pad isn't packed; it's sliced
    into chunk-sized pieces, each in its own window with a trivial
    region_map (single entry at offset 0)."""
    # 75 s region — three slices: 0-30, 30-60, 60-75.
    region = (0, 75 * _SR)
    windows = plan_packed_windows([region], _CHUNK, pad_samples=_PAD)
    assert len(windows) == 3
    assert windows[0].audio_slices == [(0, 30 * _SR)]
    assert windows[1].audio_slices == [(30 * _SR, 60 * _SR)]
    assert windows[2].audio_slices == [(60 * _SR, 75 * _SR)]
    # Each window has a single-entry region_map.
    for w in windows:
        assert len(w.region_map) == 1
        assert w.region_map[0].window_offset_samples == 0


def test_long_region_flushes_open_packing_first():
    """If we have a short region open in the packer and the next region is
    long, the open packed window must be emitted BEFORE the long-region
    slices so input order is preserved."""
    regions = [
        (0, 3 * _SR),                    # short — opens a packed window
        (10 * _SR, (10 + 60) * _SR),      # long — flush + standalone slices
        (200 * _SR, 205 * _SR),           # short — opens a new packed window
    ]
    windows = plan_packed_windows(regions, _CHUNK, pad_samples=_PAD)
    # 1 packed (with the first short) + 2 long-slice + 1 packed (with the last short) = 4
    assert len(windows) == 4
    # Order check: window 0 holds region 0; windows 1+2 hold the long
    # region; window 3 holds region 2.
    assert windows[0].audio_slices == [(0, 3 * _SR)]
    assert windows[1].audio_slices == [(10 * _SR, 40 * _SR)]
    assert windows[2].audio_slices == [(40 * _SR, 70 * _SR)]
    assert windows[3].audio_slices == [(200 * _SR, 205 * _SR)]


def test_zero_length_regions_are_skipped():
    """A degenerate (start == end) region from a misbehaving VAD must
    not produce a window or contribute to packing."""
    regions = [(0, 0), (5 * _SR, 10 * _SR)]
    windows = plan_packed_windows(regions, _CHUNK)
    assert len(windows) == 1
    assert windows[0].audio_slices == [(5 * _SR, 10 * _SR)]


def test_packing_density_reduces_window_count_vs_naive():
    """Realistic dialog scenario: 10 short regions of varying length.
    Naive 1-window-per-region produces 10 windows; packed produces fewer."""
    # 10 regions of 5-8 s each, spaced 1 s apart.
    regions = []
    cur = 0
    for i in range(10):
        length = (5 + (i % 4)) * _SR   # 5..8 s
        regions.append((cur, cur + length))
        cur += length + _SR
    windows = plan_packed_windows(regions, _CHUNK)
    # 60 s of total speech + 9 pads ≈ 65 s → fits in 3 packed windows.
    # The exact count depends on packing fill behavior but it must be
    # strictly less than 10 (the naive count).
    assert len(windows) < 10
    # And every region must appear in exactly one window's region_map.
    all_origs = []
    for w in windows:
        all_origs.extend(e.original_start_samples for e in w.region_map)
    assert sorted(all_origs) == sorted(r[0] for r in regions)


# ── Timestamp remapping ──────────────────────────────────────────────────────


def test_remap_cue_inside_first_region_passes_through():
    region_map = [RegionEntry(window_offset_samples=0,
                              original_start_samples=0,
                              length_samples=5 * _SR)]
    mapped = remap_cue_to_original(0.5, 2.5, region_map, _SR)
    assert mapped == (0.5, 2.5)


def test_remap_cue_inside_second_region_applies_offset_shift():
    """The second region sits at window offset 5.5 s but corresponds to
    original-audio start 10 s. A cue at window 6.0 s → original 10.5 s."""
    region_map = [
        RegionEntry(window_offset_samples=0,
                    original_start_samples=0,
                    length_samples=5 * _SR),
        RegionEntry(window_offset_samples=5 * _SR + _PAD,
                    original_start_samples=10 * _SR,
                    length_samples=7 * _SR),
    ]
    # Cue at 6.0 s window time → lies in region 1.
    # shift = original_start (10s) - window_offset (5s + pad) → original_time
    win_start = (5 * _SR + _PAD + 1 * _SR) / _SR   # ~6.0 s
    win_end = win_start + 2.0
    mapped = remap_cue_to_original(win_start, win_end, region_map, _SR)
    assert mapped is not None
    orig_start, orig_end = mapped
    # original time = 10s (region 1 original_start) + 1s (offset into region) = 11s
    assert abs(orig_start - 11.0) < 0.001
    assert abs(orig_end - 13.0) < 0.001


def test_remap_cue_in_pad_zone_returns_none():
    """A cue whose start sample falls between two packed regions (in the
    pad zone) doesn't correspond to real audio — Whisper hallucinated.
    Caller drops it."""
    region_map = [
        RegionEntry(window_offset_samples=0,
                    original_start_samples=0,
                    length_samples=5 * _SR),
        RegionEntry(window_offset_samples=5 * _SR + _PAD,
                    original_start_samples=10 * _SR,
                    length_samples=7 * _SR),
    ]
    # 5.1 s window time → in the pad zone (5.0 to 5.5).
    mapped = remap_cue_to_original(5.1, 5.4, region_map, _SR)
    assert mapped is None


def test_remap_cue_end_bleeds_into_pad_gets_clamped():
    """When Whisper emits a cue whose end falls past its region's
    boundary (into the trailing pad), the end is clamped to the region's
    end — never assigned to the next region's timeline."""
    region_map = [
        RegionEntry(window_offset_samples=0,
                    original_start_samples=0,
                    length_samples=5 * _SR),
        RegionEntry(window_offset_samples=5 * _SR + _PAD,
                    original_start_samples=10 * _SR,
                    length_samples=7 * _SR),
    ]
    # Cue starts inside region 0 (at 4.5s), ends past region 0's end at 5.3s.
    mapped = remap_cue_to_original(4.5, 5.3, region_map, _SR)
    assert mapped is not None
    orig_start, orig_end = mapped
    # Start: 4.5s passes through (region 0 has shift 0). End: clamped to
    # region 0's end (5.0s) since it falls in the pad zone.
    assert abs(orig_start - 4.5) < 0.001
    assert abs(orig_end - 5.0) < 0.001


def test_remap_single_region_window_is_a_no_op():
    """The non-packing fallback (one region per window with window_offset=0
    and original_start=N) must round-trip cleanly: cue at window t → cue
    at original N+t."""
    region_map = [RegionEntry(window_offset_samples=0,
                              original_start_samples=600 * _SR,
                              length_samples=30 * _SR)]
    mapped = remap_cue_to_original(7.3, 12.1, region_map, _SR)
    assert mapped is not None
    orig_start, orig_end = mapped
    assert abs(orig_start - (600 + 7.3)) < 0.001
    assert abs(orig_end - (600 + 12.1)) < 0.001
