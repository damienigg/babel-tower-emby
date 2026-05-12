"""Speech-region packing for the OpenVINO Whisper STT pipeline.

The naive chunking strategy (one 30 s window per speech region, zero-pad
the tail) wastes a lot of decoder compute on dialog-heavy films: a 7 s
utterance becomes a 30 s decoder window that's 77% zero-padding. Whisper
still processes the full 30 s mel + attention either way.

This module implements REGION PACKING — concatenating multiple short
speech regions into a single 30 s decoder window, separated by a brief
silence pad so Whisper treats them as distinct segments. After decode,
the emitted cue timestamps (which are window-relative) are demultiplexed
back to original-audio timestamps via each window's `region_map`.

Performance impact on a 2 h dialog-heavy film:
- Without packing: ~250 chunks, each ~30 s decoded.
- With packing:   ~80-120 windows, each ~30 s decoded.
- Speedup:        ~2-3× on iGPU inference time.

Trade-offs:
- Whisper occasionally emits a cue whose timestamp falls inside a pad
  zone (it can "smear" between two packed regions). Those cues are
  dropped — they correspond to silence anyway, and missing one cue is
  preferable to misattributing it to the wrong original timestamp.
- Long regions (longer than the window) still get sliced the original
  way; they don't pack with neighbors. This handles the
  long-monologue case correctly even with packing enabled.

The packer is a PURE FUNCTION over sample indices and durations — no
audio data, no torch. The unit test suite exercises it without any heavy
deps.
"""
from dataclasses import dataclass, field


@dataclass
class RegionEntry:
    """One speech region's slot inside a packed window.

    `window_offset_samples` is where this region's audio starts within
    the assembled 30 s window. `original_start_samples` is where it
    started in the original audio buffer. `length_samples` is its
    duration. The translator demultiplexes cue timestamps via:

        original_time = window_relative_time
                      - (window_offset_samples / sample_rate)
                      + (original_start_samples / sample_rate)
    """
    window_offset_samples: int
    original_start_samples: int
    length_samples: int


@dataclass
class Window:
    """One 30 s decoder input window. Either a single long-region slice
    (when a speech region exceeds the window length and gets split) or
    a packed bundle of short regions separated by silence pads.

    `audio_slices` describes which slices of the source audio buffer to
    concatenate (with `pad_samples_between` zeros inserted between
    consecutive slices). `region_map` is the per-region offset table used
    to demultiplex cue timestamps after decode.

    Invariants:
    - sum(length for each slice) + (n-1) * pad_samples_between <= chunk_samples
    - region_map has the same length as audio_slices (1:1 mapping)
    """
    audio_slices: list[tuple[int, int]]   # (start_sample, end_sample) into source audio
    pad_samples_between: int               # silence pad inserted between slices
    region_map: list[RegionEntry] = field(default_factory=list)


def plan_packed_windows(
    speech_regions: list[tuple[int, int]],
    chunk_samples: int,
    pad_samples: int = 8000,    # 0.5 s at 16 kHz — gives Whisper a clear segment boundary
    max_regions_per_window: int = 0,   # 0 = unlimited (pre-0.7.11 behavior)
) -> list[Window]:
    """Walk speech_regions in order and bin them into 30 s windows.

    Strategy:
    - A region SHORTER than `chunk_samples - pad_samples` is a candidate
      for packing. We try to fit it into the current open window
      (preceded by a pad if the window isn't empty). If it doesn't fit,
      flush the current window and start a new one with this region.
    - A region AT-OR-LONGER than `chunk_samples - pad_samples` is treated
      as a "long region": it gets one or more standalone windows (sliced
      into chunk_samples slices the original way), and the current
      packing accumulator is flushed first so its packed window doesn't
      precede the long-region slices in input order.

    `max_regions_per_window`: hard cap on packing density. Set to 0
    (the historical default) for no cap. With a film-mix audio source,
    Whisper-turbo's timestamp accuracy degrades as the number of
    discontinuities in the window grows — at the Inception measurement
    (12.4 regions/window average) Whisper drifted enough that 44 % of
    its cues landed in pad zones. A cap of 4 keeps pad overhead at
    ~5 % of the window's audio time instead of 20 %, while still
    cutting iGPU compute substantially vs no-packing-at-all.

    Pure function — no audio data or torch. Unit-testable without native deps.
    """
    out: list[Window] = []
    cur_slices: list[tuple[int, int]] = []
    cur_map: list[RegionEntry] = []
    cur_used = 0   # samples consumed in the open packed window, including pads

    def _flush() -> None:
        nonlocal cur_slices, cur_map, cur_used
        if cur_slices:
            out.append(Window(
                audio_slices=cur_slices,
                pad_samples_between=pad_samples,
                region_map=cur_map,
            ))
            cur_slices = []
            cur_map = []
            cur_used = 0

    pack_capacity = chunk_samples   # absolute upper bound on used samples
    for r_start, r_end in speech_regions:
        r_length = r_end - r_start
        if r_length <= 0:
            continue

        # Long-region branch: doesn't pack. Flush whatever's pending,
        # then emit one window per chunk_samples slice.
        if r_length >= chunk_samples - pad_samples:
            _flush()
            j = 0
            while j * chunk_samples < r_length:
                slice_start = r_start + j * chunk_samples
                slice_end = min(r_end, slice_start + chunk_samples)
                out.append(Window(
                    audio_slices=[(slice_start, slice_end)],
                    pad_samples_between=pad_samples,
                    region_map=[RegionEntry(
                        window_offset_samples=0,
                        original_start_samples=slice_start,
                        length_samples=slice_end - slice_start,
                    )],
                ))
                j += 1
            continue

        # Short-region branch: try to pack into the current window.
        # Account for the pad we'd insert before this region (skipped
        # when the window is empty).
        needed = r_length + (pad_samples if cur_slices else 0)
        # Two flush conditions: (a) the region doesn't fit in remaining
        # window capacity, (b) the window has already reached the
        # per-window cap (when set). Either way: close the window and
        # start a fresh one with no leading pad.
        cap_reached = (
            max_regions_per_window > 0
            and len(cur_slices) >= max_regions_per_window
        )
        if cur_used + needed > pack_capacity or cap_reached:
            _flush()
            needed = r_length    # no leading pad in a fresh window

        window_offset = cur_used + (pad_samples if cur_slices else 0)
        cur_slices.append((r_start, r_end))
        cur_map.append(RegionEntry(
            window_offset_samples=window_offset,
            original_start_samples=r_start,
            length_samples=r_length,
        ))
        cur_used += needed

    _flush()
    return out


def remap_cue_to_original(
    cue_start_window_s: float,
    cue_end_window_s: float,
    region_map: list[RegionEntry],
    sample_rate: int,
) -> tuple[float, float, bool] | None:
    """Given a cue's window-relative (start, end) seconds, find which
    packed region it belongs to and return its original-audio
    (start, end, was_snapped).

    The third element distinguishes two cases the caller wants to count
    separately:

    - ``was_snapped=False``: the cue's start fell cleanly inside a real
      region — no positional adjustment needed.
    - ``was_snapped=True``: the cue's start fell in a silence pad zone
      between packed regions. Pre-0.7.11 we returned None and the
      caller silently dropped the cue, which is how Inception's 733
      legitimate cues vanished (Whisper's autoregressive timestamp
      prediction drifts 100-300 ms on densely-packed windows; cues
      with valid TEXT content but slightly off TIMESTAMPS got nuked).
      We now snap the cue's start to the closest region's start
      boundary, preserving the content with at most ~0.5 s of time
      shift (the pad width), which is well below the perceptual
      audio-subtitle sync threshold (~1 s).

    Returns None only when ``region_map`` is empty, or when the snap
    would produce a degenerate cue (end ≤ start in original-audio
    time — e.g. a 0.05 s cue stamped entirely inside a pad whose
    nearest region boundary is on the wrong side).
    """
    if not region_map:
        return None

    start_sample = cue_start_window_s * sample_rate

    # Binary search over region_map would be marginally faster, but each
    # window has <= ~20 regions in practice — linear scan is simpler and
    # the per-cue cost is dwarfed by the LLM/decoder calls anyway.
    for entry in region_map:
        region_end = entry.window_offset_samples + entry.length_samples
        if entry.window_offset_samples <= start_sample < region_end:
            shift_s = (
                entry.original_start_samples - entry.window_offset_samples
            ) / sample_rate
            # Clamp the end inside the region too, in case Whisper bled
            # into the trailing pad (common at region boundaries).
            region_end_s = region_end / sample_rate
            clamped_end_window_s = min(cue_end_window_s, region_end_s)
            return (
                cue_start_window_s + shift_s,
                clamped_end_window_s + shift_s,
                False,                 # cleanly in-region
            )

    # ── Pad-zone snap recovery (0.7.11+) ────────────────────────────────
    # Pick the region whose nearest boundary is closest to the cue start,
    # then snap the cue's start onto that region's START. Always-START is
    # a deliberate simplification — using the region's END for "snap
    # backwards" cases would assign the cue to the trailing edge of a
    # region's audio context, which usually corresponds to a word that
    # was already captured by the previous cue. Snapping to the next
    # region's start keeps the cue ordering monotonic and assigns it to
    # whichever region's audio Whisper was actually decoding.
    def _distance_to_region(entry: "RegionEntry") -> float:
        region_end = entry.window_offset_samples + entry.length_samples
        return min(
            abs(start_sample - entry.window_offset_samples),
            abs(start_sample - region_end),
        )

    closest = min(region_map, key=_distance_to_region)
    snapped_start_window_s = closest.window_offset_samples / sample_rate
    shift_s = (
        closest.original_start_samples - closest.window_offset_samples
    ) / sample_rate
    closest_end_s = (
        closest.window_offset_samples + closest.length_samples
    ) / sample_rate

    # Preserve the cue's original duration where the region has room.
    # If the original end timestamp was further out than the region's
    # end, clamp to the region's end (same logic as the in-region path).
    original_duration_s = max(0.0, cue_end_window_s - cue_start_window_s)
    if original_duration_s == 0:
        # Degenerate input — drop rather than fabricate.
        return None
    snapped_end_window_s = min(
        snapped_start_window_s + original_duration_s,
        closest_end_s,
    )
    if snapped_end_window_s <= snapped_start_window_s:
        return None

    return (
        snapped_start_window_s + shift_s,
        snapped_end_window_s + shift_s,
        True,                          # snapped from pad zone
    )
