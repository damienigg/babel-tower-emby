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
        if cur_used + needed > pack_capacity:
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
) -> tuple[float, float] | None:
    """Given a cue's window-relative (start, end) seconds, find which
    packed region it belongs to and return its original-audio (start, end).

    Returns None when the cue's start falls inside a pad zone — those
    cues are Whisper hallucinations on the silence between packed
    regions and should be dropped (they don't correspond to real audio).
    """
    # Binary search over region_map would be marginally faster, but each
    # window has <= ~20 regions in practice — linear scan is simpler and
    # the per-cue cost is dwarfed by the LLM/decoder calls anyway.
    start_sample = cue_start_window_s * sample_rate
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
            )
    return None   # cue starts in a pad zone — drop
