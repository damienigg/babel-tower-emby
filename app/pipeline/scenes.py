"""ffmpeg-based scene detection and cue→scene mapping.

We use ffmpeg's `select='gt(scene,T)'` filter rather than PySceneDetect/opencv
so there are no new heavy dependencies — ffmpeg is already in the image.

Implementation note — streaming + cancel:
ffmpeg has to decode every frame to evaluate the `scene` metric, which on a
2 h film at 4K runs for many minutes of full CPU. The previous implementation
ran ffmpeg via subprocess.run(..., capture_output=True), which (a) buffered
the entire stderr in RAM until completion, (b) could not be canceled, and
(c) had no timeout — a wedged ffmpeg blocked the runner indefinitely. We now
spawn ffmpeg via Popen and read stderr line by line, calling `check_cancel`
between lines so a UI cancel takes effect on the next ffmpeg log line
(typically <2 s on a working ffmpeg). On cancel/timeout we terminate the
subprocess cleanly so the kernel reclaims the decoder threads.
"""
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

from app.pipeline.stt import Cue


@dataclass
class Scene:
    index: int
    start: float
    end: float
    description: str | None = None   # populated by scene_bible.describe_scenes


def _noop_cancel() -> None: ...


def _media_duration(media_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media_path],
        capture_output=True, text=True, check=True,
        timeout=30,
    )
    return float(result.stdout.strip())


def detect_scenes(
    media_path: str,
    *,
    threshold: float = 0.4,
    min_length_seconds: float = 1.5,
    max_scenes: int = 500,
    check_cancel: Callable[[], None] = _noop_cancel,
) -> list[Scene]:
    """Run ffmpeg's scene filter and return scene boundaries.

    `threshold` is in [0, 1]. Lower → more boundaries found. 0.3-0.5 is typical
    for film/TV; horror or sports may want lower to catch quick cuts.

    `check_cancel` is called between every ffmpeg stderr line so a user cancel
    (or wall-clock timeout) reaches the inner ffmpeg promptly. Without this
    the user could not stop a long scene-detection pass at all.
    """
    # `-an`: skip audio decoding entirely — scene detection is video-only,
    # decoding the audio stream alongside is pure waste of CPU+RAM.
    # `-loglevel info`: keep showinfo lines (the scene markers we parse) but
    # drop higher-noise levels. We can't go to `error` because that would
    # suppress the `pts_time:` lines.
    proc = subprocess.Popen(
        ["ffmpeg", "-nostdin", "-loglevel", "info", "-i", media_path,
         "-an",
         "-filter:v", f"select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    times: list[float] = []
    canceled_exc: BaseException | None = None
    try:
        # Iterating a text-mode pipe yields lines as ffmpeg flushes them.
        # Each pts_time:N marker corresponds to one detected scene boundary.
        # We parse incrementally so the stderr buffer never grows unbounded.
        for line in proc.stderr:                         # type: ignore[union-attr]
            try:
                check_cancel()
            except BaseException as e:
                # Don't raise inside the iteration — we want to terminate
                # the subprocess cleanly first (in the finally block).
                canceled_exc = e
                break
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                times.append(float(m.group(1)))
    finally:
        # Either ffmpeg finished naturally (proc.stderr hit EOF), or the
        # caller canceled. Either way: shut the subprocess down cleanly.
        # terminate() then a short wait, kill() if it didn't exit.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    if canceled_exc is not None:
        # JobCanceled / JobTimeout / KeyboardInterrupt — propagate after
        # the subprocess is reaped.
        raise canceled_exc

    duration = _media_duration(media_path)
    boundaries = [0.0] + sorted(set(times)) + [duration]

    scenes: list[Scene] = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        if end - start < min_length_seconds:
            continue
        scenes.append(Scene(index=len(scenes), start=start, end=end))
        if len(scenes) >= max_scenes:
            break
    return scenes


def map_cues_to_scenes(cues: list[Cue], scene_list: list[Scene]) -> dict[int, int]:
    """Map cue.id → scene.index by start time. Falls back to last scene for cues
    that extend past the last detected boundary."""
    if not scene_list:
        return {}
    mapping: dict[int, int] = {}
    for cue in cues:
        for scene in scene_list:
            if scene.start <= cue.start < scene.end:
                mapping[cue.id] = scene.index
                break
        else:
            mapping[cue.id] = scene_list[-1].index
    return mapping


def keyframe_timestamp(scene: Scene, position: str = "midpoint") -> float:
    """Pick a sample timestamp for the scene's representative keyframe."""
    if position == "start":
        return scene.start + 0.1   # nudge past the cut so we don't grab the prev shot
    if position == "end":
        return max(scene.start, scene.end - 0.1)
    return (scene.start + scene.end) / 2.0
