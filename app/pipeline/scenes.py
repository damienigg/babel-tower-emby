"""ffmpeg-based scene detection and cue→scene mapping.

We use ffmpeg's `select='gt(scene,T)'` filter rather than PySceneDetect/opencv
so there are no new heavy dependencies — ffmpeg is already in the image.
"""
import re
import subprocess
from dataclasses import dataclass

from app.pipeline.stt import Cue


@dataclass
class Scene:
    index: int
    start: float
    end: float
    description: str | None = None   # populated by scene_bible.describe_scenes


def _media_duration(media_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", media_path],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def detect_scenes(
    media_path: str,
    *,
    threshold: float = 0.4,
    min_length_seconds: float = 1.5,
    max_scenes: int = 500,
) -> list[Scene]:
    """Run ffmpeg's scene filter and return scene boundaries.

    `threshold` is in [0, 1]. Lower → more boundaries found. 0.3-0.5 is typical
    for film/TV; horror or sports may want lower to catch quick cuts.
    """
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-i", media_path,
         "-filter:v", f"select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True, check=False,
    )
    times: list[float] = []
    for line in result.stderr.splitlines():
        m = re.search(r"pts_time:([\d.]+)", line)
        if m:
            times.append(float(m.group(1)))

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
