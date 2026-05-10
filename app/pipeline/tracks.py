import json
import subprocess
from dataclasses import dataclass

from app.pipeline.lang import normalize


@dataclass
class AudioTrack:
    index: int
    language: str | None
    title: str | None
    is_default: bool
    is_dub: bool
    is_original: bool
    is_commentary: bool
    is_audio_description: bool
    channels: int

    @property
    def is_junk(self) -> bool:
        return self.is_commentary or self.is_audio_description


def probe(media_path: str) -> list[AudioTrack]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index,channels:stream_tags=language,title:stream_disposition=default,dub,original,comment,visual_impaired,hearing_impaired",
            "-of", "json",
            media_path,
        ],
        capture_output=True, text=True, check=True,
        # ffprobe on a healthy file completes in ms; 30s is plenty even
        # for slow network mounts. Without a timeout a wedged ffprobe
        # would park the worker indefinitely.
        timeout=30,
    )
    data = json.loads(result.stdout)
    tracks: list[AudioTrack] = []
    for s in data.get("streams", []):
        tags = s.get("tags") or {}
        disp = s.get("disposition") or {}
        title = tags.get("title")
        title_low = (title or "").lower()
        commentary = bool(disp.get("comment")) or any(w in title_low for w in ("commentary",))
        ad = bool(disp.get("visual_impaired")) or any(w in title_low for w in ("audio description", "described", "ad track"))
        tracks.append(AudioTrack(
            index=s["index"],
            language=normalize(tags.get("language")),
            title=title,
            is_default=bool(disp.get("default")),
            is_dub=bool(disp.get("dub")),
            is_original=bool(disp.get("original")),
            is_commentary=commentary,
            is_audio_description=ad,
            channels=s.get("channels", 0),
        ))
    return tracks


def select(
    tracks: list[AudioTrack],
    target_lang: str,
    source_priority: list[str],
    skip_if_target_audio_exists: bool,
) -> AudioTrack | None:
    candidates = [t for t in tracks if not t.is_junk]
    if not candidates:
        return None

    if skip_if_target_audio_exists and any(t.language == target_lang for t in candidates):
        return None

    def rank(t: AudioTrack) -> tuple:
        lang_rank = len(source_priority)
        for i, code in enumerate(source_priority):
            if code == "*" or t.language == code:
                lang_rank = i
                break
        # Prefer original over dub when otherwise equal
        dub_penalty = 1 if t.is_dub and not t.is_original else 0
        # Tie-break with default disposition
        default_bonus = 0 if t.is_default else 1
        return (lang_rank, dub_penalty, default_bonus, t.index)

    return min(candidates, key=rank)
