"""Audio extraction from arbitrary container formats into a 16 kHz mono
WAV that the Whisper backends can consume.

Why a temp WAV (not a pipe): both STT backends want a `pathlib.Path` they
can pass to soundfile.SoundFile / faster_whisper. Piping ffmpeg→stdin→
soundfile is doable but the cleanup path on cancel/timeout is fragile,
and the disk roundtrip is cheap (~250 MB write at sequential IO speeds
for a 2 h film). Acceptable.

Why temp file lives under settings.cache_dir, NOT /tmp:
A 2 h mono-16 kHz 16-bit WAV is ~250 MB. On TrueNAS Scale, /tmp is often
backed by tmpfs (or a tiny system dataset) — every temp wav we put there
counts against host memory AND can collide with the container's 6 GB
cgroup limit if multiple jobs queue up. Putting them in
`<cache_dir>/tmp` lands them on the same persistent volume the user
already sized for the model cache, which is bind-mounted from /mnt/cache
in the default compose. No bytes "spill" into host RAM.
"""
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.config import settings


# Hard cap on how long audio extraction can run. A typical 2 h film
# extracts in 1-2 min; we set this generously at 60 min so even slow
# network mounts or huge episodics (multi-hour concert recordings)
# don't hit the wall. The job-level wall-clock timeout is the real
# fence — this is just defense-in-depth against a wedged ffmpeg.
_AUDIO_EXTRACT_TIMEOUT_SECONDS = 3600


def _tmp_dir() -> Path:
    """Return the directory we put temp wavs in, creating it if needed.
    Reads settings.cache_dir each call so test fixtures that swap the
    cache_dir work without restart."""
    d = Path(settings.cache_dir) / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextmanager
def extract_audio(media_path: str, track_index: int):
    """Extract a single audio track to a 16kHz mono WAV temp file under
    settings.cache_dir/tmp/. Yields the path; deletes it on context exit
    even when the caller raised.
    """
    # delete=False so we control teardown in the `finally` (the with-block
    # would clobber the path on __exit__ before we yield).
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, dir=str(_tmp_dir()),
    ) as tmp:
        out_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                "-i", media_path,
                "-map", f"0:{track_index}",
                "-ac", "1",
                "-ar", "16000",
                "-c:a", "pcm_s16le",
                str(out_path),
            ],
            check=True,
            timeout=_AUDIO_EXTRACT_TIMEOUT_SECONDS,
        )
        yield out_path
    finally:
        out_path.unlink(missing_ok=True)
