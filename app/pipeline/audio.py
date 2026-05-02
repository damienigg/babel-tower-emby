import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def extract_audio(media_path: str, track_index: int):
    """Extract a single audio track to a 16kHz mono WAV temp file."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
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
        )
        yield out_path
    finally:
        out_path.unlink(missing_ok=True)
