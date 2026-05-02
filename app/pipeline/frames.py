"""In-memory single-frame JPEG extraction via ffmpeg. No temp files —
returns the encoded bytes directly so callers can base64-encode them for
multimodal API calls."""
import subprocess


def extract_frame_bytes(media_path: str, timestamp: float, max_size: int = 1024) -> bytes:
    """Extract one JPEG frame at `timestamp` seconds, scaled so the long edge
    is at most `max_size` pixels. Uses ffmpeg's input-seek (-ss before -i) for
    speed; precision is to the nearest keyframe, which is fine for our use.
    """
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error",
         "-ss", str(timestamp),
         "-i", media_path,
         "-frames:v", "1",
         "-vf", f"scale='if(gt(iw,ih),min({max_size},iw),-2)':'if(gt(iw,ih),-2,min({max_size},ih))'",
         "-q:v", "3",
         "-f", "image2pipe",
         "-vcodec", "mjpeg",
         "-"],
        capture_output=True, check=True,
    )
    return result.stdout
