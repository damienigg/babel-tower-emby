import hashlib
import json
from pathlib import Path

from app.config import settings


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    h = hashlib.sha256()
    h.update(str(path.resolve()).encode())
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    return h.hexdigest()[:16]


def cache_key(
    media_fingerprint: str,
    target_lang: str,
    model: str,
    provider: str,
    source_priority: list[str],
    mode: str,
    scene_threshold: float | None = None,
) -> str:
    """`scene_threshold` is included only for scene/cinematic modes — when set,
    a different threshold produces a different bible and therefore a different
    final VTT. Audio mode passes None so its keys stay stable."""
    parts = [
        media_fingerprint,
        target_lang,
        model,
        provider,
        ",".join(source_priority),
        mode,
    ]
    if scene_threshold is not None:
        parts.append(f"thr={scene_threshold:.3f}")
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def cache_path(key: str) -> Path:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.cache_dir / f"{key}.json"


def load(key: str) -> dict | None:
    p = cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        # Corrupt or unreadable cache file — treat as a miss so we recompute.
        # The next store() call will overwrite it cleanly.
        return None


def store(key: str, payload: dict) -> None:
    cache_path(key).write_text(json.dumps(payload))
