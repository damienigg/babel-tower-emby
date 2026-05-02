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
    *,
    scene_threshold: float | None = None,
    translation_llm_model: str | None = None,
    vision_llm_model: str | None = None,
) -> str:
    """Build a stable cache key. Each kwarg is included only when relevant —
    callers pass None for kwargs that don't affect the output for their request:

    - `scene_threshold`: relevant for scene/cinematic modes. Different threshold
      → different scene bible → different final VTT.
    - `translation_llm_model`: relevant when provider="llm". Different LLM model
      → different translation output. Switching from claude-opus-4-7 to
      gpt-4o or qwen2.5:72b must invalidate the cache.
    - `vision_llm_model`: relevant for scene/cinematic modes (the bible content
      depends on which LLM described the keyframes).
    """
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
    if translation_llm_model:
        parts.append(f"tllm={translation_llm_model}")
    if vision_llm_model:
        parts.append(f"vllm={vision_llm_model}")
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
