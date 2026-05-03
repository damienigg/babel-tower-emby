"""Scene-bible generation. Sends keyframes to the configured Vision LLM (any
backend the Vision-model slot points at — Anthropic native, OpenAI, Ollama,
LM Studio, etc.) and fills in Scene.description for each scene. Cached on disk
per (file fingerprint, vision LLM model id, detection threshold) so re-runs
reuse the bible across modes and target languages."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from app.config import settings
from app.pipeline.llm import (
    ImageContent, LLMError, SystemBlock, TextContent, get_vision_llm,
)
from app.pipeline.scenes import Scene
from app.pipeline.translate._util import batches
from app.pipeline.translate.base import TranslationError


_DESCRIBE_PROMPT = """For each frame, write a 1-2 sentence description capturing:
- Characters visible (count, age range, role/relationship if inferable)
- Setting (location, time of day, indoor/outdoor)
- Notable on-screen text (signs, notes, screens) — quote it exactly
- Mood/tone if it's clearly conveyed

Be concise and factual. Do not speculate beyond what's visible. Each description
goes to a translator who needs context for pronouns, gendered agreement, and the
broader scene — keep it terse and informational, not narrative."""


_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["index", "description"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scenes"],
    "additionalProperties": False,
}


def _active_llm_model_id() -> str:
    """The vision LLM's model id. Used as part of the bible cache key so
    switching models invalidates the bible."""
    return settings.vision_llm_model


def _bible_cache_path(media_fingerprint: str) -> Path:
    raw = (
        f"bible|{media_fingerprint}|{_active_llm_model_id()}"
        f"|{settings.scene_detection_threshold:.3f}"
    )
    key = hashlib.sha256(raw.encode()).hexdigest()[:24]
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings.cache_dir / f"scene-bible-{key}.json"


def load_cached_bible(media_fingerprint: str) -> list[Scene] | None:
    p = _bible_cache_path(media_fingerprint)
    if not p.exists():
        return None
    try:
        return [Scene(**d) for d in json.loads(p.read_text())]
    except (json.JSONDecodeError, OSError, TypeError):
        return None


def store_cached_bible(media_fingerprint: str, scene_list: list[Scene]) -> None:
    _bible_cache_path(media_fingerprint).write_text(
        json.dumps([asdict(s) for s in scene_list])
    )


def describe_scenes(scene_list: list[Scene], keyframes: dict[int, bytes]) -> list[Scene]:
    """Fill in Scene.description for each scene whose keyframe bytes are in
    `keyframes` (mapping scene.index → JPEG bytes). Modifies in place."""
    try:
        client = get_vision_llm()
    except LLMError as e:
        raise TranslationError(str(e)) from e

    by_index = {s.index: s for s in scene_list}
    system = [SystemBlock(text=_DESCRIBE_PROMPT, cacheable=True)]

    for batch in batches(scene_list, settings.scene_bible_batch_size):
        content = []
        for scene in batch:
            kf = keyframes.get(scene.index)
            if not kf:
                continue
            content.append(TextContent(text=f"Scene {scene.index}:"))
            content.append(ImageContent(data=kf, media_type="image/jpeg"))
        if not content:
            continue

        try:
            text = client.chat(
                system=system, content=content,
                max_tokens=8000, response_schema=_OUTPUT_SCHEMA,
            )
        except LLMError as e:
            raise TranslationError(f"Scene-bible generation failed: {e}") from e

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise TranslationError(f"LLM returned invalid JSON for scenes: {e}") from e

        for entry in parsed.get("scenes", []):
            scene = by_index.get(entry.get("index"))
            if scene:
                scene.description = entry.get("description") or ""

    return scene_list
