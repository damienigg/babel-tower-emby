"""LLM client factories — one slot for translation, one for vision.

Each slot independently picks a backend (anthropic native or OpenAI-compatible)
so users can mix-and-match — e.g. cheap fast text model for translation +
strong vision model for scene descriptions, or a single API key driving both
when one provider serves both slots.
"""
from app.config import settings
from app.pipeline.llm.base import (
    ContentBlock, ImageContent, LLMClient, LLMError, SystemBlock, TextContent,
)


def _build(
    *,
    type_: str,
    model: str,
    endpoint: str,
    api_key: str | None,
    supports_vision: bool,
) -> LLMClient:
    type_ = (type_ or "").lower()
    if type_ == "anthropic":
        from app.pipeline.llm.anthropic import AnthropicLLM
        # Anthropic uses its own SDK with the global Anthropic API; endpoint
        # is implicit. The slot's own api_key is the one and only source.
        return AnthropicLLM(api_key=(api_key or ""), model=model)
    if type_ == "openai_compat":
        from app.pipeline.llm.openai_compat import OpenAICompatLLM
        return OpenAICompatLLM(
            base_url=endpoint,
            api_key=api_key,
            model=model,
            supports_vision=supports_vision,
        )
    raise LLMError(f"Unknown LLM type {type_!r} (expected 'anthropic' or 'openai_compat')")


def get_translation_llm() -> LLMClient:
    """LLM used for subtitle translation. In cinematic mode, also receives
    per-cue keyframes — must be vision-capable for that path to work."""
    return _build(
        type_=settings.translation_llm_type,
        model=settings.translation_llm_model,
        endpoint=settings.translation_llm_endpoint,
        api_key=settings.translation_llm_api_key,
        # For openai_compat translation LLMs, vision is optional (only needed for
        # cinematic mode). Declared via translation_llm_supports_vision.
        supports_vision=bool(settings.translation_llm_supports_vision),
    )


def get_vision_llm() -> LLMClient:
    """LLM used for scene-bible building. Always needs vision."""
    return _build(
        type_=settings.vision_llm_type,
        model=settings.vision_llm_model,
        endpoint=settings.vision_llm_endpoint,
        api_key=settings.vision_llm_api_key,
        supports_vision=True,   # vision LLM is by definition vision-capable
    )


__all__ = [
    "get_translation_llm", "get_vision_llm",
    "LLMClient", "LLMError",
    "SystemBlock", "ContentBlock", "TextContent", "ImageContent",
]
