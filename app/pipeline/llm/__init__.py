"""LLM client factory for the translation slot.

Pre-0.7.32 there was a second slot for vision (scene bible building +
cinematic per-cue frames). Both consumer paths were removed; only the
translation slot survives. Translation is text-only and does not need
a vision-capable model.
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
    """LLM used for subtitle translation. Text-only since 0.7.32 (the
    vision-aware cinematic path was removed). ``supports_vision`` is
    hard-coded False below for the openai_compat backend — even if the
    underlying model can do vision, the translation pipeline doesn't
    send images anymore."""
    return _build(
        type_=settings.translation_llm_type,
        model=settings.translation_llm_model,
        endpoint=settings.translation_llm_endpoint,
        api_key=settings.translation_llm_api_key,
        supports_vision=False,
    )


__all__ = [
    "get_translation_llm",
    "LLMClient", "LLMError",
    "SystemBlock", "ContentBlock", "TextContent", "ImageContent",
]
