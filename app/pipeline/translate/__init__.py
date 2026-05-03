from app.pipeline.translate.base import TranslationError, TranslationProvider


def get_provider(name: str) -> TranslationProvider:
    name = (name or "").lower()
    # `claude` is a legacy alias from when the only supported LLM was Anthropic.
    # New configurations should use `llm`, which dispatches to whichever LLM
    # backend is configured (Anthropic native or OpenAI-compatible) per the
    # Translation model slot in Settings.
    if name in ("llm", "claude"):
        from app.pipeline.translate.llm import LLMTranslationProvider
        return LLMTranslationProvider()
    if name == "deepl":
        from app.pipeline.translate.deepl import DeepLProvider
        return DeepLProvider()
    if name == "nllb":
        from app.pipeline.translate.nllb import NLLBProvider
        return NLLBProvider()
    raise ValueError(f"Unknown translation provider: {name!r}. Choose llm, deepl, or nllb.")


__all__ = ["TranslationProvider", "TranslationError", "get_provider"]
