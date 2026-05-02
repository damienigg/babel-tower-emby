"""Backend-agnostic LLM client interface. All callers (translation, scene
bible) speak this protocol so swapping Anthropic ↔ any OpenAI-compatible
endpoint (OpenAI, Ollama, LocalAI, OpenRouter, Together, Groq, Gemini's
compat endpoint, vLLM, etc.) is a single config flip.
"""
from dataclasses import dataclass
from typing import Protocol, Union


class LLMError(Exception):
    pass


@dataclass
class SystemBlock:
    """A piece of the system prompt. `cacheable=True` hints to backends that
    support prompt caching (Anthropic) that this block (and everything before
    it) is stable across calls. Backends without prompt caching ignore the flag."""
    text: str
    cacheable: bool = False


@dataclass
class TextContent:
    text: str


@dataclass
class ImageContent:
    data: bytes                  # raw bytes
    media_type: str = "image/jpeg"


ContentBlock = Union[TextContent, ImageContent]


class LLMClient(Protocol):
    """Single-turn chat client. Translation + scene-bible callers compose a
    list of system blocks and a list of content blocks (text + images), then
    ask the client for a string response."""

    def supports_vision(self) -> bool:
        """True if the configured backend/model can ingest ImageContent blocks."""
        ...

    def chat(
        self,
        *,
        system: list[SystemBlock],
        content: list[ContentBlock],
        max_tokens: int = 16000,
        response_schema: dict | None = None,
    ) -> str:
        """Run a single user-turn completion. `response_schema` is a JSON
        Schema; backends that support it (Anthropic native, OpenAI proper)
        enforce it strictly. Backends that only support free-form JSON mode
        (most OpenAI-compat servers) get the schema injected into the system
        prompt and rely on the model to follow it. Caller still validates
        and parses the result."""
        ...
