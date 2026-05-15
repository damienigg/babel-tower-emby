"""LLM-backed translation provider. Delegates to the configured LLM
backend (Anthropic native or OpenAI-compatible) via app.pipeline.llm.
The translation prompt and JSON schema are the same regardless of
backend; only the wire format differs.

Pre-0.7.32 this module also threaded a scene bible + per-cue keyframes
into the LLM call for the "scene" and "cinematic" modes. Those modes
were removed; the provider is now text-only. The ImageContent /
multimodal plumbing in ``app.pipeline.llm`` is no longer reached from
here but stays in place — translation just doesn't use it.
"""
import json
from typing import Callable

from app.config import settings
from app.pipeline.llm import (
    ContentBlock, LLMError, SystemBlock, TextContent, get_translation_llm,
)
from app.pipeline.stt import Cue
from app.pipeline.translate._util import batches
from app.pipeline.translate.base import TranslationError


def _noop_progress(frac: float) -> None: ...
def _noop_cancel() -> None: ...


_SYSTEM_PROMPT = """You are a professional subtitle translator producing high-quality dialogue subtitles for an audiovisual production.

# Translation principles
- Produce natural, idiomatic target-language phrasing. Avoid word-for-word literal translation when it sounds unnatural.
- Preserve speaker tone, register (formal/informal), and emotional content of the original line.
- Use cultural and idiomatic equivalents for slang, idioms, and culturally-specific references when a direct translation would lose meaning.
- Preserve proper nouns (names of people, places, brands) unless the target language has an established convention for them.
- For ambiguous gender or number, choose the most natural option in the target language given the surrounding context.
- When the source uses profanity, render it at the same intensity in the target language. Do not soften or strengthen.
- Honorifics, titles, and forms of address should be adapted to target-language conventions.
- Numbers, dates, currencies, and units of measurement follow target-language formatting conventions when natural to do so.

# Subtitle constraints
- Cues must be concise. Prefer shorter, punchier translations over wordy ones; subtitles compete with the picture for the viewer's attention.
- Match the emotional pacing of the source: short staccato lines must remain short.
- Avoid restating information that is already obvious from preceding cues.
- Punctuation should follow target-language conventions, not the source's.

# Output format
You will receive a JSON array of subtitle cues, each with an integer `id` and a `text` field.
Return a JSON object with a `translations` array of the same length, in the same order, where each entry has the matching `id` and the translated `text`.
Do not add, remove, reorder, merge, or split cues. The output array length must exactly equal the input array length, with the same ids in the same order.
"""


_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["translations"],
    "additionalProperties": False,
}


class LLMTranslationProvider:
    def __init__(self) -> None:
        try:
            self._client = get_translation_llm()
        except LLMError as e:
            raise TranslationError(str(e)) from e

    def translate(
        self,
        cues: list[Cue],
        source_lang: str,
        target_lang: str,
        *,
        progress: Callable[[float], None] = _noop_progress,
        check_cancel: Callable[[], None] = _noop_cancel,
    ) -> list[Cue]:
        batch_size = settings.translation_batch_size

        out: list[Cue] = []
        total = max(1, len(cues))
        for batch in batches(cues, batch_size):
            check_cancel()
            out.extend(self._translate_batch(batch, source_lang, target_lang))
            progress(len(out) / total)
        progress(1.0)
        return out

    def _translate_batch(
        self,
        batch: list[Cue],
        source_lang: str,
        target_lang: str,
    ) -> list[Cue]:
        payload = [{"id": c.id, "text": c.text} for c in batch]

        user_content: list[ContentBlock] = [
            TextContent(text=json.dumps(payload, ensure_ascii=False)),
        ]

        # System prefix: cacheable principles, then per-job lang config
        # (cacheable too — putting it last lets the LLM provider's
        # cache mark every preceding block as cacheable in a single
        # marker).
        system: list[SystemBlock] = [
            SystemBlock(text=_SYSTEM_PROMPT, cacheable=True),
            SystemBlock(
                text=f"Source language: {source_lang}\nTarget language: {target_lang}",
                cacheable=True,
            ),
        ]

        try:
            text = self._client.chat(
                system=system,
                content=user_content,
                max_tokens=16000,
                response_schema=_OUTPUT_SCHEMA,
            )
        except LLMError as e:
            raise TranslationError(str(e)) from e

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise TranslationError(f"LLM returned invalid JSON: {e}") from e

        translations = parsed.get("translations", [])
        if len(translations) != len(batch):
            raise TranslationError(
                f"Length mismatch: expected {len(batch)} cues, got {len(translations)}"
            )

        # Detect duplicate ids before dict-deduplication silently drops one
        # of them. The previous `{t["id"]: t["text"] for t in translations}`
        # would happily accept `[{id:0, ...}, {id:0, ...}, {id:1, ...}]` for
        # a 3-cue batch — losing cue 2's translation under cue 0's text.
        # Length check above catches the wrong-count case; this catches the
        # right-count-wrong-distinct case.
        ids_seen = [t["id"] for t in translations]
        if len(set(ids_seen)) != len(ids_seen):
            duplicates = sorted({i for i in ids_seen if ids_seen.count(i) > 1})
            raise TranslationError(
                f"Duplicate cue id(s) {duplicates} in translation response — "
                f"model dropped or duplicated cues"
            )

        by_id = {t["id"]: t["text"] for t in translations}
        out: list[Cue] = []
        for c in batch:
            if c.id not in by_id:
                raise TranslationError(f"Missing translation for cue id {c.id}")
            out.append(Cue(id=c.id, start=c.start, end=c.end, text=by_id[c.id]))
        return out
