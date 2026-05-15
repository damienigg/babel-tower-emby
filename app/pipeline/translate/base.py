from typing import Callable, Protocol

from app.pipeline.stt import Cue


class TranslationError(Exception):
    pass


def _noop_progress(frac: float) -> None: ...
def _noop_cancel() -> None: ...


class TranslationProvider(Protocol):
    def translate(
        self,
        cues: list[Cue],
        source_lang: str,
        target_lang: str,
        *,
        progress: Callable[[float], None] = _noop_progress,
        check_cancel: Callable[[], None] = _noop_cancel,
    ) -> list[Cue]:
        """Translate cues. Timing (start/end) and ids must be preserved;
        the returned list length must match the input.

        Pre-0.7.32 the protocol carried an optional ``context``
        argument (``TranslationContext``) used by the scene / cinematic
        modes to thread a scene bible + per-cue keyframes through to
        the translator. Those modes were removed and so was the
        context parameter — providers are now text-only.

        ``progress`` is called with fractional completion in [0,1] as
        batches finish; ``check_cancel`` is called between batches and
        raises JobCanceled if the user has clicked cancel.
        """
        ...
