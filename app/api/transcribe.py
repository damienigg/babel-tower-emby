"""Public path-based API. Useful for ad-hoc curl calls and external tools that
already know the media path. The Emby-driven flow lives in manage.py."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.processor import (
    BadRequest,
    MediaNotFound,
    NoSpeech,
    NoSuitableTrack,
    NotImplementedYet,
    ProcessError,
    ProcessRequest,
    TranslationFailed,
    process,
)


router = APIRouter()


# Maps each typed pipeline error to its HTTP status. Order doesn't matter; we
# look up by exact class.
_STATUS_MAP: dict[type[ProcessError], int] = {
    NotImplementedYet: 501,
    NoSuitableTrack: 409,
    NoSpeech: 422,
    MediaNotFound: 404,
    BadRequest: 400,
    TranslationFailed: 502,
}


class TranscribeRequest(BaseModel):
    media_path: str
    target_lang: str
    source_lang_priority: list[str] = Field(default_factory=lambda: ["en", "ja", "*"])
    translation_provider: str = "llm"
    mode: str = "audio"   # audio | scene | cinematic — only `audio` works today
    skip_if_target_audio_exists: bool = True


class TrackInfo(BaseModel):
    index: int
    language: str | None
    title: str | None


class TranscribeResponse(BaseModel):
    vtt: str
    source_track: TrackInfo
    detected_source_language: str
    cue_count: int
    mode: str
    cached: bool
    took_seconds: float


@router.post("/transcribe-translate", response_model=TranscribeResponse)
def transcribe_translate(req: TranscribeRequest) -> TranscribeResponse:
    try:
        result = process(ProcessRequest(
            media_path=req.media_path,
            target_lang=req.target_lang,
            source_lang_priority=req.source_lang_priority,
            translation_provider=req.translation_provider,
            mode=req.mode,
            skip_if_target_audio_exists=req.skip_if_target_audio_exists,
        ))
    except ProcessError as e:
        raise HTTPException(_STATUS_MAP.get(type(e), 500), str(e)) from e

    return TranscribeResponse(
        vtt=result.vtt,
        source_track=TrackInfo(
            index=result.source_track_index,
            language=result.source_track_language,
            title=result.source_track_title,
        ),
        detected_source_language=result.detected_source_language,
        cue_count=result.cue_count,
        mode=result.mode,
        cached=result.cached,
        took_seconds=result.took_seconds,
    )
