"""ISO language code normalization. ffprobe emits 3-letter ISO 639-2; the rest
of the pipeline (Whisper, our user-facing API) speaks 2-letter ISO 639-1.
Container metadata writers (mkvpropedit, ffmpeg -metadata) want 639-2 again,
so we also expose the reverse mapping for the language write-back path."""

_THREE_TO_TWO = {
    "eng": "en", "fra": "fr", "fre": "fr", "spa": "es", "deu": "de", "ger": "de",
    "ita": "it", "por": "pt", "rus": "ru", "jpn": "ja", "kor": "ko", "zho": "zh",
    "chi": "zh", "ara": "ar", "hin": "hi", "ben": "bn", "tur": "tr", "vie": "vi",
    "tha": "th", "pol": "pl", "nld": "nl", "dut": "nl", "swe": "sv", "nor": "no",
    "dan": "da", "fin": "fi", "ces": "cs", "cze": "cs", "ell": "el", "gre": "el",
    "heb": "he", "hun": "hu", "ron": "ro", "rum": "ro", "ukr": "uk", "ind": "id",
    "msa": "ms", "may": "ms", "fil": "tl", "tgl": "tl", "cat": "ca",
}

# Reverse map for the metadata write-back path. Where a language has both
# bibliographic and terminological 3-letter codes (fr → fra/fre, de → deu/ger,
# zh → zho/chi, etc.), we pick the modern terminological code — that's what
# Matroska / MP4 metadata writers expect.
_TWO_TO_THREE = {
    "en": "eng", "fr": "fra", "es": "spa", "de": "deu", "it": "ita",
    "pt": "por", "ru": "rus", "ja": "jpn", "ko": "kor", "zh": "zho",
    "ar": "ara", "hi": "hin", "bn": "ben", "tr": "tur", "vi": "vie",
    "th": "tha", "pl": "pol", "nl": "nld", "sv": "swe", "no": "nor",
    "da": "dan", "fi": "fin", "cs": "ces", "el": "ell", "he": "heb",
    "hu": "hun", "ro": "ron", "uk": "ukr", "id": "ind", "ms": "msa",
    "tl": "tgl", "ca": "cat",
}


def normalize(code: str | None) -> str | None:
    """Return a lowercase 2-letter ISO 639-1 code, or None if unknown/missing."""
    if not code:
        return None
    code = code.lower().strip()
    if code in ("und", "zxx", "mul", ""):
        return None
    if len(code) == 2:
        return code
    if code in _THREE_TO_TWO:
        return _THREE_TO_TWO[code]
    if "-" in code:  # e.g. "en-US"
        return normalize(code.split("-", 1)[0])
    return None


def to_iso6392(code: str | None) -> str | None:
    """Return a 3-letter ISO 639-2 code for embedding in container metadata
    (Matroska, MP4). Accepts either an already-3-letter code or a 2-letter
    code; returns None if we don't have a mapping."""
    if not code:
        return None
    code = code.lower().strip()
    if len(code) == 3 and code in _THREE_TO_TWO:
        return code   # already 3-letter, in our known set
    if code in _TWO_TO_THREE:
        return _TWO_TO_THREE[code]
    return None
