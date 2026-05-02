"""ISO language code normalization. ffprobe emits 3-letter ISO 639-2; the rest
of the pipeline (Whisper, our user-facing API) speaks 2-letter ISO 639-1."""

_THREE_TO_TWO = {
    "eng": "en", "fra": "fr", "fre": "fr", "spa": "es", "deu": "de", "ger": "de",
    "ita": "it", "por": "pt", "rus": "ru", "jpn": "ja", "kor": "ko", "zho": "zh",
    "chi": "zh", "ara": "ar", "hin": "hi", "ben": "bn", "tur": "tr", "vie": "vi",
    "tha": "th", "pol": "pl", "nld": "nl", "dut": "nl", "swe": "sv", "nor": "no",
    "dan": "da", "fin": "fi", "ces": "cs", "cze": "cs", "ell": "el", "gre": "el",
    "heb": "he", "hun": "hu", "ron": "ro", "rum": "ro", "ukr": "uk", "ind": "id",
    "msa": "ms", "may": "ms", "fil": "tl", "tgl": "tl", "cat": "ca",
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
