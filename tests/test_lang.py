from app.pipeline.lang import normalize, to_iso6392


def test_normalize_passthrough_two_letter():
    assert normalize("en") == "en"
    assert normalize("fr") == "fr"


def test_normalize_three_letter_to_two():
    assert normalize("eng") == "en"
    assert normalize("fra") == "fr"
    assert normalize("fre") == "fr"
    assert normalize("jpn") == "ja"
    assert normalize("zho") == "zh"
    assert normalize("chi") == "zh"


def test_normalize_uppercase():
    assert normalize("ENG") == "en"
    assert normalize("EN") == "en"


def test_normalize_locale_suffix():
    assert normalize("en-US") == "en"
    assert normalize("zh-Hans") == "zh"


def test_normalize_unknown_returns_none():
    assert normalize("xyz") is None
    assert normalize("not-a-language") is None


def test_normalize_und_zxx_mul_treated_as_unknown():
    assert normalize("und") is None
    assert normalize("zxx") is None
    assert normalize("mul") is None


def test_normalize_none_or_empty():
    assert normalize(None) is None
    assert normalize("") is None


def test_to_iso6392_two_letter_to_three():
    assert to_iso6392("en") == "eng"
    assert to_iso6392("fr") == "fra"
    assert to_iso6392("ja") == "jpn"
    assert to_iso6392("zh") == "zho"
    assert to_iso6392("de") == "deu"


def test_to_iso6392_passthrough_three_letter():
    assert to_iso6392("eng") == "eng"
    assert to_iso6392("fra") == "fra"
    # Bibliographic codes also stay valid since they're in our forward map.
    assert to_iso6392("fre") == "fre"


def test_to_iso6392_unknown_returns_none():
    assert to_iso6392("xyz") is None
    assert to_iso6392("not-a-language") is None
    assert to_iso6392(None) is None
    assert to_iso6392("") is None


def test_to_iso6392_uses_terminological_codes():
    """Where a language has both bibliographic ('fre') and terminological
    ('fra') codes, prefer the modern terminological one for write-back —
    it's what current Matroska/MP4 tools expect."""
    assert to_iso6392("fr") == "fra"
    assert to_iso6392("de") == "deu"
    assert to_iso6392("zh") == "zho"
    assert to_iso6392("cs") == "ces"
    assert to_iso6392("el") == "ell"
