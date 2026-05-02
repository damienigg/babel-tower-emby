from app.pipeline.lang import normalize


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
