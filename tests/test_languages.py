from voxalign.languages import resolve_language_pack


def test_resolve_language_pack_alias() -> None:
    pack = resolve_language_pack("en-US")
    assert pack.code == "en"
    assert pack.normalizer_id == "english-basic-v1"


def test_english_normalization_rules() -> None:
    pack = resolve_language_pack("en")
    result = pack.normalize("  Hello,  WORLD!!! It's   me.  ")

    assert result.normalized == "hello world it's me"
    assert result.tokens == ["hello", "world", "it's", "me"]


def test_generic_fallback_unknown_language() -> None:
    pack = resolve_language_pack("ko")
    result = pack.normalize("안녕  세상!!! hello")

    assert pack.code == "ko"
    assert pack.normalizer_id == "generic-unicode-v1"
    assert result.tokens == ["안녕", "세상", "hello"]


def test_european_language_resolution() -> None:
    pack = resolve_language_pack("fr")
    result = pack.normalize("Bonjour le monde!")

    assert pack.code == "fr"
    assert pack.normalizer_id == "generic-unicode-v1"
    assert result.tokens == ["bonjour", "le", "monde"]
