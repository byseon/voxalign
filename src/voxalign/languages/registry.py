"""Language pack registry and resolution."""

from __future__ import annotations

from voxalign.languages.base import BaseLanguagePack
from voxalign.languages.english import ENGLISH_PACK
from voxalign.languages.generic import GENERIC_PACK, GenericLanguagePack

_EUROPEAN_CODES = {
    "bg",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "es",
    "et",
    "eu",
    "fi",
    "fr",
    "ga",
    "gl",
    "hr",
    "hu",
    "is",
    "it",
    "lt",
    "lv",
    "mk",
    "mt",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
}

_EXTRA_GENERIC_LANGUAGE_PACKS: dict[str, BaseLanguagePack] = {
    "ko": GenericLanguagePack(code="ko", name="Korean"),
}
for code in sorted(_EUROPEAN_CODES):
    _EXTRA_GENERIC_LANGUAGE_PACKS[code] = GenericLanguagePack(code=code, name=code.upper())

_LANGUAGE_PACKS: dict[str, BaseLanguagePack] = {
    "en": ENGLISH_PACK,
    "und": GENERIC_PACK,
    **_EXTRA_GENERIC_LANGUAGE_PACKS,
}

_ALIASES = {
    "auto": "und",
    "en-us": "en",
    "en-gb": "en",
    "en-ca": "en",
    "en-au": "en",
    "ko-kr": "ko",
}


def resolve_language_pack(language_code: str) -> BaseLanguagePack:
    """Resolve a language code to the best available language pack."""
    canonical = _ALIASES.get(language_code.casefold(), language_code.casefold())
    return _LANGUAGE_PACKS.get(canonical, GENERIC_PACK)
