"""Language pack registry and resolution."""

from __future__ import annotations

from voxalign.languages.base import BaseLanguagePack
from voxalign.languages.english import ENGLISH_PACK
from voxalign.languages.generic import GENERIC_PACK

_LANGUAGE_PACKS: dict[str, BaseLanguagePack] = {
    "en": ENGLISH_PACK,
    "und": GENERIC_PACK,
}

_ALIASES = {
    "auto": "und",
    "en-us": "en",
    "en-gb": "en",
    "en-ca": "en",
    "en-au": "en",
}


def resolve_language_pack(language_code: str) -> BaseLanguagePack:
    """Resolve a language code to the best available language pack."""
    canonical = _ALIASES.get(language_code.casefold(), language_code.casefold())
    return _LANGUAGE_PACKS.get(canonical, GENERIC_PACK)
