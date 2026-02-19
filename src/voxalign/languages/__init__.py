"""Language packs and transcript normalization."""

from voxalign.languages.base import BaseLanguagePack, NormalizedTranscript
from voxalign.languages.registry import resolve_language_pack

__all__ = ["BaseLanguagePack", "NormalizedTranscript", "resolve_language_pack"]
