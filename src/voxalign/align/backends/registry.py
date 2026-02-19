"""Alignment backend registry."""

from __future__ import annotations

from voxalign.align.backends.base import AlignmentBackend, BackendName
from voxalign.align.backends.ctc_trellis import CtcTrellisBackend
from voxalign.align.backends.phoneme_first import PhonemeFirstBackend
from voxalign.align.backends.uniform import UniformBackend

_BACKENDS: dict[BackendName, AlignmentBackend] = {
    "uniform": UniformBackend(),
    "ctc_trellis": CtcTrellisBackend(),
    "phoneme_first": PhonemeFirstBackend(),
}


def resolve_backend(name: BackendName) -> AlignmentBackend:
    """Resolve backend name to implementation."""
    return _BACKENDS[name]
