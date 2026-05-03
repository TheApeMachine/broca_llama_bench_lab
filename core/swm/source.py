"""Provenance tags for SWM slot writers.

Each slot carries the identity of the organ (or substrate operator) that
produced it. Provenance is used by the recursion controller to decide which
organs may overwrite which slots and by introspection tooling to trace the
origin of a thought.
"""

from __future__ import annotations

from enum import Enum


class SWMSource(str, Enum):
    """Identifies which organ or operator produced an SWM slot."""

    LLAMA = "llama"
    GLINER2 = "gliner2"
    GLICLASS = "gliclass"
    VJEPA = "vjepa"
    DINOV2 = "dinov2"
    DEPTH = "depth"
    WHISPER = "whisper"
    SUBSTRATE_ALGEBRA = "substrate_algebra"
    EXTERNAL_INPUT = "external_input"
