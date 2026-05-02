"""Frozen specialist model organs for the Mosaic cognitive substrate.

Each organ is a frozen pre-trained model that handles a specific cognitive
modality. Organs communicate with the substrate via a uniform interface:
they accept raw input (image, audio, text) and produce a fixed-dimensional
representation that can be projected into the substrate's cognitive frame
space or used directly by the algebraic components.

Architecture:
    Raw input → Frozen Organ → [trainable projector] → Substrate → [grafts] → Frozen LLM

Brain-area mapping:
    - Perception (V-JEPA, I-JEPA, DINOv2, Depth Anything) → Visual cortex
    - Auditory (Whisper) → Auditory cortex
    - Binding (ImageBind) → Multi-sensory association cortex
    - Extraction (GLiNER2) → Wernicke's area (language comprehension)
    - Affect (GoEmotions) → Limbic system / insula

All organs are:
    - Frozen (weights never updated)
    - Lazy-loaded (only instantiated when first used)
    - Device-aware (share device with the substrate)
    - Fallback-safe (graceful degradation when deps missing)
"""

from __future__ import annotations

from .base import Organ, OrganRegistry, OrganOutput
from .extraction import ExtractionOrgan
from .affect import AffectOrgan

__all__ = [
    "Organ",
    "OrganRegistry",
    "OrganOutput",
    "ExtractionOrgan",
    "AffectOrgan",
]

# Lazy imports for heavy organs (avoid importing torch-heavy vision/audio at module level)
def __getattr__(name: str):
    _lazy = {
        "PerceptionOrgan": (".perception", "PerceptionOrgan"),
        "AuditoryOrgan": (".auditory", "AuditoryOrgan"),
        "BindingOrgan": (".binding", "BindingOrgan"),
    }
    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
