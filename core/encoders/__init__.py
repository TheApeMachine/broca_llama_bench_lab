"""Frozen specialist model encoders for the Mosaic cognitive substrate.

Each encoder is a frozen pre-trained model that handles a specific input
modality. Encoders expose a uniform interface: they accept raw input (image,
audio, text) and produce a fixed-dimensional representation that can be
projected into the substrate's cognitive frame space or used directly by the
algebraic components.

Architecture:
    Raw input → Frozen encoder → [trainable projector] → Substrate → [grafts] → Frozen LLM

Included modalities:
    - Perception (DINOv2, I-JEPA, V-JEPA, Depth Anything)
    - Auditory (Whisper)
    - Binding (ImageBind)
    - Extraction (GLiNER2)
    - Semantic classification (GLiClass)
    - Affect (GoEmotions)

All encoders are:
    - Frozen (weights never updated)
    - Lazy-loaded (only instantiated when first used)
    - Device-aware (share device with the substrate)
    - Strict (missing dependencies or incomplete outputs raise immediately)
"""

from __future__ import annotations

from .base import Encoder, EncoderRegistry, EncoderOutput
from .classification import SemanticClassificationEncoder
from .extraction import ExtractionEncoder
from .affect import AffectEncoder

__all__ = [
    "Encoder",
    "EncoderRegistry",
    "EncoderOutput",
    "SemanticClassificationEncoder",
    "ExtractionEncoder",
    "AffectEncoder",
    "DINOv2Encoder",
    "IJEPAEncoder",
    "VJEPAEncoder",
    "DepthEncoder",
    "create_perception_encoders",
    "AuditoryEncoder",
    "BindingEncoder",
]

# Lazy imports for heavy encoders (avoid importing torch-heavy vision/audio at module level)
def __getattr__(name: str):
    _lazy = {
        "DINOv2Encoder": (".perception", "DINOv2Encoder"),
        "IJEPAEncoder": (".perception", "IJEPAEncoder"),
        "VJEPAEncoder": (".perception", "VJEPAEncoder"),
        "DepthEncoder": (".perception", "DepthEncoder"),
        "create_perception_encoders": (".perception", "create_perception_encoders"),
        "AuditoryEncoder": (".auditory", "AuditoryEncoder"),
        "BindingEncoder": (".binding", "BindingEncoder"),
    }
    if name in _lazy:
        module_path, attr = _lazy[name]
        import importlib
        mod = importlib.import_module(module_path, __package__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
