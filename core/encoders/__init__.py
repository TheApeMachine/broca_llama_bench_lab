"""Frozen specialist model encoders for the Mosaic cognitive substrate.

The package exports encoder classes lazily so importing a lightweight submodule
such as ``core.encoders.affect`` does not import every model backend.
"""

from __future__ import annotations

from typing import Any

from ..infra.lazy_exports import LazyExportRegistry

_EXPORTS: dict[str, tuple[str, str]] = {
    "Encoder": (".base", "Encoder"),
    "EncoderRegistry": (".base", "EncoderRegistry"),
    "EncoderOutput": (".base", "EncoderOutput"),
    "SemanticClassificationEncoder": (".classification", "SemanticClassificationEncoder"),
    "ExtractionEncoder": (".extraction", "ExtractionEncoder"),
    "AffectEncoder": (".affect", "AffectEncoder"),
    "DINOv2Encoder": (".perception", "DINOv2Encoder"),
    "IJEPAEncoder": (".perception", "IJEPAEncoder"),
    "VJEPAEncoder": (".perception", "VJEPAEncoder"),
    "DepthEncoder": (".perception", "DepthEncoder"),
    "create_perception_encoders": (".perception", "create_perception_encoders"),
    "AuditoryEncoder": (".auditory", "AuditoryEncoder"),
    "BindingEncoder": (".binding", "BindingEncoder"),
}

_registry = LazyExportRegistry(package=__package__ or __name__, exports=_EXPORTS)
__all__ = _registry.names()


def __getattr__(name: str) -> Any:
    return _registry.resolve(globals(), name)


def __dir__() -> list[str]:
    return _registry.dir_entries(globals())
