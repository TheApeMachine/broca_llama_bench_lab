"""Typed multimodal observations entering the cognitive substrate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class CognitiveObservation:
    """A frozen-organ observation before it becomes a workspace frame."""

    modality: str
    source: str
    features: torch.Tensor
    confidence: float
    answer: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modality = str(self.modality).strip().lower()
        source = str(self.source).strip()
        answer = str(self.answer).strip()

        if not modality:
            raise ValueError("CognitiveObservation requires a modality")
        if not source:
            raise ValueError("CognitiveObservation requires a source")
        if not answer:
            raise ValueError("CognitiveObservation requires an answer summary")

        feature_tensor = self.features.detach().float().cpu().flatten()
        if feature_tensor.numel() <= 0:
            raise ValueError("CognitiveObservation features must be non-empty")
        if not torch.isfinite(feature_tensor).all():
            raise ValueError("CognitiveObservation features must be finite")

        conf = float(self.confidence)
        if not math.isfinite(conf):
            raise ValueError("CognitiveObservation confidence must be finite")
        if conf < 0.0 or conf > 1.0:
            raise ValueError("CognitiveObservation confidence must be in [0, 1]")

        evidence = dict(self.evidence)
        json.dumps(evidence, sort_keys=True)

        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "features", feature_tensor)
        object.__setattr__(self, "confidence", conf)
        object.__setattr__(self, "evidence", evidence)

    @property
    def subject(self) -> str:
        return self.modality

    def frame_evidence(self) -> dict[str, Any]:
        feature_norm = float(self.features.norm().item())
        raw_instruments = self.evidence.get("instruments") or []
        if not isinstance(raw_instruments, list) or not all(isinstance(item, str) for item in raw_instruments):
            raise TypeError("CognitiveObservation evidence instruments must be a list[str]")
        return {
            **self.evidence,
            "modality": self.modality,
            "source": self.source,
            "feature_dim": int(self.features.numel()),
            "feature_norm": feature_norm,
            "instruments": ["multimodal_perception", *raw_instruments],
        }
