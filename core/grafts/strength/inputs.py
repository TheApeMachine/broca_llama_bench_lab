"""Inputs for derived graft strength."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel

from ...affect.evidence import AffectEvidence
from ...frame import CognitiveFrame
from ...numeric import Probability


class StrengthInputs(BaseModel):
    """Signals composed into one graft strength."""

    probability: ClassVar[Probability] = Probability()

    intent_actionability: float
    memory_confidence: float
    conformal_set_size: int
    affect_certainty: float

    @classmethod
    def from_frame(cls, frame: CognitiveFrame, affect: Any) -> "StrengthInputs":
        evidence = dict(frame.evidence or {})
        is_actionable = bool(evidence.get("is_actionable", frame.intent != "unknown"))
        return cls(
            intent_actionability=1.0 if is_actionable else 0.0,
            memory_confidence=cls.probability.unit_interval(frame.confidence),
            conformal_set_size=int(evidence.get("conformal_set_size", 0) or 0),
            affect_certainty=AffectEvidence.certainty(affect),
        )
