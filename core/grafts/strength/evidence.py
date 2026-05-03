"""Evidence model for derived graft strength."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ...frame import CognitiveFrame


class StrengthEvidence(BaseModel):
    """Frame evidence needed to derive graft strength."""

    model_config = ConfigDict(extra="ignore")

    is_actionable: bool
    conformal_set_size: int = 0

    @classmethod
    def from_frame(cls, frame: CognitiveFrame) -> "StrengthEvidence":
        payload = dict(frame.evidence or {})

        payload["is_actionable"] = bool(
            payload.get("is_actionable", frame.intent != "unknown")
        )
        
        payload["conformal_set_size"] = int(payload.get("conformal_set_size", 0) or 0)
        
        return cls(**payload)

    @property
    def actionability(self) -> float:
        return 1.0 if self.is_actionable else 0.0
