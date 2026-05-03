"""Chat graft plan model."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

from ..frame import CognitiveFrame
from .token_bias import TokenBias


class ChatGraftPlan(BaseModel):
    """Graft and decode settings for one generated chat reply."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frame: CognitiveFrame
    confidence: float
    effective_temperature: float
    broca_features: torch.Tensor | None
    logit_bias: dict[int, float]
    bias_top: list[TokenBias]
    derived_target_snr_scale: float

    @property
    def has_broca_features(self) -> bool:
        return self.broca_features is not None

    def start_metadata(self, *, timestamp: float) -> dict:
        return {
            "intent": self.frame.intent,
            "subject": self.frame.subject,
            "answer": self.frame.answer,
            "confidence": float(self.confidence),
            "eff_temperature": float(self.effective_temperature),
            "bias_token_count": len(self.logit_bias),
            "bias_top": [item.model_dump() for item in self.bias_top],
            "has_broca_features": self.has_broca_features,
            "derived_target_snr_scale": float(self.derived_target_snr_scale),
            "ts": float(timestamp),
        }
