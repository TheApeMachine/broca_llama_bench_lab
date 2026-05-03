"""Chat graft plan model."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict

from ..frame import CognitiveFrame


class ChatGraftPlan(BaseModel):
    """Graft and decode settings for one generated chat reply.

    Concepts cross the frozen-LLM boundary as continuous directions, not as
    final-layer logit edits. ``concept_token_ids`` and ``repulsion_token_ids``
    name the lm_head rows whose bundled directions the SubstrateConceptGraft
    adds (attraction) and subtracts (repulsion) at the last residual position.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    frame: CognitiveFrame
    confidence: float
    effective_temperature: float
    broca_features: torch.Tensor | None
    concept_token_ids: dict[str, list[int]]
    repulsion_token_ids: dict[str, list[int]]
    concept_preview: list[dict[str, int | str | float]]
    derived_target_snr_scale: float

    @property
    def has_broca_features(self) -> bool:
        return self.broca_features is not None

    @property
    def has_concepts(self) -> bool:
        return bool(self.concept_token_ids) or bool(self.repulsion_token_ids)

    def start_metadata(self, *, timestamp: float) -> dict:
        return {
            "intent": self.frame.intent,
            "subject": self.frame.subject,
            "answer": self.frame.answer,
            "confidence": float(self.confidence),
            "eff_temperature": float(self.effective_temperature),
            "concept_count": len(self.concept_token_ids),
            "repulsion_count": len(self.repulsion_token_ids),
            "concept_preview": [dict(item) for item in self.concept_preview],
            "has_broca_features": self.has_broca_features,
            "derived_target_snr_scale": float(self.derived_target_snr_scale),
            "ts": float(timestamp),
        }
