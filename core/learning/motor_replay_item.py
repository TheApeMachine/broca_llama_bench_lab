"""Motor replay item model."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class MotorReplayItem(BaseModel):
    """Persistable replay item for motor learning."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[dict[str, str]]
    generated_token_ids: list[int]
    substrate_confidence: float
    substrate_inertia: float
    broca_features: torch.Tensor | None = None

    def to_replay_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [dict(message) for message in self.messages],
            "speech_plan_tokens": torch.tensor(
                list(self.generated_token_ids),
                dtype=torch.long,
            ),
            "substrate_confidence": float(self.substrate_confidence),
            "substrate_inertia": float(self.substrate_inertia),
        }
        if self.broca_features is not None:
            payload["broca_features"] = self.broca_features.detach().cpu().clone()
        return payload
