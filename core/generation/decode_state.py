"""Host decode state model."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, model_validator


class DecodeState(BaseModel):
    """Extra state passed into the host during one decode step."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    tokenizer: Any
    substrate_confidence: float
    substrate_inertia: float
    substrate_target_snr_scale: float
    return_past_key_values: bool = True
    broca_features: torch.Tensor | None = None
    broca_logit_bias: dict[int, float] | None = None
    broca_logit_bias_decay: float | None = None
    past_key_values: Any = None

    @model_validator(mode="after")
    def validate_logit_bias_state(self) -> "DecodeState":
        if self.broca_logit_bias and self.broca_logit_bias_decay is None:
            raise ValueError("DecodeState requires broca_logit_bias_decay with bias")
        if self.broca_logit_bias is None and self.broca_logit_bias_decay is not None:
            raise ValueError("DecodeState cannot decay absent broca_logit_bias")
        return self

    def to_extra_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tokenizer": self.tokenizer,
            "substrate_confidence": float(self.substrate_confidence),
            "substrate_inertia": float(self.substrate_inertia),
            "substrate_target_snr_scale": float(self.substrate_target_snr_scale),
            "return_past_key_values": bool(self.return_past_key_values),
        }
        if self.broca_features is not None:
            payload["broca_features"] = self.broca_features
        if self.broca_logit_bias is not None:
            payload["broca_logit_bias"] = dict(self.broca_logit_bias)
        if self.broca_logit_bias_decay is not None:
            payload["broca_logit_bias_decay"] = float(self.broca_logit_bias_decay)
        if self.past_key_values is not None:
            payload["past_key_values"] = self.past_key_values
        return payload
