"""Logit bias state model."""

from __future__ import annotations

import torch
from pydantic import BaseModel, ConfigDict


class LogitBiasState(BaseModel):
    """Validated state needed by the substrate logit-bias graft."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="ignore")

    broca_logit_bias: dict[int, float]
    broca_logit_bias_decay: float
    substrate_confidence: float
    substrate_inertia: float
    substrate_target_snr_scale: float
    last_indices: torch.Tensor

    @property
    def has_bias(self) -> bool:
        return bool(self.broca_logit_bias)

    @property
    def active_scale(self) -> bool:
        return float(self.substrate_target_snr_scale) > 0.0

    @property
    def active_decay(self) -> bool:
        return float(self.broca_logit_bias_decay) > 0.0

    def last_indices_on(self, device: torch.device) -> torch.Tensor:
        return self.last_indices.to(device)
