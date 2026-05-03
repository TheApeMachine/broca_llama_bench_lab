"""TrainableFeatureGraft — projects continuous frame features into the residual stream.

The host transformer stays frozen. This module learns how to project a
faculty-state vector plus a production-step embedding into ``d_model``-space
so the substrate's continuous cognitive frame influences the host's hidden
state at the last position of the prefix.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..grafting.grafts import (
    BaseGraft,
    DEFAULT_GRAFT_TARGET_SNR,
    snr_magnitude,
    state_confidence,
    state_inertia,
    state_target_snr_scale,
)


class TrainableFeatureGraft(BaseGraft):
    """Trainable bridge from latent cognitive frames to language tokens."""

    def __init__(
        self,
        d_features: int,
        d_model: int,
        *,
        max_steps: int = 10,
        step_dim: int = 16,
        hidden: int = 160,
        target_snr: float = DEFAULT_GRAFT_TARGET_SNR,
    ):
        super().__init__()
        self.d_features = int(d_features)
        self.max_steps = int(max_steps)
        self.target_snr = float(target_snr)
        self.mixer_priority = 0.35
        self.norm = nn.LayerNorm(d_features)
        self.step_emb = nn.Embedding(max_steps, step_dim)
        self.net = nn.Sequential(
            nn.Linear(d_features + step_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        nn.init.normal_(self.net[0].weight, std=0.02)
        nn.init.zeros_(self.net[0].bias)
        nn.init.normal_(self.net[2].weight, std=0.02)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled or "broca_features" not in state:
            return x
        
        feats = state["broca_features"]
        
        if not isinstance(feats, torch.Tensor):
            feats = torch.tensor(feats, device=x.device, dtype=x.dtype)
        
        param_dtype = self.norm.weight.dtype
        feats = feats.to(x.device, param_dtype)
        
        if feats.ndim == 1:
            feats = feats.view(1, -1).expand(x.shape[0], -1)
        
        if feats.shape[-1] != self.d_features:
            raise ValueError(
                f"expected broca_features dim {self.d_features}, got {feats.shape[-1]}"
            )
        
        step = state.get("broca_step", torch.zeros(x.shape[0], device=x.device, dtype=torch.long))
        
        if not isinstance(step, torch.Tensor):
            step = torch.full((x.shape[0],), int(step), device=x.device, dtype=torch.long)
        
        step = step.to(x.device).long().view(-1).clamp(0, self.max_steps - 1)
        
        z = torch.cat(
            [self.norm(feats), self.step_emb(step).to(device=x.device, dtype=param_dtype)],
            dim=-1,
        )
        
        last_raw = state.get("last_indices")
        
        if last_raw is None:
            raise ValueError(
                "TrainableFeatureGraft.forward: missing required state key 'last_indices'"
            )
        
        last = last_raw.to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)
        host_at_last = x[rows, last]
        direction = F.normalize(self.net(z).to(device=x.device, dtype=x.dtype), dim=-1)
        
        magnitude = snr_magnitude(
            host_at_last,
            target_snr=self.target_snr,
            confidence=state_confidence(state),
            inertia=state_inertia(state),
            substrate_scale=state_target_snr_scale(state),
        )
        
        out = x.clone()
        out[rows, last] += direction * magnitude
        
        return out
