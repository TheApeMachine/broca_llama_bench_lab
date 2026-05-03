"""SubstrateLogitBiasGraft — context-aware logit push on substrate-supplied tokens.

The substrate supplies the set of token ids it wants to surface; the graft
itself derives the actual push at every step from the host's current logit
distribution so the bias is meaningful regardless of how confident the LLM
happens to be.

State keys it reads (``broca_logit_bias`` is required, the rest are optional):

* ``broca_logit_bias`` — ``Mapping[int, float]`` of base nat-scale bonuses.
* ``broca_logit_bias_decay`` — semantic multiplier (typically ``1.0`` until
  the target concept has appeared in the prefix, then collapses).
* ``substrate_confidence`` — scalar in ``[0, 1]`` from the substrate frame.
* ``substrate_inertia`` — ``log1p(prefix_length)`` ; how much momentum the
  LLM has built up that the bias must shout over.

The dynamic push has two parts:

1. ``target_boost = max(0, max_logit - target_logit) * confidence`` — drag
   the target up to (and past) the current top logit, scaled by how strongly
   the substrate believes in the answer.
2. ``stubborn_push = stubbornness * base_bonus`` — augment with a structural
   bonus where ``stubbornness`` is the normalized peakedness of the
   distribution (1.0 when the LLM is a delta, ≈0 when it is uniform). A
   confident-but-wrong LLM gets a bigger nudge than an indecisive one.

The combined value is then scaled by ``decay`` (semantic) and ``inertia``
(sequence-length) so the bias keeps up with autoregressive momentum until
the target concept actually appears.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..grafting.grafts import (
    BaseGraft,
    state_confidence,
    state_inertia,
    state_target_snr_scale,
)


class SubstrateLogitBiasGraft(BaseGraft):
    """Dynamic, context-aware logit bias on substrate-supplied vocabulary IDs."""

    def __init__(self):
        super().__init__()
        self.mixer_priority = 0.5

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x

        bias = state.get("broca_logit_bias")
        
        if not bias:
            return x
        
        substrate_scale = state_target_snr_scale(state)
        
        if substrate_scale <= 0.0:
            return x
        
        decay_raw = state.get("broca_logit_bias_decay", 1.0)
        
        try:
            decay = float(decay_raw)
        except (TypeError, ValueError):
            decay = 1.0
        
        if decay <= 0.0:
            return x

        confidence = max(0.0, min(1.0, float(state_confidence(state))))
        inertia = max(float(state_inertia(state)), 1e-6)

        last_raw = state.get("last_indices")
        
        if last_raw is None:
            raise ValueError(
                "SubstrateLogitBiasGraft.forward: missing required state key 'last_indices'"
            )
        
        last = last_raw.to(x.device)
        rows = torch.arange(x.shape[0], device=x.device)

        out = x.clone()
        last_logits = out[rows, last].float()
        max_logit = last_logits.max(dim=-1, keepdim=True).values
        log_probs = F.log_softmax(last_logits, dim=-1)
        probs = log_probs.exp()
        entropy_val = -(probs * log_probs).sum(dim=-1)
        log_vocab = math.log(max(2, last_logits.shape[-1]))
        
        # peakedness ∈ [0, 1]: 1.0 when distribution is a delta, ~0 when uniform
        stubbornness = (1.0 - entropy_val / log_vocab).clamp(0.0, 1.0).unsqueeze(-1)

        for token_id, bonus in bias.items():
            tid = int(token_id)
        
            if tid < 0 or tid >= out.shape[-1]:
                continue
        
            cur = last_logits[:, tid : tid + 1]
            target_boost = (max_logit - cur).clamp_min(0.0) * confidence
            stubborn_push = stubbornness * float(bonus)
        
            delta = (
                (target_boost + stubborn_push) * decay * inertia * substrate_scale
            ).to(out.dtype)
        
            out[rows, last, tid] = out[rows, last, tid] + delta.squeeze(-1)
        
        return out
