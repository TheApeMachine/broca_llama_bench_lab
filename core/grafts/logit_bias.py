"""SubstrateLogitBiasGraft — context-aware logit push on substrate-supplied tokens.

The substrate supplies the set of token ids it wants to surface; the graft
itself derives the actual push at every step from the host's current logit
distribution so the bias is meaningful regardless of how confident the LLM
happens to be.

State keys it validates when ``broca_logit_bias`` is present:

* ``broca_logit_bias`` — ``Mapping[int, float]`` of base nat-scale bonuses.
* ``broca_logit_bias_decay`` — semantic multiplier (typically ``1.0`` until
  the target concept has appeared in the prefix, then collapses).
* ``substrate_confidence`` — scalar in ``[0, 1]`` from the substrate frame.
* ``substrate_inertia`` — ``log1p(prefix_length)`` ; how much momentum the
  LLM has built up that the bias must shout over.
* ``substrate_target_snr_scale`` — derived graft strength scalar.
* ``last_indices`` — batch row positions where the bias applies.

The dynamic push has two parts:

1. ``target_boost = max(0, max_logit - target_logit) * confidence`` — drag
   the target up to (and past) the current top logit, scaled by how strongly
   the substrate believes in the answer.
2. ``stubborn_push = stubbornness * base_bonus`` — augment with a structural
   bonus where ``stubbornness`` is the normalized peakedness of the
   distribution (1.0 when the LLM is a delta, near 0 when it is uniform). A
   confident-but-wrong LLM gets a bigger nudge than an indecisive one.

The combined value is then scaled by ``decay`` (semantic) and ``inertia``
(sequence-length) so the bias keeps up with autoregressive momentum until
the target concept actually appears.
"""

from __future__ import annotations

import torch

from ..grafting.grafts import BaseGraft
from ..numeric import Probability, TensorDistribution
from .logit_bias_state import LogitBiasState


class SubstrateLogitBiasGraft(BaseGraft):
    """Dynamic, context-aware logit bias on substrate-supplied vocabulary IDs."""

    def __init__(self):
        super().__init__()
        self.mixer_priority = 0.5
        self.probability = Probability()
        self.distribution = TensorDistribution()

    def forward(self, x: torch.Tensor, state: dict) -> torch.Tensor:
        if not self.enabled:
            return x

        if not state.get("broca_logit_bias"):
            return x

        bias_state = LogitBiasState(**state)

        if not bias_state.has_bias:
            return x

        if not bias_state.active_scale:
            return x

        if not bias_state.active_decay:
            return x

        confidence = self.probability.unit_interval(bias_state.substrate_confidence)
        inertia = max(float(bias_state.substrate_inertia), 1e-6)
        last = bias_state.last_indices_on(x.device)
        rows = torch.arange(x.shape[0], device=x.device)

        out = x.clone()
        last_logits = out[rows, last].float()
        max_logit = last_logits.max(dim=-1, keepdim=True).values
        stubbornness = self.distribution.peakedness(last_logits).unsqueeze(-1)

        for token_id, bonus in bias_state.broca_logit_bias.items():
            tid = int(token_id)

            if tid < 0 or tid >= out.shape[-1]:
                continue

            cur = last_logits[:, tid : tid + 1]
            target_boost = (max_logit - cur).clamp_min(0.0) * confidence
            stubborn_push = stubbornness * float(bonus)

            delta = (
                (target_boost + stubborn_push)
                * float(bias_state.broca_logit_bias_decay)
                * inertia
                * float(bias_state.substrate_target_snr_scale)
            ).to(out.dtype)

            out[rows, last, tid] = out[rows, last, tid] + delta.squeeze(-1)

        return out
